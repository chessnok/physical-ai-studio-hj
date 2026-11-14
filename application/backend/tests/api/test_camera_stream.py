from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import quote
from uuid import uuid4

from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from api.dependencies import get_camera_claim_registry, get_scheduler
from main import app
from services.camera_claims import CameraClaim, CameraClaimRegistry


def _camera_query(*, fingerprint: dict[str, str] | None = None) -> str:
    return json.dumps(
        {
            "driver": "usb_camera",
            "name": "front",
            "fingerprint": fingerprint or {"serial": "cam-front"},
            "hardware_name": None,
            "payload": {"width": 640, "height": 480, "fps": 30},
        }
    )


def _client(claims: CameraClaimRegistry) -> TestClient:
    scheduler = MagicMock()
    app.dependency_overrides[get_scheduler] = lambda: scheduler
    app.dependency_overrides[get_camera_claim_registry] = lambda: claims
    return TestClient(app)


def test_stream_refuses_a_camera_a_session_holds() -> None:
    claims = CameraClaimRegistry()
    claims.claim(
        [
            CameraClaim(
                fingerprint={"serial": "cam-front"},
                settings=(640, 480, 30),
                holder="session-a",
                project_id=uuid4(),
                project_name="Alpha",
            )
        ]
    )
    client = _client(claims)
    try:
        with patch("api.camera.build_shared_camera") as build:
            with client.websocket_connect(f"/api/cameras/ws?camera={quote(_camera_query())}") as websocket:
                error = websocket.receive_json()
            build.assert_not_called()
        assert error["error_code"] == "device_in_use"
        assert "Alpha" in error["message"]
    finally:
        app.dependency_overrides.clear()


def test_stream_endpoint_overwrites_settings_of_an_unclaimed_camera() -> None:
    client = _client(CameraClaimRegistry())
    try:
        with (
            patch("api.camera.build_shared_camera") as build,
            patch("api.camera.Rate") as rate_cls,
            patch("utils.jpeg.encode_jpeg_rgb", return_value=b"jpeg"),
        ):
            rate_cls.return_value.sleep = AsyncMock(side_effect=WebSocketDisconnect())
            with client.websocket_connect(f"/api/cameras/ws?camera={quote(_camera_query())}"):
                pass
            assert build.call_args.kwargs["overwrite_settings"] is True
    finally:
        app.dependency_overrides.clear()


def test_supported_formats_of_a_realsense_follow_its_output_type() -> None:
    client = _client(CameraClaimRegistry())
    try:
        with patch(
            "utils.realsense_rgbd.RealSenseRGBDCamera.query_formats", return_value=[(1280, 720, 15), (1280, 720, 30)]
        ) as query:
            fingerprint = quote(json.dumps({"serial": "243622062352"}))
            response = client.get(
                f"/api/cameras/supported_formats/realsense?fingerprint={fingerprint}&output_type=both"
            )
            again = client.get(f"/api/cameras/supported_formats/realsense?fingerprint={fingerprint}")
        assert response.status_code == 200
        assert response.json() == [{"width": 1280, "height": 720, "fps": [15, 30]}]
        # Not cached: replugging into another USB port changes what the camera offers.
        assert [call.args for call in query.call_args_list] == [
            ({"serial": "243622062352"}, "both"),
            ({"serial": "243622062352"}, "color"),
        ]
        assert again.status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_a_realsense_preview_shows_color_or_else_colorized_depth() -> None:
    import numpy as np

    from utils.jpeg import encode_jpeg_camera_frame
    from utils.realsense_rgbd import pack_rgbd

    color = np.zeros((4, 6, 3), np.uint8)
    depth = np.full((4, 6), 1000, np.uint16)
    with patch("utils.jpeg.encode_jpeg_rgb", return_value=b"jpeg") as encode:
        encode_jpeg_camera_frame(pack_rgbd(color, depth))
        assert encode.call_args.args[0].shape == (4, 6, 3)
        assert encode.call_args.args[0] is not None
        encode_jpeg_camera_frame(pack_rgbd(None, depth))
        shown = encode.call_args.args[0]
        assert shown.shape == (4, 6, 3)
        assert shown.dtype == np.uint8


def _frozen_camera() -> MagicMock:
    """A camera whose publisher stopped delivering: ``read_latest`` keeps returning one frame."""
    import numpy as np

    frame = MagicMock()
    frame.data = np.zeros((4, 4, 3), np.uint8)
    frame.sequence = 1
    camera = MagicMock()
    camera.read_latest.return_value = frame
    return camera


def test_a_preview_ends_when_its_socket_closes_even_if_the_camera_sends_nothing() -> None:
    client = _client(CameraClaimRegistry())
    try:
        with (
            patch("api.camera.build_shared_camera") as build,
            patch("api.camera._PREVIEW_STALL_S", 60.0),
            patch("utils.jpeg.encode_jpeg_rgb", return_value=b"jpeg"),
        ):
            build.return_value = _frozen_camera()
            with client.websocket_connect(f"/api/cameras/ws?camera={quote(_camera_query())}") as websocket:
                assert websocket.receive_bytes() == b"jpeg"
            # Leaving the block closes the socket; the test client waits for the handler to end.
            build.return_value.disconnect.assert_called_once()
    finally:
        app.dependency_overrides.clear()


def test_a_stalled_camera_is_reported_so_the_preview_reconnects() -> None:
    client = _client(CameraClaimRegistry())
    try:
        with (
            patch("api.camera.build_shared_camera") as build,
            patch("api.camera._PREVIEW_STALL_S", 0.1),
            patch("utils.jpeg.encode_jpeg_rgb", return_value=b"jpeg"),
        ):
            build.return_value = _frozen_camera()
            with client.websocket_connect(f"/api/cameras/ws?camera={quote(_camera_query())}") as websocket:
                assert websocket.receive_bytes() == b"jpeg"
                error = websocket.receive_json()
            assert error["error_code"] == "camera_unavailable"
            assert "stopped sending frames" in error["message"]
            build.return_value.disconnect.assert_called_once()
    finally:
        app.dependency_overrides.clear()


def test_a_camera_that_cannot_open_says_why() -> None:
    from utils.realsense_rgbd import RealSenseModeError

    client = _client(CameraClaimRegistry())
    try:
        with patch("api.camera.build_shared_camera") as build:
            build.return_value.connect.side_effect = RealSenseModeError("no color mode 1280x720@30fps. ... USB 2.1")
            with client.websocket_connect(f"/api/cameras/ws?camera={quote(_camera_query())}") as websocket:
                error = websocket.receive_json()
        assert error["error_code"] == "camera_unavailable"
        assert "USB 2.1" in error["message"]
    finally:
        app.dependency_overrides.clear()
