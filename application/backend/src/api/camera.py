import asyncio
import contextlib
import json
import time
from collections import defaultdict
from collections.abc import Coroutine
from functools import cache
from typing import Annotated, Any, Literal

import anyio
from fastapi import APIRouter, Depends, Query, WebSocket, status
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from physicalai.capture import SharedCamera
from physicalai.capture.errors import CaptureError

from api.dependencies import CameraClaimRegistryDep
from exceptions import DeviceInUseError
from schemas.camera import SupportedCameraFormat
from schemas.project_camera import Camera as ProjectCamera
from schemas.project_camera import CameraAdapter
from schemas.robot_stream import ErrorEvent
from utils.camera_factory import build_shared_camera
from workers.rate import Rate

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])

RealsenseOutputType = Literal["color", "depth", "both"]


def _formats_to_response(raw: list[tuple[int, int, int]]) -> list[SupportedCameraFormat]:
    """Group (w, h, fps) tuples into SupportedCameraFormat responses."""
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for w, h, fps in raw:
        grouped[(w, h)].append(fps)

    return [
        SupportedCameraFormat(width=w, height=h, fps=sorted(fps_list)) for (w, h), fps_list in sorted(grouped.items())
    ]


# Cache required on MacOS to avoid repeated cam.open() which may block camera stream
@cache
def _query_formats(driver: str, fingerprint_key: str) -> list[SupportedCameraFormat]:
    """Query real formats from a device via physicalai.capture."""

    if driver == "basler":
        # TODO: Replace with cached Basler hardware discovery once implementated in physicalai.capture
        return _formats_to_response([(640, 480, 30), (768, 480, 30), (1920, 1200, 30)])

    try:
        deserialized_fingerprint = json.loads(fingerprint_key)
    except json.JSONDecodeError as e:
        msg = f"Unable to parse fingerprint into a dictionary: {e}"
        raise ValueError(msg)

    if driver == "usb_camera":
        from physicalai.capture import UVCCamera

        return _formats_to_response(UVCCamera.query_formats(deserialized_fingerprint))

    msg = f"Format discovery not supported for driver {driver!r}"
    raise ValueError(msg)


def _query_realsense_formats(fingerprint: dict, output_type: RealsenseOutputType) -> list[SupportedCameraFormat]:
    """The modes a RealSense offers for ``output_type`` on its current USB connection.

    Not cached: listing them does not open the camera, and they change when it is replugged
    into another port (USB 2 offers far fewer).
    """
    from utils.realsense_rgbd import RealSenseRGBDCamera

    return _formats_to_response(RealSenseRGBDCamera.query_formats(fingerprint, output_type))


@router.get("/supported_formats/{driver}")
async def get_supported_formats(
    driver: str,
    fingerprint: str,
    output_type: Annotated[
        RealsenseOutputType,
        Query(description="RealSense only: the modes for color, depth, or color with depth aligned to it"),
    ] = "color",
) -> list[SupportedCameraFormat]:
    """Returns the supported camera resolution and fps associated to the camera."""
    try:
        parsed = json.loads(fingerprint)
    except json.JSONDecodeError as e:
        raise ValueError(f"Unable to parse fingerprint into a dictionary: {e}") from e
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("Camera fingerprint must be a non-empty JSON object")
    if driver == "realsense":
        return await asyncio.to_thread(_query_realsense_formats, parsed, output_type)
    fingerprint_key = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return _query_formats(driver, fingerprint_key)


def get_camera_from_query(websocket: WebSocket) -> ProjectCamera:
    """Parse camera from query parameters."""
    camera_param = websocket.query_params.get("camera")
    if not camera_param:
        raise ValueError("Missing 'camera' query parameter")

    try:
        camera_data = json.loads(camera_param)
        return CameraAdapter.validate_python(camera_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in camera parameter: {e}")
    except Exception as e:
        raise ValueError(f"Invalid camera configuration: {e}")


@router.get("/ws", tags=["WebSocket"], summary="Camera streaming (WebSocket)", status_code=426)
async def camera_websocket_openapi(
    camera: Annotated[str | None, Query(description="JSON-serialized ProjectCamera configuration")] = None,  # noqa: ARG001
) -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/ws")
async def camera_websocket(
    websocket: WebSocket,
    claims: CameraClaimRegistryDep,
    camera: Annotated[ProjectCamera, Depends(get_camera_from_query)],
) -> None:
    """
    WebSocket endpoint for camera streaming.

    Query Parameters:
        camera: JSON serialized ProjectCamera

    Protocol:
        Client sends JSON message on connect:
            camera: JSON serialized ProjectCamera

        Server sends jpeg encoded bytes, or one JSON error before closing:
            {"event": "error", "message": str, "error_code": str}
    """
    await websocket.accept()

    try:
        if camera.fingerprint is None:
            raise ValueError("Camera must be reselected")
        if (claim := claims.holder_of(camera.fingerprint)) is not None:
            raise DeviceInUseError(
                device=f"Camera {camera.name!r}", holder=f"a runtime session in project {claim.project_name!r}"
            )
        # SharedCamera already captures in its own publisher process; a preview needs no worker.
        # Another resolution than the running publisher's replaces it: the preview follows the form.
        shared_camera = build_shared_camera(config=camera, validate_on_connect=False, overwrite_settings=True)
        await asyncio.to_thread(shared_camera.connect)
        try:
            await _until_closed_or_failed(websocket, _stream_preview(websocket, shared_camera, camera))
        finally:
            # Shielded: a cancelled socket task must still detach from the publisher.
            with anyio.CancelScope(shield=True):
                await asyncio.to_thread(shared_camera.disconnect)
    except WebSocketDisconnect:
        pass
    except DeviceInUseError as error:
        await websocket.send_json(ErrorEvent(message=error.message, error_code=error.error_code).model_dump())
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    except (CaptureError, ValueError) as error:
        # Such as a RealSense mode its USB connection does not offer: say so, rather than just drop the socket.
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await websocket.send_json(ErrorEvent(message=str(error), error_code="camera_unavailable").model_dump())
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)


_PREVIEW_STALL_S = 5.0
"""How long a preview waits for a new frame before it reports the camera stalled and closes."""


async def _stream_preview(websocket: WebSocket, shared_camera: SharedCamera, camera: ProjectCamera) -> None:
    """Send every new frame as JPEG at the camera's rate.

    Raises:
        CaptureError: The camera sent nothing new for ``_PREVIEW_STALL_S``; reconnecting starts a fresh publisher.
    """
    from utils.jpeg import encode_jpeg_camera_frame

    rate = Rate(camera.payload.fps, name=f"camera-preview-{camera.name}")
    sent: int | None = None
    last_new_frame = time.monotonic()
    while True:
        frame = await asyncio.to_thread(shared_camera.read_latest)
        if frame.sequence != sent:
            await websocket.send_bytes(await asyncio.to_thread(encode_jpeg_camera_frame, frame.data))
            sent, last_new_frame = frame.sequence, time.monotonic()
        elif time.monotonic() - last_new_frame > _PREVIEW_STALL_S:
            raise CaptureError(f"Camera {camera.name!r} stopped sending frames")
        await rate.sleep()


async def _until_closed_or_failed(websocket: WebSocket, stream: Coroutine[Any, Any, None]) -> None:
    """Run ``stream`` until it ends or the socket closes, whichever is first.

    The stream only sends, so without watching the socket a preview whose camera sends nothing
    new would outlive its socket, keep its subscriber attached, and hold up the server's
    graceful shutdown (uvicorn waits for every connection's task) forever.

    Raises:
        Exception: What ``stream`` raised.
    """
    streaming = asyncio.ensure_future(stream)
    watching = asyncio.ensure_future(_until_disconnect(websocket))
    try:
        done, _pending = await asyncio.wait({streaming, watching}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (streaming, watching):
            task.cancel()
        await asyncio.gather(streaming, watching, return_exceptions=True)
    if streaming in done:
        streaming.result()


async def _until_disconnect(websocket: WebSocket) -> None:
    """Return once the client disconnects; anything it sends is ignored."""
    while (await websocket.receive())["type"] != "websocket.disconnect":
        pass
