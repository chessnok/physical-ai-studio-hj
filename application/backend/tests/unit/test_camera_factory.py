"""What Studio captures a camera with, and what it exports for the runtime."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.project_camera import CameraAdapter
from utils.camera_factory import build_camera_config, build_capture_config, build_shared_camera, capture_service_name


def _realsense(output_type: str = "both", width: int = 1280, height: int = 720, fps: int = 30, **payload):
    return CameraAdapter.validate_python(
        {
            "driver": "realsense",
            "name": "top realsense",
            "fingerprint": {"serial": "243622062352"},
            "hardware_name": "Intel RealSense D415",
            "payload": {"width": width, "height": height, "fps": fps, "output_type": output_type, **payload},
        }
    )


def _usb():
    return CameraAdapter.validate_python(
        {
            "driver": "usb_camera",
            "name": "arm",
            "fingerprint": {"path": "/dev/video4"},
            "hardware_name": None,
            "payload": {"width": 1920, "height": 1080, "fps": 30},
        }
    )


@pytest.mark.parametrize("output_type", ["color", "depth", "both"])
def test_realsense_is_captured_with_studio_rgbd_camera_and_its_output_type(output_type):
    config = build_capture_config(_realsense(output_type))
    assert config.class_path == "utils.realsense_rgbd.RealSenseRGBDCamera"
    assert config.init_args == {
        "serial_number": "243622062352",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "output_type": output_type,
    }


def test_each_realsense_output_type_has_its_own_publisher_service():
    names = {capture_service_name(_realsense(output_type)) for output_type in ("color", "depth", "both")}
    assert len(names) == 3
    assert capture_service_name(_realsense("both")) == "physicalai/camera/RealSenseRGBDCamera/243622062352-both/frame"
    shared = build_shared_camera(_realsense("both"))
    assert shared.service_name == capture_service_name(_realsense("both"))
    assert shared.device_id == "243622062352-both"


def test_other_drivers_keep_their_portable_recipe_and_derived_service():
    camera = _usb()
    assert build_capture_config(camera) == build_camera_config(camera)
    assert capture_service_name(camera) is None


@pytest.mark.parametrize(
    ("width", "height", "depth"),
    [((1920, 1080, (1280, 720))), ((1280, 720, (1280, 720))), ((960, 540, (848, 480))), ((640, 480, (640, 480)))],
)
def test_runtime_export_gives_physicalai_realsense_a_depth_mode_it_has(width, height, depth):
    # physicalai's RealSenseCamera also streams depth, by default at the color resolution.
    config = build_camera_config(_realsense("color", width, height, 15))
    assert config.class_path == "physicalai.capture.RealSenseCamera"
    assert (config.init_args["depth_width"], config.init_args["depth_height"]) == depth
    assert "output_type" not in config.init_args


def test_realsense_without_a_serial_must_be_reselected():
    camera = _realsense()
    camera.fingerprint = None
    with pytest.raises(ValueError, match="reselected"):
        build_capture_config(camera)


def test_depth_range_must_keep_some_depth():
    with pytest.raises(ValidationError, match="depth_range_min"):
        _realsense(depth_range_min=3.0, depth_range_max=2.0)


def test_realsense_payload_defaults_stay_color_only():
    camera = CameraAdapter.validate_python(
        {
            "driver": "realsense",
            "name": "top",
            "fingerprint": {"serial": "1"},
            "hardware_name": None,
            "payload": {"width": 1280, "height": 720, "fps": 30},
        }
    )
    assert camera.payload.output_type == "color"
    assert (camera.payload.depth_range_min, camera.payload.depth_range_max) == (0.3, 3.0)
