"""Factory for building camera instances from backend camera configs.

Maps backend driver names to physicalai.capture Config recipes and
filters per-driver kwargs so that only constructor-safe parameters reach the camera.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from physicalai.capture import CameraType, ColorMode, SharedCamera
from physicalai.config import Config

from utils.v4l2_frame_interval import fix_zero_frame_intervals

if TYPE_CHECKING:
    from schemas.project_camera import Camera

MIGRATED_DRIVERS: frozenset[str] = frozenset({"usb_camera", "realsense", "basler", "ipcam"})

DRIVER_KEY_MAP: dict[str, str] = {
    "uvc": "usb_camera",
    "realsense": "realsense",
    "basler": "basler",
    "ipcam": "ipcam",
}

_DRIVER_TO_CAMERA_TYPE: dict[str, CameraType] = {
    "usb_camera": CameraType.UVC,
    "realsense": CameraType.REALSENSE,
    "basler": CameraType.BASLER,
    "ipcam": CameraType.IP,
}

_DRIVER_TO_CLASS_PATH: dict[str, str] = {
    "usb_camera": "physicalai.capture.UVCCamera",
    "realsense": "physicalai.capture.RealSenseCamera",
    "basler": "physicalai.capture.BaslerCamera",
    "ipcam": "physicalai.capture.IPCamera",
}

# What Studio's own publishers capture with, where it differs from the portable class above:
# physicalai's RealSenseCamera streams depth at the color resolution and publishes color only.
_DRIVER_TO_CAPTURE_CLASS_PATH: dict[str, str] = {
    "realsense": "utils.realsense_rgbd.RealSenseRGBDCamera",
}

# Depth resolutions of Intel RealSense D400 cameras, for a portable recipe that has to pick one blindly.
_D400_DEPTH_SIZES: frozenset[tuple[int, int]] = frozenset(
    {(1280, 720), (848, 480), (640, 480), (640, 360), (480, 270), (424, 240)}
)

# Per-driver kwargs that are safe to pass through to the nested camera recipe.
_ALLOWED_KWARGS: dict[str, frozenset[str]] = {
    "usb_camera": frozenset({"width", "height", "fps"}),
    "realsense": frozenset({"width", "height", "fps"}),
    "basler": frozenset({"width", "height", "fps"}),
    "ipcam": frozenset({"width", "height", "fps", "url"}),
}


def build_camera_config(config: Camera) -> Config:
    """Build the portable ``physicalai.capture`` recipe for a camera, as a runtime export runs it.

    Studio captures some drivers with its own class instead; see :func:`build_capture_config`.
    """
    class_path = _DRIVER_TO_CLASS_PATH[config.driver]
    allowed = _ALLOWED_KWARGS.get(config.driver, frozenset())

    payload = config.payload.model_dump()
    init_args: dict[str, Any] = {k: v for k, v in payload.items() if k in allowed and v is not None}

    fingerprint = config.fingerprint
    if not fingerprint:
        raise ValueError("Camera must be reselected")

    if config.driver == "usb_camera":
        init_args["device"] = fingerprint
    elif config.driver != "ipcam":
        serial = fingerprint.get("serial")
        if not isinstance(serial, str) or not serial:
            raise ValueError(f"Camera fingerprint for {config.driver!r} must include a non-empty 'serial'")
        init_args["serial_number"] = serial

    if config.driver == "realsense":
        # physicalai's RealSenseCamera always streams depth too, by default at the color resolution,
        # which the depth sensor may lack (1920x1080, 960x540): give it one it has.
        width, height = init_args.get("width", 640), init_args.get("height", 480)
        depth = (
            (width, height) if (width, height) in _D400_DEPTH_SIZES else (1280, 720) if width >= 1280 else (848, 480)
        )
        init_args["depth_width"], init_args["depth_height"] = depth

    return Config(class_path, init_args)


def build_capture_config(config: Camera) -> Config:
    """Build the recipe Studio's camera publisher captures with.

    A RealSense camera is captured with :class:`utils.realsense_rgbd.RealSenseRGBDCamera`, which
    streams what ``output_type`` asks for (color, depth aligned to color, or both) and fails with
    the modes the device offers on its current USB connection when the requested one is not among
    them. Every other driver uses its portable recipe.
    """
    if config.driver != "realsense":
        return build_camera_config(config)
    payload = config.payload
    serial = (config.fingerprint or {}).get("serial")
    if not config.fingerprint:
        raise ValueError("Camera must be reselected")
    if not isinstance(serial, str) or not serial:
        raise ValueError("Camera fingerprint for 'realsense' must include a non-empty 'serial'")
    return Config(
        _DRIVER_TO_CAPTURE_CLASS_PATH["realsense"],
        {
            "serial_number": serial,
            "width": payload.width,
            "height": payload.height,
            "fps": payload.fps,
            "output_type": payload.output_type,
        },
    )


def capture_service_name(config: Camera) -> str | None:
    """The shared-memory service a camera's publisher serves, when it is not the derived default.

    A RealSense publisher delivers only the streams of its ``output_type``, so each output type
    gets its own service: a subscriber never attaches to a publisher that lacks the streams it needs.
    """
    if config.driver != "realsense":
        return None
    serial = (config.fingerprint or {}).get("serial")
    return f"physicalai/camera/RealSenseRGBDCamera/{serial}-{config.payload.output_type}/frame"


def is_migrated(driver: str) -> bool:
    """Return True if *driver* is supported by physicalai.capture."""
    return driver in MIGRATED_DRIVERS


def driver_to_camera_type(driver: str) -> CameraType:
    """Convert a backend driver name to a CameraType enum value."""
    try:
        return _DRIVER_TO_CAMERA_TYPE[driver]
    except KeyError:
        msg = f"unsupported driver {driver!r}; expected one of {sorted(MIGRATED_DRIVERS)}"
        raise ValueError(msg) from None


def build_shared_camera(
    config: Camera,
    *,
    validate_on_connect: bool = False,
    overwrite_settings: bool = False,
    idle_timeout: float = 5.0,
) -> SharedCamera:
    """Build a SharedCamera from a backend Camera schema.

    Args:
        config: Backend camera configuration (discriminated union).
        validate_on_connect: If ``True``, :meth:`~SharedCamera.connect` raises
            :class:`~physicalai.capture.errors.CaptureError` when an
            existing publisher's resolution does not match the requested
            ``width``/``height``. Use ``False`` for preview streams
            (tolerates mismatch) and ``True`` for recording / inference
            when the initial attachment must match the requested config.
        overwrite_settings: If ``True``, attempt to reconfigure the publisher
            to match requested settings when a config mismatch is detected.
            Requires a publisher that supports the control channel (v2+).
        idle_timeout: Seconds with zero subscribers before the publisher
            self-exits.  Preview-class callers should use a short value
            (e.g. 0.5) for fast turnover on resolution changes;
            record-class callers should use a longer value (e.g. 5.0).

    Returns:
        A configured (but not yet connected) SharedCamera instance.
    """
    if config.driver not in _DRIVER_TO_CLASS_PATH:
        msg = f"unsupported driver {config.driver!r}; expected one of {sorted(MIGRATED_DRIVERS)}"
        raise ValueError(msg)

    if config.driver == "usb_camera":
        # A replugged UVC camera may report a 1/0 frame interval, which fails the open.
        fix_zero_frame_intervals()

    camera_config = build_capture_config(config)
    logger.debug(f"camera config for {config.name}: {camera_config}")

    return SharedCamera(
        camera=camera_config,
        service_name=capture_service_name(config),
        color_mode=ColorMode.RGB,
        validate_on_connect=validate_on_connect,
        overwrite_settings=overwrite_settings,
        idle_timeout=idle_timeout,
    )
