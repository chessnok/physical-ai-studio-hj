"""Intel RealSense capture that delivers color, depth aligned to color, or both.

``physicalai.capture.RealSenseCamera`` always streams depth at the color
resolution (which fails for color modes the depth sensor lacks, such as
1920x1080) and its shared-memory publisher only forwards color. Studio
records depth too, so it captures RealSense devices with this camera instead
and keeps the hardware-agnostic transport: color and depth of one frameset
travel together in a single ``uint8`` frame, so they always share a timestamp.

Packed frame layouts, by channel count:

- 3: color only, ``RGB`` (unchanged from a plain color camera).
- 2: depth only, the little-endian bytes of ``uint16`` millimetres.
- 5: color and depth aligned to color: ``R, G, B, depth low byte, depth high byte``.

:func:`unpack_rgbd` splits a packed frame back into ``(color, depth)``.

``pyrealsense2`` is imported on connect only, so the packing helpers and mode
selection stay importable (and testable) without the SDK.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from loguru import logger
from physicalai.capture.camera import Camera, ColorMode
from physicalai.capture.errors import CaptureError, CaptureTimeoutError, NotConnectedError
from physicalai.capture.frame import Frame

if TYPE_CHECKING:
    from collections.abc import Iterable

RealSenseOutput = Literal["color", "depth", "both"]
OUTPUT_TYPES: tuple[RealSenseOutput, ...] = ("color", "depth", "both")

COLOR_CHANNELS = 3
DEPTH_CHANNELS = 2
RGBD_CHANNELS = COLOR_CHANNELS + DEPTH_CHANNELS

Mode = tuple[int, int, int]
"""``(width, height, fps)``."""


class RealSenseModeError(CaptureError):
    """The requested streams are not available on this device and connection."""


def output_has_color(output_type: str) -> bool:
    """Whether a camera with ``output_type`` delivers a color image."""
    return output_type in {"color", "both"}


def output_has_depth(output_type: str) -> bool:
    """Whether a camera with ``output_type`` delivers a depth map."""
    return output_type in {"depth", "both"}


def packed_channels(output_type: str) -> int:
    """Channels of the packed frame a camera with ``output_type`` delivers."""
    return {"color": COLOR_CHANNELS, "depth": DEPTH_CHANNELS, "both": RGBD_CHANNELS}[output_type]


def pack_rgbd(color: np.ndarray | None, depth: np.ndarray | None) -> np.ndarray:
    """Pack an ``(H, W, 3)`` uint8 color image and/or an ``(H, W)`` uint16 depth map into one uint8 frame.

    Raises:
        ValueError: Neither image is given, or their shapes or dtypes do not fit.
    """
    if depth is None:
        if color is None:
            raise ValueError("Nothing to pack: neither color nor depth was given")
        return color
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Depth must be a 2-D uint16 array, got {depth.dtype} {depth.shape}")
    depth_bytes = np.ascontiguousarray(depth, dtype="<u2").view(np.uint8).reshape(*depth.shape, DEPTH_CHANNELS)
    if color is None:
        return depth_bytes
    if color.shape[:2] != depth.shape or color.dtype != np.uint8 or color.shape[2:] != (COLOR_CHANNELS,):
        raise ValueError(f"Color {color.dtype} {color.shape} does not match depth {depth.shape}")
    packed = np.empty((*depth.shape, RGBD_CHANNELS), dtype=np.uint8)
    packed[..., :COLOR_CHANNELS] = color
    packed[..., COLOR_CHANNELS:] = depth_bytes
    return packed


def unpack_rgbd(data: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Split a packed frame into ``(color, depth)``: ``(H, W, 3)`` uint8 and ``(H, W, 1)`` uint16, or ``None``.

    Raises:
        ValueError: ``data`` is not a packed RealSense frame.
    """
    if data.dtype != np.uint8 or data.ndim != 3:
        raise ValueError(f"Not a packed RealSense frame: {data.dtype} {data.shape}")
    channels = data.shape[2]
    if channels == COLOR_CHANNELS:
        return data, None
    if channels == DEPTH_CHANNELS:
        return None, _depth_from_bytes(data)
    if channels == RGBD_CHANNELS:
        return data[..., :COLOR_CHANNELS], _depth_from_bytes(data[..., COLOR_CHANNELS:])
    raise ValueError(f"Not a packed RealSense frame: {channels} channels")


def _depth_from_bytes(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data).view("<u2").astype(np.uint16, copy=False)


@dataclass(frozen=True, slots=True)
class StreamPlan:
    color: Mode | None
    depth: Mode | None

    def describe(self) -> str:
        parts = []
        if self.color is not None:
            parts.append(f"color {_mode(self.color)}")
        if self.depth is not None:
            parts.append(f"depth {_mode(self.depth)}")
        return " + ".join(parts)


def _mode(mode: Mode) -> str:
    return f"{mode[0]}x{mode[1]}@{mode[2]}fps"


def _list_modes(modes: Iterable[Mode]) -> str:
    listed = sorted(set(modes))
    return ", ".join(_mode(mode) for mode in listed) if listed else "none"


def _usb_hint(usb_type: str | None) -> str:
    if usb_type and usb_type.startswith("2"):
        return (
            f" The camera is connected over USB {usb_type}, which only offers low-bandwidth modes;"
            " plug it into a USB 3 port with a USB 3 cable for 1280x720 at 30 fps."
        )
    return ""


def select_streams(  # noqa: PLR0913
    color_modes: Iterable[Mode],
    depth_modes: Iterable[Mode],
    *,
    width: int,
    height: int,
    fps: int,
    output_type: str,
    depth_size: tuple[int, int] | None = None,
    usb_type: str | None = None,
    device: str = "RealSense",
) -> StreamPlan:
    """Choose the color and depth streams to enable, or explain why the request cannot be met.

    Color is the requested mode exactly. Depth-only streams depth at the requested mode.
    With both, depth is aligned to color, so its own resolution only sets the detail
    behind the aligned map: ``depth_size`` if given, else the color resolution when the
    depth sensor has it, else the largest depth mode with the same fps and aspect ratio.

    Raises:
        RealSenseModeError: The device does not offer what is asked at this connection.
        ValueError: ``output_type`` is unknown.
    """
    if output_type not in OUTPUT_TYPES:
        raise ValueError(f"Unknown RealSense output type {output_type!r}; expected one of {OUTPUT_TYPES}")
    colors, depths = set(color_modes), set(depth_modes)
    requested = (width, height, fps)
    hint = _usb_hint(usb_type)

    if output_type == "depth":
        if requested not in depths:
            raise RealSenseModeError(
                f"{device} has no depth mode {_mode(requested)}. Depth modes on this connection: "
                f"{_list_modes(depths)}.{hint}"
            )
        return StreamPlan(color=None, depth=requested)

    if requested not in colors:
        raise RealSenseModeError(
            f"{device} has no color mode {_mode(requested)}. Color modes on this connection: "
            f"{_list_modes(colors)}.{hint}"
        )
    if output_type == "color":
        return StreamPlan(color=requested, depth=None)

    same_fps = [mode for mode in depths if mode[2] == fps]
    if depth_size is not None:
        depth = (*depth_size, fps)
        if depth not in depths:
            raise RealSenseModeError(
                f"{device} has no depth mode {_mode(depth)}. Depth modes at {fps} fps on this connection: "
                f"{_list_modes(same_fps)}.{hint}"
            )
        return StreamPlan(color=requested, depth=depth)
    if requested in depths:
        return StreamPlan(color=requested, depth=requested)
    if not same_fps:
        raise RealSenseModeError(
            f"{device} has no depth mode at {fps} fps to align to color {_mode(requested)}. "
            f"Depth modes on this connection: {_list_modes(depths)}.{hint}"
        )
    aspect = width / height
    same_aspect = [mode for mode in same_fps if abs(mode[0] / mode[1] - aspect) < 0.02]
    depth = max(same_aspect or same_fps, key=lambda mode: mode[0] * mode[1])
    return StreamPlan(color=requested, depth=depth)


class RealSenseRGBDCamera(Camera):
    """RealSense camera that streams color, depth aligned to color, or both, as packed frames.

    Args:
        serial_number: The device's serial number.
        width: Color width; also the depth width for depth-only output.
        height: Color height; also the depth height for depth-only output.
        fps: Frame rate of every enabled stream.
        output_type: ``"color"``, ``"depth"`` or ``"both"``; only those streams are enabled.
        color_mode: Channel order of the color image (``rgb`` or ``bgr``).
        depth_width: Depth sensor width with ``"both"``; chosen automatically when omitted.
        depth_height: Depth sensor height with ``"both"``; chosen automatically when omitted.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        serial_number: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        output_type: RealSenseOutput = "both",
        color_mode: ColorMode = ColorMode.RGB,
        depth_width: int | None = None,
        depth_height: int | None = None,
    ) -> None:
        super().__init__(color_mode=color_mode)
        if output_type not in OUTPUT_TYPES:
            raise ValueError(f"Unknown RealSense output type {output_type!r}; expected one of {OUTPUT_TYPES}")
        if self._color_mode == ColorMode.GRAY:
            raise ValueError("RealSenseRGBDCamera delivers rgb or bgr color, not gray")
        if (depth_width is None) != (depth_height is None):
            raise ValueError("Give both depth_width and depth_height, or neither")
        self._serial_number = serial_number
        self._width = width
        self._height = height
        self._fps = fps
        self._output_type: RealSenseOutput = output_type
        self._depth_size = None if depth_width is None or depth_height is None else (depth_width, depth_height)
        self._pipeline: Any | None = None
        self._align: Any | None = None
        self._latest: Frame | None = None
        self._sequence = 0
        self.plan: StreamPlan | None = None

    @property
    def output_type(self) -> RealSenseOutput:
        return self._output_type

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    @property
    def device_id(self) -> str:
        return f"realsense:{self._serial_number}"

    def connect(self, timeout: float = 5.0) -> None:
        """Check the requested mode against the device, start the streams and wait for a first frameset.

        Raises:
            CaptureError: The device is missing or busy.
            RealSenseModeError: The device does not offer the requested mode on this connection.
            CaptureTimeoutError: No frameset arrived within ``timeout``.
        """
        import pyrealsense2 as rs_module

        rs = cast("Any", rs_module)
        device = _find_device(rs, self._serial_number)
        name = _info(rs, device, "name") or "RealSense"
        usb_type = _info(rs, device, "usb_type_descriptor")
        color_modes, depth_modes = _stream_modes(rs, device)
        label = f"{name} {self._serial_number}" + (f" (USB {usb_type})" if usb_type else "")
        plan = select_streams(
            color_modes,
            depth_modes,
            width=self._width,
            height=self._height,
            fps=self._fps,
            output_type=self._output_type,
            depth_size=self._depth_size,
            usb_type=usb_type,
            device=label,
        )

        config = rs.config()
        config.enable_device(self._serial_number)
        if plan.color is not None:
            fmt = rs.format.bgr8 if self._color_mode == ColorMode.BGR else rs.format.rgb8
            config.enable_stream(rs.stream.color, plan.color[0], plan.color[1], fmt, plan.color[2])
        if plan.depth is not None:
            config.enable_stream(rs.stream.depth, plan.depth[0], plan.depth[1], rs.format.z16, plan.depth[2])
        pipeline = rs.pipeline()
        try:
            pipeline.start(config)
        except RuntimeError as error:
            raise CaptureError(
                f"Could not start {label} with {plan.describe()}: {error}. "
                "Is another program using the camera?" + _usb_hint(usb_type)
            ) from error
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color) if plan.color is not None and plan.depth is not None else None
        self.plan = plan
        self._sequence = 0
        try:
            self._latest = self._process(self._wait(timeout))
        except Exception:
            self._do_disconnect()
            raise
        logger.info("{} streaming {} as {!r}", label, plan.describe(), self._output_type)

    def read(self, timeout: float = 2.0) -> Frame:
        """The next frameset, packed.

        Raises:
            NotConnectedError: The camera is not connected.
            CaptureTimeoutError: No frameset arrived within ``timeout``.
        """
        self._latest = self._process(self._wait(timeout))
        return self._latest

    def read_latest(self) -> Frame:
        """The newest frameset, packed, without blocking; the previous one if nothing new arrived.

        Raises:
            NotConnectedError: The camera is not connected.
            CaptureError: No frameset has arrived yet.
        """
        pipeline = self._require_pipeline()
        newest = None
        while frameset := pipeline.poll_for_frames():
            newest = frameset
        if newest is not None:
            self._latest = self._process(newest)
        if self._latest is None:
            raise CaptureError("No frame available")
        return self._latest

    def _require_pipeline(self) -> Any:
        if self._pipeline is None:
            raise NotConnectedError
        return self._pipeline

    def _wait(self, timeout: float) -> Any:
        pipeline = self._require_pipeline()
        try:
            return pipeline.wait_for_frames(int(timeout * 1000))
        except RuntimeError as error:
            raise CaptureTimeoutError(f"No RealSense frameset within {timeout}s") from error

    def _process(self, frameset: Any) -> Frame:
        timestamp = time.monotonic()
        if self._align is not None:
            frameset = self._align.process(frameset)
        color = depth = None
        if output_has_color(self._output_type):
            color_frame = frameset.get_color_frame()
            if not color_frame:
                raise CaptureError("The frameset has no color frame")
            color = np.asanyarray(color_frame.get_data())
        if output_has_depth(self._output_type):
            depth_frame = frameset.get_depth_frame()
            if not depth_frame:
                raise CaptureError("The frameset has no depth frame")
            depth = np.asanyarray(depth_frame.get_data())
        self._sequence += 1
        # The SDK reuses its buffers: copy out of them, which packing already does for depth.
        data = pack_rgbd(None if color is None else (color.copy() if depth is None else color), depth)
        return Frame(data=data, timestamp=timestamp, sequence=self._sequence)

    def _do_disconnect(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        self._align = None
        self._latest = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                logger.debug("Stopping RealSense {} failed", self._serial_number)

    @classmethod
    def query_formats(cls, device_id: str | dict, output_type: str = "color") -> list[Mode]:
        """The ``(width, height, fps)`` modes this device offers for ``output_type`` on its current connection.

        For ``"both"``, color modes with a depth mode at the same fps to align.
        """
        import pyrealsense2 as rs_module

        rs = cast("Any", rs_module)
        serial = str(device_id.get("serial", "")) if isinstance(device_id, dict) else str(device_id)
        try:
            device = _find_device(rs, serial)
        except CaptureError:
            return []
        color_modes, depth_modes = _stream_modes(rs, device)
        if output_type == "depth":
            return sorted(depth_modes)
        if output_type == "both":
            depth_fps = {mode[2] for mode in depth_modes}
            return sorted(mode for mode in color_modes if mode[2] in depth_fps)
        return sorted(color_modes)


def _find_device(rs: Any, serial: str) -> Any:
    devices = list(rs.context().query_devices())
    for device in devices:
        if _info(rs, device, "serial_number") == serial:
            return device
    found = ", ".join(filter(None, (_info(rs, device, "serial_number") for device in devices))) or "none"
    raise CaptureError(f"RealSense {serial} is not connected (connected RealSense devices: {found})")


def _info(rs: Any, device: Any, field: str) -> str | None:
    key = getattr(rs.camera_info, field)
    return str(device.get_info(key)) if device.supports(key) else None


def _stream_modes(rs: Any, device: Any) -> tuple[set[Mode], set[Mode]]:
    """The device's ``rgb8`` color and ``z16`` depth modes on its current connection."""
    color: set[Mode] = set()
    depth: set[Mode] = set()
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            stream, fmt = profile.stream_type(), profile.format()
            if stream == rs.stream.color and fmt == rs.format.rgb8:
                target = color
            elif stream == rs.stream.depth and fmt == rs.format.z16:
                target = depth
            else:
                continue
            video = profile.as_video_stream_profile()
            target.add((video.width(), video.height(), profile.fps()))
    return color, depth
