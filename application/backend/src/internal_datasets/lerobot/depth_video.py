"""Read depth-map videos back for people: thumbnails and browser-playable previews.

A depth feature is a 12-bit ``gray12le`` HEVC video of quantized depth (see
``StudioDepthEncoderConfig``), which browsers cannot play. Its frames are
dequantized to millimetres with the parameters recorded in the feature's
``info`` and colorized; a preview is re-encoded with the same frame timestamps,
so an episode's ``from_timestamp`` seeks to the same frame in it.
"""

from __future__ import annotations

import hashlib
import os
import threading
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_PREVIEW_CODECS = (("libx264", {"crf": "23", "preset": "veryfast"}), ("libsvtav1", {"crf": "35", "preset": "10"}))
"""Browser-playable encoders for previews, in order of preference."""
_RANGE_FRAMES = 15
"""How many frames from the start of a video set its color range, so colors do not flicker."""
_locks: dict[Path, threading.Lock] = {}
_locks_guard = threading.Lock()


def depth_quantization(info: Mapping[str, Any] | None) -> dict[str, Any]:
    """The quantization a depth feature was recorded with, as ``dequantize_depth`` takes it.

    Falls back to LeRobot's defaults for what the ``info`` lacks, as LeRobot's reader does.
    """
    from lerobot.configs.video import DEPTH_ENCODER_INFO_FIELD_NAMES, DepthEncoderConfig

    info = info or {}
    defaults = {name: getattr(DepthEncoderConfig, name) for name in DEPTH_ENCODER_INFO_FIELD_NAMES}
    return {name: info.get(f"video.{name}", default) for name, default in defaults.items()}


def _to_millimetres(codes: np.ndarray, quantization: Mapping[str, Any]) -> np.ndarray:
    from lerobot.datasets.depth_utils import dequantize_depth

    depth = dequantize_depth(codes, **quantization, output_unit="mm", output_tensor=False)
    return np.asarray(depth).reshape(codes.shape[-2:])


def decode_depth_frame_mm(
    video_path: Path, timestamp: float, tolerance_s: float, info: Mapping[str, Any] | None
) -> np.ndarray:
    """The ``(H, W)`` depth in millimetres of the frame at ``timestamp`` of a depth video."""
    from lerobot.datasets.video_utils import decode_video_frames

    codes = decode_video_frames(video_path, [timestamp], tolerance_s, backend="pyav", is_depth=True)
    return _to_millimetres(np.asarray(codes[0]), depth_quantization(info))


PREVIEW_CACHE_BYTES = 2 * 1024**3
"""How much the cached previews may take; the least recently used go first."""


def depth_preview_path(video_path: Path, cache_dir: Path) -> Path:
    """Where the preview of ``video_path`` is cached; a new one when the video changes, as recording appends to it.

    ``<video>-<version>.mp4``: the video's part names every version of it, so older ones can be dropped.
    """
    stat = video_path.stat()
    source = hashlib.sha256(str(video_path.resolve()).encode()).hexdigest()[:16]
    version = hashlib.sha256(f"{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[:16]
    return cache_dir / "depth_previews" / f"{source}-{version}.mp4"


def _evict(preview: Path) -> None:
    """Drop the previews of older versions of the same video, then the least recently used beyond the cap."""
    source = preview.name.split("-", 1)[0]
    others = [path for path in preview.parent.glob("*.mp4") if path != preview and ".part." not in path.name]
    for path in others:
        if path.name.startswith(f"{source}-"):
            path.unlink(missing_ok=True)
    kept = sorted((path for path in others if path.exists()), key=lambda path: path.stat().st_atime, reverse=True)
    total = preview.stat().st_size
    for path in kept:
        total += path.stat().st_size
        if total > PREVIEW_CACHE_BYTES:
            path.unlink(missing_ok=True)


def ensure_depth_preview(video_path: Path, cache_dir: Path, info: Mapping[str, Any] | None) -> Path:
    """A browser-playable, colorized copy of a depth video, made once and cached.

    Raises:
        RuntimeError: No browser-playable encoder is available.
    """
    preview = depth_preview_path(video_path, cache_dir)
    with _locks_guard:
        lock = _locks.setdefault(preview, threading.Lock())
    with lock:
        if not preview.is_file():
            preview.parent.mkdir(parents=True, exist_ok=True)
            partial = preview.with_suffix(f".{os.getpid()}.part.mp4")
            try:
                transcode_depth_preview(video_path, partial, info)
                partial.replace(preview)
            finally:
                partial.unlink(missing_ok=True)
            _evict(preview)
    return preview


def transcode_depth_preview(source: Path, target: Path, info: Mapping[str, Any] | None) -> None:
    """Re-encode a depth video as colorized H.264 (or AV1) with the same frame timestamps.

    Raises:
        RuntimeError: No browser-playable encoder is available.
    """
    import av

    from internal_datasets.lerobot.video_encoding_settings import _is_vcodec_usable
    from utils.depth_colormap import colorize_depth

    codec, options = next(((name, opts) for name, opts in _PREVIEW_CODECS if _is_vcodec_usable(name)), (None, None))
    if codec is None or options is None:
        raise RuntimeError("No browser-playable video encoder (libx264 or libsvtav1) to preview depth with")
    quantization = depth_quantization(info)
    with av.open(str(source)) as reader, av.open(str(target), "w") as writer:
        stream = reader.streams.video[0]
        time_base = stream.time_base or Fraction(1, int(stream.average_rate or 30))
        output = writer.add_stream(codec, rate=stream.average_rate or 30, options=dict(options))
        if not isinstance(output, av.VideoStream):
            raise RuntimeError(f"Codec {codec!r} did not create a video stream")
        output.width, output.height = stream.width, stream.height
        output.pix_fmt = "yuv420p"
        output.time_base = time_base
        near = far = None
        frames: list[tuple[Any, np.ndarray]] = []
        for frame in reader.decode(stream):
            depth = _to_millimetres(frame.to_ndarray(format="gray12le"), quantization)
            if near is None:
                frames.append((frame.pts, depth))
                if len(frames) < _RANGE_FRAMES:
                    continue
                near, far = _color_range(np.stack([d for _, d in frames]))
                pending, frames = frames, []
            else:
                pending = [(frame.pts, depth)]
            for pts, mm in pending:
                _encode(writer, output, colorize_depth(mm, near, far), pts, time_base)
        if frames:  # a video shorter than the range sample
            near, far = _color_range(np.stack([d for _, d in frames]))
            for pts, mm in frames:
                _encode(writer, output, colorize_depth(mm, near, far), pts, time_base)
        for packet in output.encode():
            writer.mux(packet)


def _color_range(depth: np.ndarray) -> tuple[float, float]:
    valid = depth[depth > 0]
    if not valid.size:
        return 0.0, 1000.0
    low, high = np.percentile(valid[:: max(1, valid.size // 200_000)], (2, 98))
    return float(low), float(high)


def _encode(writer: Any, output: Any, rgb: np.ndarray, pts: int | None, time_base: Fraction) -> None:
    import av

    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
    frame.pts = pts  # type: ignore[assignment]  # PyAV accepts None: no timestamp
    frame.time_base = time_base
    for packet in output.encode(frame):
        writer.mux(packet)
