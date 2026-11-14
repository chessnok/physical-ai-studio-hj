"""JPEG encoding utilities using OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


def encode_jpeg_rgb(image: np.ndarray, quality: int = 80) -> bytes:
    """Encode an RGB image to JPEG bytes.

    Args:
        image: HxWx3 numpy array in RGB channel order.
        quality: JPEG quality 1-100 (default 80).

    Returns:
        JPEG-encoded bytes.
    """
    # Convert RGB → BGR (OpenCV convention) and ensure contiguous memory layout.
    bgr = np.ascontiguousarray(image[:, :, ::-1])
    success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        msg = "JPEG encoding failed"
        raise RuntimeError(msg)
    return buf.tobytes()


def encode_jpeg_frame(image: np.ndarray, quality: int = 80) -> bytes:
    """Encode a store frame to JPEG bytes: an RGB image as is, a ``uint16`` depth map colorized.

    Returns:
        JPEG-encoded bytes.
    """
    if image.dtype == np.uint16:
        from utils.depth_colormap import colorize_depth

        image = colorize_depth(image)
    return encode_jpeg_rgb(image, quality)


def encode_jpeg_camera_frame(data: np.ndarray, quality: int = 80) -> bytes:
    """Encode a camera publisher's frame for a preview: its color image, else its colorized depth.

    A RealSense publisher packs color and depth into one frame (see :mod:`utils.realsense_rgbd`);
    any other camera's frame is an RGB image.

    Returns:
        JPEG-encoded bytes.
    """
    if data.dtype == np.uint8 and data.ndim == 3 and data.shape[2] != 3:
        from utils.realsense_rgbd import unpack_rgbd

        color, depth = unpack_rgbd(data)
        shown = color if color is not None else depth
        if shown is None:
            raise ValueError("The frame holds neither color nor depth")
        data = shown
    return encode_jpeg_frame(data, quality)
