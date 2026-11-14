"""Turn depth maps into color images people can look at; recordings keep the millimetres."""

from __future__ import annotations

import cv2
import numpy as np

_SAMPLE_STEP = 4
"""Every 4th pixel each way sets the color range; enough, and 16x cheaper than all of them."""
_MIN_SPAN_MM = 100.0


def colorize_depth(depth: np.ndarray, near_mm: float | None = None, far_mm: float | None = None) -> np.ndarray:
    """An ``(H, W, 3)`` RGB image of an ``(H, W)`` or ``(H, W, 1)`` depth map in millimetres.

    Near is red, far is blue, and no depth (0) is black. Without ``near_mm``/``far_mm`` the range
    spans the 2nd to the 98th percentile of the frame's valid depth.
    """
    depth = depth.reshape(depth.shape[:2]).astype(np.float32, copy=False)
    valid = depth > 0
    if near_mm is None or far_mm is None:
        sample = depth[::_SAMPLE_STEP, ::_SAMPLE_STEP]
        sample = sample[sample > 0]
        low, high = np.percentile(sample, (2, 98)) if sample.size else (0.0, 1.0)
        near_mm = float(low) if near_mm is None else near_mm
        far_mm = float(high) if far_mm is None else far_mm
    span = max(far_mm - near_mm, _MIN_SPAN_MM)
    scaled = np.clip((far_mm - depth) * (255.0 / span), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    bgr[~valid] = 0
    return np.ascontiguousarray(bgr[:, :, ::-1])
