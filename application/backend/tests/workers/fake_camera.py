"""A camera without hardware, for real publisher processes in tests.

A publisher imports it by class path, so it must stay importable as ``tests.workers.fake_camera``
from the backend directory, where tests run.
"""

from __future__ import annotations

import time

import numpy as np
from physicalai.capture.camera import Camera
from physicalai.capture.errors import NotConnectedError
from physicalai.capture.frame import Frame

CLASS_PATH = "tests.workers.fake_camera.FakeCamera"


class FakeCamera(Camera):
    """Frames of ``height`` x ``width`` at ``fps``, each filled with its sequence number."""

    def __init__(self, *, serial_number: str, width: int = 64, height: int = 48, fps: int = 30) -> None:
        super().__init__()
        self._serial_number = serial_number
        self._shape = (height, width, 3)
        self._fps = fps
        self._sequence = 0
        self._connected = False
        self._latest: Frame | None = None

    def connect(self, timeout: float = 5.0) -> None:
        self._connected = True
        self._latest = self._frame()

    def _do_disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def device_id(self) -> str:
        return f"fake:{self._serial_number}"

    def _frame(self) -> Frame:
        self._sequence += 1
        data = np.full(self._shape, self._sequence % 256, dtype=np.uint8)
        return Frame(data=data, timestamp=time.monotonic(), sequence=self._sequence)

    def read(self, timeout: float = 2.0) -> Frame:
        if not self._connected:
            raise NotConnectedError
        time.sleep(1 / self._fps)
        self._latest = self._frame()
        return self._latest

    def read_latest(self) -> Frame:
        if not self._connected or self._latest is None:
            raise NotConnectedError
        return self._latest
