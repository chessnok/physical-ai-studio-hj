"""Studio runs its camera publishers: real publisher processes, with a camera that needs no hardware.

Regression for the camera form's preview: changing the resolution reconfigured the running
publisher in place, which killed it, and every later connect then waited on a service nobody
published to, until Studio was restarted.
"""

from __future__ import annotations

import json
import os
import signal
import time
import uuid
from typing import TYPE_CHECKING

import pytest
from physicalai.capture import ColorMode
from physicalai.config import Config

from tests.workers.fake_camera import CLASS_PATH
from utils.camera_factory import StudioSharedCamera
from utils.camera_publishers import CameraBusyError, CameraPublishers, _is_publisher_process

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytest.importorskip("iceoryx2")


@pytest.fixture
def publishers(tmp_path: Path) -> Iterator[CameraPublishers]:
    publishers = CameraPublishers(records_dir=tmp_path / "publishers")
    yield publishers
    publishers.stop_all()


@pytest.fixture
def device() -> str:
    # A service of its own, so nothing else on this machine is touched.
    return f"studio-test-{uuid.uuid4().hex[:12]}"


def _camera(publishers: CameraPublishers, device: str, width: int, *, replace: bool = True) -> StudioSharedCamera:
    camera = StudioSharedCamera(
        camera=Config(CLASS_PATH, {"serial_number": device, "width": width, "height": 48, "fps": 30}),
        service_name=f"physicalai/camera/FakeCamera/{device}/frame",
        color_mode=ColorMode.RGB,
        validate_on_connect=True,
        idle_timeout=5.0,
    )
    camera.device, camera.replace_settings, camera.publishers = device, replace, publishers
    return camera


def _pid(publishers: CameraPublishers, device: str) -> int:
    pid = publishers._running[device].pid
    assert pid is not None
    return pid


def test_changing_settings_stops_the_old_publisher_before_starting_one_with_the_new(publishers, device) -> None:
    old = _camera(publishers, device, 64)
    old.connect()
    assert old.read(timeout=2).data.shape == (48, 64, 3)
    old_pid = _pid(publishers, device)

    # The form's preview asks for another resolution while the old preview is still attached.
    new = _camera(publishers, device, 80)
    new.connect()
    try:
        assert not _is_publisher_process(old_pid)
        assert _pid(publishers, device) != old_pid
        assert new.read(timeout=2).data.shape == (48, 80, 3)
        # The same settings again reuse the running publisher.
        again = _camera(publishers, device, 80)
        again.connect()
        assert _pid(publishers, device) == publishers._running[device].pid
        again.disconnect()
    finally:
        old.disconnect()
        new.disconnect()


def test_a_camera_in_use_is_not_switched_to_other_settings_unless_allowed(publishers, device) -> None:
    recording = _camera(publishers, device, 64)
    recording.connect()
    try:
        with pytest.raises(CameraBusyError, match="other settings"):
            _camera(publishers, device, 80, replace=False).connect()
        assert recording.read(timeout=2).data.shape == (48, 64, 3)
    finally:
        recording.disconnect()


def test_a_dead_publisher_is_replaced_although_its_service_lives_on(publishers, device) -> None:
    attached = _camera(publishers, device, 64)
    attached.connect()
    dead_pid = _pid(publishers, device)
    os.kill(dead_pid, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while _is_publisher_process(dead_pid) and time.monotonic() < deadline:
        time.sleep(0.05)

    # ``attached`` keeps the service open: it looks served, but nobody publishes to it.
    camera = _camera(publishers, device, 64)
    camera.connect(timeout=3)
    try:
        assert _pid(publishers, device) != dead_pid
        assert camera.read(timeout=2).data.shape == (48, 64, 3)
        assert attached.read(timeout=2).data.shape == (48, 64, 3)  # old subscribers recover too
    finally:
        camera.disconnect()
        attached.disconnect()


def test_publishers_are_recorded_and_stopped_with_the_backend(publishers, device, tmp_path) -> None:
    camera = _camera(publishers, device, 64)
    camera.connect()
    pid = _pid(publishers, device)
    record = json.loads((tmp_path / "publishers" / f"{pid}.json").read_text())
    assert record == {"pid": pid, "owner": os.getpid(), "service": camera.service_name}

    camera.disconnect()
    publishers.stop_all()
    assert not _is_publisher_process(pid)
    assert not list((tmp_path / "publishers").glob("*.json"))


def test_a_new_backend_stops_the_publishers_a_killed_one_left_behind(publishers, device, tmp_path) -> None:
    camera = _camera(publishers, device, 64)
    camera.connect()
    pid = _pid(publishers, device)
    camera.disconnect()
    # As a killed backend leaves it: running, recorded, owned by a process that is gone.
    publishers._running.clear()
    record = tmp_path / "publishers" / f"{pid}.json"
    record.write_text(json.dumps({"pid": pid, "owner": 2**22 + 1, "service": camera.service_name}))

    restarted = CameraPublishers(records_dir=tmp_path / "publishers")
    assert restarted.sweep_orphans() == [pid]
    assert not _is_publisher_process(pid)
    assert not record.exists()
