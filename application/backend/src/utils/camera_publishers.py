"""The camera publishers Studio runs: started, replaced and stopped by Studio, never left behind.

A camera is captured by a publisher process that shares its frames over iceoryx2
(``physicalai.capture.SharedCamera``). On its own, ``SharedCamera.connect`` starts a
publisher only when it finds none serving the camera's service, and otherwise asks the
running one to reconfigure itself in place. Studio cannot rely on either:

- The service outlives a publisher that died while subscribers in this process were
  attached (a reconfigure that crashed it, a camera that fell off USB). It then looks
  served, so no publisher is started and every connect times out, until a restart.
- An in-place reconfigure races the publisher's capture loop and can kill it, and a
  subscriber that is not allowed to reconfigure silently gets the old settings.
- A publisher runs in its own session, so a backend that is killed leaves it behind
  holding the device; its dead subscribers still count as subscribers, so it never goes
  idle and never exits.

So this backend starts every publisher itself, one per camera device, with exactly the
settings asked for: a request with other settings stops the running publisher and waits
for it to release the device before starting the new one; a publisher that died, or does
not deliver, is replaced; and every publisher is stopped when the backend shuts down.
Each one is recorded on disk with its start time, so a backend that starts after one was
killed stops what it left behind, and nothing else.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from loguru import logger
from physicalai.capture.errors import CaptureError

if TYPE_CHECKING:
    from pathlib import Path

    from physicalai.capture.transport._publisher import CameraPublisher
    from physicalai.config import Config

PUBLISHER_MODULE = "physicalai.capture.transport._publisher_worker"
"""What a publisher process runs, as its command line shows it."""

_STOP_TIMEOUT_S = 5.0
"""How long a publisher has to release its camera after SIGTERM before it is killed."""


class CameraBusyError(CaptureError):
    """The camera is captured with other settings, or by a publisher Studio does not own."""


@dataclass
class _Publisher:
    process: CameraPublisher
    service_name: str
    recipe: dict[str, Any]

    @property
    def pid(self) -> int | None:
        popen = self.process._process  # the handle CameraPublisher keeps
        return None if popen is None else popen.pid

    @property
    def alive(self) -> bool:
        return self.process.is_alive


@dataclass
class CameraPublishers:
    """The publishers this backend runs, by camera device.

    Args:
        records_dir: Where each running publisher is recorded, for a later backend to stop
            what a killed one left behind; ``None`` records nothing.
    """

    records_dir: Path | None = None
    _running: dict[str, _Publisher] = field(default_factory=dict)
    _locks: dict[str, threading.Lock] = field(default_factory=dict)
    _guard: threading.Lock = field(default_factory=threading.Lock)
    _requests: dict[str, tuple[int, str, dict[str, Any]]] = field(default_factory=dict)
    """The newest request per device: its number, service and recipe."""

    def _lock(self, device: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(device, threading.Lock())

    def ensure(
        self, device: str, service_name: str, camera: Config, *, idle_timeout: float, replace: bool = True
    ) -> None:
        """Make sure a live publisher of ``device`` serves ``service_name`` with exactly ``camera``.

        Args:
            device: What identifies the physical camera; one publisher per device at a time.
            service_name: The iceoryx2 service the publisher serves.
            camera: The camera recipe the publisher captures with.
            idle_timeout: Seconds without subscribers after which the publisher exits.
            replace: Whether a running publisher with other settings may be replaced.

        Raises:
            CameraBusyError: The camera runs with other settings and ``replace`` is off, or a
                publisher this backend did not start serves it.
            CaptureError: The publisher could not open the camera.
        """
        recipe = camera.to_dict()
        with self._guard:
            number = self._requests.get(device, (0, "", {}))[0] + 1
            self._requests[device] = (number, service_name, recipe)
        with self._lock(device):
            newest, newest_service, newest_recipe = self._requests[device]
            if number != newest and (newest_service, newest_recipe) != (service_name, recipe):
                # Settings changed again while this request waited (a camera form clicked through
                # quickly): the newer request wins, rather than whichever thread got the lock last.
                raise CameraBusyError(
                    f"Camera {device} was asked for {_describe(newest_recipe)} after this request for "
                    f"{_describe(recipe)}; the newer settings win."
                )
            running = self._running.get(device)
            if running is not None and running.alive:
                if running.service_name == service_name and running.recipe == recipe:
                    return
                if not replace:
                    raise CameraBusyError(
                        f"Camera {device} is captured with other settings ({_describe(running.recipe)}); "
                        f"it cannot switch to {_describe(recipe)} while it is in use."
                    )
                logger.info(
                    "Camera {}: stopping its publisher ({}) to start one with {}",
                    device,
                    _describe(running.recipe),
                    _describe(recipe),
                )
            if running is not None:
                self._stop(device)
            self._clear_foreign_publishers(service_name)
            self._start(device, service_name, camera, recipe, idle_timeout)

    def restart(self, device: str, service_name: str, camera: Config, *, idle_timeout: float) -> None:
        """Replace the publisher of ``device`` with a fresh one, for one that is alive but does not deliver."""
        with self._lock(device):
            logger.warning("Camera {}: its publisher delivers no frames; starting a fresh one", device)
            self._stop(device)
            self._clear_foreign_publishers(service_name)
            self._start(device, service_name, camera, camera.to_dict(), idle_timeout)

    def stop_all(self) -> None:
        """Stop every publisher this backend started, releasing the cameras."""
        for device in list(self._running):
            with self._lock(device):
                self._stop(device)

    def sweep_orphans(self) -> list[int]:
        """Stop the publishers a Studio backend started and left behind when it was killed. Returns their pids.

        Only publishers recorded in :attr:`records_dir` by a backend that is gone: a process is
        matched by pid *and* start time, so a reused pid is never mistaken for one. Publishers of
        other programs, which their own subscribers may still use, are never touched.
        """
        stopped: list[int] = []
        for record_path, record in self._records():
            if _same_process(record["owner"], record.get("owner_started")) and record["owner"] != os.getpid():
                continue  # a live backend's publisher
            if _same_process(record["pid"], record.get("started")) and _is_publisher_process(record["pid"]):
                _terminate(record["pid"])
                stopped.append(record["pid"])
            record_path.unlink(missing_ok=True)
        if stopped:
            logger.warning("Stopped camera publishers left behind by a backend that is gone: {}", stopped)
            _cleanup_dead_nodes()
        return stopped

    def _records(self) -> list[tuple[Path, dict[str, Any]]]:
        """The recorded publishers; unreadable records are removed."""
        if self.records_dir is None or not self.records_dir.is_dir():
            return []
        records = []
        for record_path in self.records_dir.glob("*.json"):
            try:
                record = json.loads(record_path.read_text())
                record["pid"], record["owner"] = int(record["pid"]), int(record["owner"])
            except (OSError, ValueError, KeyError, TypeError):
                record_path.unlink(missing_ok=True)
                continue
            records.append((record_path, record))
        return records

    def _left_behind(self, pid: int) -> bool:
        """Whether ``pid`` is a publisher a Studio backend that is gone started, as recorded."""
        for _path, record in self._records():
            if record["pid"] == pid and _same_process(pid, record.get("started")):
                return not _same_process(record["owner"], record.get("owner_started"))
        return False

    def _start(self, device: str, service_name: str, camera: Config, recipe: dict, idle_timeout: float) -> None:
        from physicalai.capture.transport._publisher import CameraPublisher
        from physicalai.capture.transport._spec import CameraPublisherConfig

        publisher = CameraPublisher(CameraPublisherConfig(camera=camera), service_name, idle_timeout=idle_timeout)
        publisher.start()  # waits until the camera delivers, or raises with why it cannot
        running = _Publisher(publisher, service_name, recipe)
        self._running[device] = running
        self._record(running)
        logger.info("Camera {}: publisher {} started with {}", device, running.pid, _describe(recipe))

    def _stop(self, device: str) -> None:
        running = self._running.pop(device, None)
        if running is None:
            return
        pid = running.pid
        running.process.stop(timeout=_STOP_TIMEOUT_S)  # SIGTERM, then SIGKILL; reaps it either way
        if pid is not None and self.records_dir is not None:
            (self.records_dir / f"{pid}.json").unlink(missing_ok=True)

    def _clear_foreign_publishers(self, service_name: str) -> None:
        """Stop publishers of ``service_name`` that a backend which is gone left behind.

        Raises:
            CameraBusyError: A publisher that a live process started serves it.
        """
        for pid in _publisher_pids(service_name):
            if self._left_behind(pid):
                logger.warning(
                    "Stopping camera publisher {} of {}, left behind by a backend that is gone", pid, service_name
                )
                _terminate(pid)
            else:
                raise CameraBusyError(
                    f"Camera service {service_name} is served by publisher process {pid}, which this Studio "
                    f"backend did not start. Stop the program using the camera (or `kill {pid}`) and retry."
                )

    def _record(self, running: _Publisher) -> None:
        if self.records_dir is None or running.pid is None:
            return
        self.records_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "pid": running.pid,
            "started": _start_time(running.pid),
            "owner": os.getpid(),
            "owner_started": _start_time(os.getpid()),
            "service": running.service_name,
        }
        (self.records_dir / f"{running.pid}.json").write_text(json.dumps(record))


def _describe(recipe: dict[str, Any]) -> str:
    init_args = recipe.get("init_args") or {}
    mode = f"{init_args.get('width')}x{init_args.get('height')}@{init_args.get('fps')}fps"
    output = init_args.get("output_type")
    return f"{mode} {output}" if output else mode


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cmdline(pid: int) -> list[str]:
    import psutil

    try:
        return psutil.Process(pid).cmdline()
    except (psutil.Error, OSError):
        return []


def _is_publisher_process(pid: int) -> bool:
    return PUBLISHER_MODULE in _cmdline(pid)


def _start_time(pid: int) -> float | None:
    """When process ``pid`` started, which tells it apart from a later process that got its pid."""
    import psutil

    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.Error, OSError):
        return None


def _same_process(pid: int, started: float | None) -> bool:
    """Whether ``pid`` is alive and is the process that started at ``started`` (if known)."""
    if not _process_alive(pid):
        return False
    if started is None:
        return True
    now = _start_time(pid)
    return now is not None and abs(now - started) < 0.01


def _terminate(pid: int, timeout: float = _STOP_TIMEOUT_S) -> bool:
    """SIGTERM the publisher ``pid``, and SIGKILL it if it has not exited within ``timeout``.

    Returns:
        Whether it had to be killed (``False``: it stopped on SIGTERM, or was gone).
    """
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while _is_publisher_process(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if not _is_publisher_process(pid):
        return False
    logger.warning("Camera publisher {} did not stop on SIGTERM within {}s; killing it", pid, timeout)
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)
    return True


def _publisher_pids(service_name: str) -> set[int]:
    """The live publisher processes attached to ``service_name``; empty when there is no such service."""
    try:
        iox2 = cast("Any", import_module("iceoryx2"))
        node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
        try:
            service = (
                node.service_builder(iox2.ServiceName.new(service_name))
                .publish_subscribe(iox2.Slice[ctypes.c_uint8])
                .open()
            )
        except Exception:  # no such service
            return set()
        pids = {state[0].id.pid for state in service.nodes if isinstance(state, iox2.NodeState.Alive)}
        del service
    except Exception:
        logger.opt(exception=True).debug("Could not list the nodes of {}", service_name)
        return set()
    return {pid for pid in pids if pid != os.getpid() and _is_publisher_process(pid)}


def _cleanup_dead_nodes() -> None:
    """Release what killed iceoryx2 nodes left registered, such as a killed publisher's port."""
    try:
        iox2 = cast("Any", import_module("iceoryx2"))
        iox2.Node.try_cleanup_dead_nodes(iox2.ServiceType.Ipc, iox2.config.global_config())
    except Exception:
        logger.opt(exception=True).debug("iceoryx2 dead node cleanup failed")


_publishers: CameraPublishers | None = None


def camera_publishers() -> CameraPublishers:
    """This backend's publishers, recorded under the storage directory."""
    global _publishers  # noqa: PLW0603
    if _publishers is None:
        from settings import get_settings

        _publishers = CameraPublishers(records_dir=get_settings().storage_dir / "run" / "camera-publishers")
    return _publishers
