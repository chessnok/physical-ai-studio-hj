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
Each one is recorded on disk, so a backend that starts after one was killed stops what
it left behind.
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
        with self._lock(device):
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
        """Stop the publishers that programs which are gone left behind. Returns their pids.

        Those this backend's predecessors recorded, and any other publisher process that outlived
        the program that started it: it holds its camera, and its dead subscribers keep it from
        ever going idle. Publishers of live programs are left alone.
        """
        stopped = _orphan_publisher_pids()
        for pid in stopped:
            _terminate(pid)
        if self.records_dir is not None and self.records_dir.is_dir():
            stopped += self._sweep_records()
        if stopped:
            logger.warning("Stopped camera publishers left behind by a backend that is gone: {}", stopped)
            _cleanup_dead_nodes()
        return stopped

    def _sweep_records(self) -> list[int]:
        stopped: list[int] = []
        for record_path in cast("Path", self.records_dir).glob("*.json"):
            try:
                record = json.loads(record_path.read_text())
                pid, owner = int(record["pid"]), int(record["owner"])
            except (OSError, ValueError, KeyError, TypeError):
                record_path.unlink(missing_ok=True)
                continue
            if owner != os.getpid() and _process_alive(owner):
                continue  # another live backend's publisher
            if _is_publisher_process(pid):
                _terminate(pid)
                stopped.append(pid)
            record_path.unlink(missing_ok=True)
        return stopped

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
            if _is_orphan(pid):
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
        record = {"pid": running.pid, "owner": os.getpid(), "service": running.service_name}
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


def _is_orphan(pid: int) -> bool:
    """Whether the publisher process ``pid`` outlived the program that started it.

    A publisher runs in its own session, so when its program exits it is adopted by init or a
    session manager, which is not a Python program like the ones that start publishers.
    """
    import psutil

    try:
        parent = psutil.Process(pid).parent()
    except (psutil.Error, OSError):
        return False
    if parent is None:
        return True
    try:
        return "python" not in os.path.basename(parent.exe() or parent.name()).lower()
    except (psutil.Error, OSError):
        return "python" not in parent.name().lower()


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


def _orphan_publisher_pids() -> list[int]:
    """Publisher processes of this user that outlived the program that started them."""
    import psutil

    pids = []
    for process in psutil.process_iter(["pid", "cmdline", "uids"]):
        try:
            mine = process.info["uids"].real == os.getuid()
            if mine and PUBLISHER_MODULE in (process.info["cmdline"] or []) and _is_orphan(process.pid):
                pids.append(process.pid)
        except (psutil.Error, OSError, AttributeError):
            continue
    return pids


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
