import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext

import cv2
import numpy as np
from loguru import logger
from physicalai.capture import SharedCamera
from physicalai.capture.frame import Frame

from runtime_session.bus import Bus, publish_status
from runtime_session.store import ImageSpec, Store, depth_key
from schemas.project_camera import Camera
from schemas.runtime_session import ComponentState
from utils.camera_factory import build_shared_camera
from utils.realsense_rgbd import output_has_color, output_has_depth, unpack_rgbd
from workers.base import ThreadWorker
from workers.rate import Rate

POLLS_PER_FRAME = 2
"""How many times per frame interval a camera is polled, so a new frame reaches the store within half a frame."""


def camera_image_spec(config: Camera) -> ImageSpec:
    """The frame shape a camera's color store slot has."""
    # TODO explicitly add width, height to ip camera
    return ImageSpec(height=config.payload.height or 480, width=config.payload.width or 640)


def depth_downscale() -> int:
    """By how much depth maps are decimated for recording (``DEPTH_RECORD_DOWNSCALE``)."""
    from settings import get_settings

    return get_settings().depth_record_downscale


def camera_observations(key: str, config: Camera) -> dict[str, ImageSpec]:
    """The store slots a camera at ``key`` writes: its color image, its depth map, or both.

    A RealSense camera's ``output_type`` picks them; its depth map, at :func:`depth_key`, is
    aligned to the color image and decimated by :func:`depth_downscale`: depth pixel ``(u, v)``
    is color pixel ``(s*u, s*v)``. Any other camera has its color image only.
    """
    color = camera_image_spec(config)
    if config.driver != "realsense":
        return {key: color}
    output_type = config.payload.output_type
    slots: dict[str, ImageSpec] = {}
    if output_has_color(output_type):
        slots[key] = color
    if output_has_depth(output_type):
        scale = depth_downscale()
        slots[depth_key(key)] = ImageSpec.depth(-(-color.height // scale), -(-color.width // scale))
    return slots


def depth_range_mm(config: Camera) -> tuple[int, int] | None:
    """The depth a RealSense camera keeps, in millimetres; ``None`` for a camera without depth."""
    if config.driver != "realsense" or not output_has_depth(config.payload.output_type):
        return None
    if config.payload.depth_range_problem() is not None:  # a stored row from before ranges were checked
        logger.warning(
            "Camera {!r}: {}; recording its depth unfiltered", config.name, config.payload.depth_range_problem()
        )
        return None
    return round(config.payload.depth_range_min * 1000), round(config.payload.depth_range_max * 1000)


class CameraWorker(ThreadWorker[SharedCamera]):
    """Read one camera into its store slots (see :func:`camera_observations`) as its frames arrive.

    A thread: ``SharedCamera`` already runs the capture in a publisher process.

    A camera that drops off USB is reopened: its publisher exits once the device
    stops answering, after which ``read_latest`` keeps returning the last frame.
    So a camera that sends nothing new for :attr:`stale_after_s` is released,
    reported as ``starting`` and reopened every :attr:`reconnect_interval_s`
    until it is back, without writing the frozen frame in the meantime.
    """

    stale_after_s: float = 0.5
    """How long a camera may go without a new frame before it counts as gone."""

    reconnect_interval_s: float = 1.0
    """How long to wait between attempts to reopen a camera that is gone."""

    def __init__(
        self,
        config: Camera,
        store: Store,
        key: str,
        is_locked: bool = False,
        *,
        session_id: str | None = None,
        component: str | None = None,
    ) -> None:
        """Pass ``session_id`` and ``component`` to report a failure on that session's bus."""
        super().__init__(name=f"CameraWorker-{config.name}")
        self.config = config
        self.session_id = session_id
        self.component = component
        self._bus: Bus | None = None
        self._camera: SharedCamera | None = None
        self.store = store
        self.key = key
        self.is_locked = is_locked
        self._slots = camera_observations(key, config)
        self._color_key = key if key in self._slots else None
        self._depth_key = depth_key(key) if depth_key(key) in self._slots else None
        self._depth_range = depth_range_mm(config)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[SharedCamera]:
        camera = self._open()
        self._camera = camera
        try:
            with Bus.join(self.session_id) if self.session_id else nullcontext() as self._bus:
                yield camera
        finally:
            logger.info("Camera run loop stopped. Disconnecting")
            # Whichever camera is open now: a reconnect replaces the one yielded.
            if self._camera is not None and self._camera.is_connected:
                self._camera.disconnect()

    def _open(self) -> SharedCamera:
        camera = build_shared_camera(
            config=self.config,
            # Recorded frames must have the configured settings: never attach to a publisher with others.
            validate_on_connect=True,
            overwrite_settings=not self.is_locked,
        )
        camera.connect()
        return camera

    async def run_loop(self, context: SharedCamera) -> None:
        try:
            await self._capture(context)
        except Exception as error:
            self._invalidate()
            self._publish("failed", str(error))
            raise

    def _invalidate(self) -> None:
        for key in self._slots:
            self.store.invalidate_observation(key)

    async def _capture(self, context: SharedCamera) -> None:
        # Polled faster than frames arrive, so each new frame reaches the store within half a frame.
        rate = Rate(self.config.payload.fps * POLLS_PER_FRAME, name=self.name)
        camera: SharedCamera | None = context
        previous = None
        reconnecting = False
        last_new_frame = time.monotonic()
        while camera is not None and not self.should_stop():
            frame = camera.read_latest()
            if frame is not previous:
                previous, last_new_frame = frame, time.monotonic()
                self._write(frame)
                if reconnecting:
                    self._publish("running")
                    reconnecting = False
            elif time.monotonic() - last_new_frame > self.stale_after_s:
                reconnecting = True
                camera = await self._reconnect(camera)
                previous, last_new_frame = None, time.monotonic()
                continue
            await rate.sleep()

    def _write(self, frame: Frame) -> None:
        """Write a frame's color image and depth map to their slots, with the frame's timestamp.

        Color is written before depth, both with the frameset's timestamp, so a reader that finds
        the same timestamp on both has a matching pair.

        Raises:
            RuntimeError: The camera does not deliver the streams its configuration asks for.
        """
        if self._depth_key is None:
            color, depth = frame.data, None
        else:
            color, depth = unpack_rgbd(np.asarray(frame.data))
            if depth is None or (self._color_key is not None and color is None):
                wanted = "color and depth" if self._color_key is not None else "depth"
                raise RuntimeError(
                    f"Camera {self.config.name!r} delivers {frame.data.shape[-1]}-channel frames without the "
                    f"{wanted} streams it is configured for; it may still be running with older settings. "
                    "Reload the environment."
                )
        if self._color_key is not None and color is not None:
            self.store.write_observation(
                self._color_key, _fit(color, self._slots[self._color_key]), timestamp=frame.timestamp
            )
        if self._depth_key is not None and depth is not None:
            depth = _fit(depth, self._slots[self._depth_key])
            if self._depth_range is not None:
                nearest, farthest = self._depth_range
                depth = np.where((depth < nearest) | (depth > farthest), np.uint16(0), depth)
            self.store.write_observation(self._depth_key, depth, timestamp=frame.timestamp)

    async def _reconnect(self, lost: SharedCamera) -> SharedCamera | None:
        """Release a camera that stopped sending frames and reopen it until it is back.

        Returns:
            The reopened camera, or ``None`` if the worker was asked to stop first.
        """
        logger.warning("Camera {!r} sent no frame for {:.1f}s; reconnecting", self.config.name, self.stale_after_s)
        self._invalidate()
        self._publish("starting", f"no frames for {self.stale_after_s:.1f}s; reconnecting")
        self._camera = None
        try:
            lost.disconnect()
        except Exception:
            logger.exception("Disconnecting the lost camera {!r} failed", self.config.name)
        attempts = 0
        while not self.should_stop():
            await self.sleep(self.reconnect_interval_s)
            if self.should_stop():
                break
            attempts += 1
            try:
                camera = await asyncio.to_thread(self._open)
            except Exception as error:
                logger.debug("Camera {!r} not back yet (attempt {}): {}", self.config.name, attempts, error)
                continue
            self._camera = camera
            logger.info("Camera {!r} reconnected after {} attempt(s)", self.config.name, attempts)
            return camera
        return None

    def _publish(self, state: ComponentState, error: str | None = None) -> None:
        if self._bus is not None and self.component is not None:
            publish_status(self._bus, self.component, state, error)


def _fit(image: np.ndarray, slot: ImageSpec) -> np.ndarray:
    """``image`` at the slot's resolution: decimated for a depth slot, resized if the publisher runs at another one.

    Depth is decimated or resized with nearest-neighbour, so every value is a measured one: no
    depth is invented between surfaces.
    """
    if image.shape[:2] == slot.shape[:2]:
        return image
    if slot.is_depth:
        scale = -(-image.shape[0] // slot.height)
        if image[::scale, ::scale].shape[:2] == slot.shape[:2]:
            return np.ascontiguousarray(image[::scale, ::scale])
    size = (slot.width, slot.height)
    image = np.ascontiguousarray(image)  # a color view into an RGB-D frame is strided
    if not slot.is_depth:
        return cv2.resize(image, size)
    return cv2.resize(image.reshape(image.shape[:2]), size, interpolation=cv2.INTER_NEAREST).reshape(slot.shape)
