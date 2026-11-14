import asyncio
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from runtime_session.store import ACTION, STATE, FeatureSpec, Store
from schemas.runtime_session import StartEpisodeCommand
from workers.camera_worker import CameraWorker, camera_image_spec
from workers.dataset_worker import Recorder
from workers.model_worker import Inference, InferRequest


def _make_config(width=640, height=480, fps=30):
    from schemas.project_camera import CameraAdapter

    return CameraAdapter.validate_python(
        {
            "driver": "usb_camera",
            "name": "test_cam",
            "fingerprint": {"path": "/dev/video0"},
            "hardware_name": None,
            "payload": {"width": width, "height": height, "fps": fps},
        }
    )


def _make_frame(height: int = 480, width: int = 640, value: int = 0, timestamp: float = 1.0):
    mock_frame = MagicMock()
    mock_frame.data = np.full((height, width, 3), value, dtype=np.uint8)
    mock_frame.timestamp = timestamp
    return mock_frame


@pytest.fixture
def store():
    store = Store.create(FeatureSpec((), {"cam": camera_image_spec(_make_config())}), uuid.uuid4().hex)
    yield store
    store.unlink()


def _enter_and_leave(worker: CameraWorker, check=lambda camera: None) -> None:
    async def run() -> None:
        async with worker.lifecycle() as camera:
            check(camera)

    asyncio.run(run())


class TestCameraWorker:
    def test_lifecycle_builds_and_connects_the_camera(self, store):
        config = _make_config()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_cam = mock_build.return_value
            mock_cam.is_connected = True
            worker = CameraWorker(config, store, "cam")
            mock_build.assert_not_called()

            def check(camera) -> None:
                assert camera is mock_cam
                mock_cam.connect.assert_called_once()
                mock_cam.disconnect.assert_not_called()

            _enter_and_leave(worker, check)
            mock_build.assert_called_once_with(config=config, validate_on_connect=True, overwrite_settings=True)
            mock_cam.disconnect.assert_called_once()

    def test_a_locked_camera_keeps_its_settings(self, store):
        config = _make_config()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            _enter_and_leave(CameraWorker(config, store, "cam", is_locked=True))
            mock_build.assert_called_once_with(config=config, validate_on_connect=True, overwrite_settings=False)

    def test_lifecycle_skips_disconnect_if_not_connected(self, store):
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_build.return_value.is_connected = False
            _enter_and_leave(CameraWorker(_make_config(), store, "cam"))
            mock_build.return_value.disconnect.assert_not_called()

    def test_run_loop_writes_frames_into_the_store_until_stopped(self, store):
        worker = CameraWorker(_make_config(), store, "cam")
        frames = [_make_frame(value=1, timestamp=1.0), _make_frame(240, 320, value=2, timestamp=2.0)]

        def read_latest():
            if len(frames) == 1:
                worker.request_stop()
            return frames.pop(0)

        camera = MagicMock()
        camera.read_latest.side_effect = read_latest
        asyncio.run(worker.run_loop(camera))

        sample = store.read_observation("cam")
        assert sample.timestamp == 2.0
        # A frame at another resolution is resized to the slot.
        assert sample.value.shape == (480, 640, 3)
        assert np.all(sample.value == 2)


def _fast_worker(store, **kwargs) -> CameraWorker:
    """A worker that gives up on a silent camera, and retries, within milliseconds."""
    worker = CameraWorker(_make_config(), store, "cam", **kwargs)
    worker.stale_after_s = 0.05
    worker.reconnect_interval_s = 0.01
    return worker


def _frozen_camera() -> MagicMock:
    """A camera whose publisher has exited: ``read_latest`` keeps returning its last frame."""
    camera = MagicMock()
    camera.read_latest.return_value = _make_frame(value=1, timestamp=1.0)
    camera.is_connected = True
    return camera


def _camera_that_stops(worker: CameraWorker, frame) -> MagicMock:
    """A camera that delivers ``frame`` once, then stops the worker."""
    camera = MagicMock()
    camera.is_connected = True

    def read_latest():
        worker.request_stop()
        return frame

    camera.read_latest.side_effect = read_latest
    return camera


class TestCameraReconnect:
    def test_a_camera_that_stops_sending_frames_is_reopened(self, store):
        worker = _fast_worker(store)
        old = _frozen_camera()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            new = _camera_that_stops(worker, _make_frame(value=7, timestamp=5.0))
            mock_build.return_value = new
            asyncio.run(worker.run_loop(old))

        old.disconnect.assert_called_once()
        new.connect.assert_called_once()
        sample = store.read_observation("cam")
        assert sample.timestamp == 5.0
        assert np.all(sample.value == 7)

    def test_recording_and_inference_cannot_consume_a_reconnecting_camera(self, store):
        worker = _fast_worker(store)
        for key in (STATE, ACTION):
            store.write(key, [], timestamp=1.0)
        mutation, model = MagicMock(), MagicMock()
        model.predict_action_chunk.return_value = np.zeros((1, 1))
        recorder = Recorder(mutation, store, publish=MagicMock())
        recorder.start(StartEpisodeCommand(request_id="start", task="pick"))
        inference = Inference(model, store, {"images.cam": "cam"})

        def reopen(**_kwargs):
            assert store.read_observation("cam") is None
            recorder.record_frame()
            mutation.add_frame.assert_not_called()
            assert not recorder.status().is_recording
            mutation.discard_buffer.assert_called_once()
            assert "Episode discarded" in recorder.publish.call_args.args[0].error
            with pytest.raises(RuntimeError, match="has no frame yet"):
                inference.infer(InferRequest(state=[]))
            model.predict_action_chunk.assert_not_called()
            return _camera_that_stops(worker, _make_frame(value=7, timestamp=5.0))

        with patch("workers.camera_worker.build_shared_camera", side_effect=reopen):
            asyncio.run(worker.run_loop(_frozen_camera()))

        recorder.record_frame()
        mutation.add_frame.assert_not_called()  # Recovery must not resume the discarded episode.
        recorder.start(StartEpisodeCommand(request_id="restart", task="pick"))
        recorder.record_frame()
        inference.infer(InferRequest(state=[]))
        mutation.add_frame.assert_called_once()
        model.predict_action_chunk.assert_called_once()
        assert np.all(mutation.add_frame.call_args.args[0]["cam"] == 7)
        assert np.all(model.predict_action_chunk.call_args.args[0]["images.cam"] == 7)

    def test_a_read_failure_invalidates_the_last_frame(self, store):
        worker = _fast_worker(store)
        store.write_observation("cam", _make_frame().data, timestamp=1.0)
        camera = MagicMock()
        camera.read_latest.side_effect = RuntimeError("camera unplugged")
        with pytest.raises(RuntimeError, match="camera unplugged"):
            asyncio.run(worker.run_loop(camera))
        assert store.read_observation("cam") is None

    def test_reopening_keeps_trying_until_the_camera_is_back(self, store):
        from physicalai.capture.errors import CaptureError

        worker = _fast_worker(store)
        absent = MagicMock()
        absent.connect.side_effect = CaptureError("No camera found")
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            back = _camera_that_stops(worker, _make_frame(value=9, timestamp=6.0))
            mock_build.side_effect = [absent, absent, back]
            asyncio.run(worker.run_loop(_frozen_camera()))

        assert mock_build.call_count == 3
        assert store.read_observation("cam").timestamp == 6.0

    def test_a_stop_while_reopening_ends_the_loop(self, store):
        from physicalai.capture.errors import CaptureError

        worker = _fast_worker(store)

        def build(**_kwargs):
            worker.request_stop()
            absent = MagicMock()
            absent.connect.side_effect = CaptureError("No camera found")
            return absent

        with patch("workers.camera_worker.build_shared_camera", side_effect=build):
            asyncio.run(worker.run_loop(_frozen_camera()))

        assert worker.should_stop()

    def test_reconnecting_is_reported_and_then_running_again(self, store):
        worker = _fast_worker(store, component="environment/cam")
        worker._bus = MagicMock()
        with (
            patch("workers.camera_worker.build_shared_camera") as mock_build,
            patch("workers.camera_worker.publish_status") as mock_status,
        ):
            mock_build.return_value = _camera_that_stops(worker, _make_frame(timestamp=5.0))
            asyncio.run(worker.run_loop(_frozen_camera()))

        states = [call.args[2] for call in mock_status.call_args_list]
        assert states == ["starting", "running"]
        assert "reconnecting" in mock_status.call_args_list[0].args[3]

    def test_lifecycle_releases_the_reopened_camera(self, store):
        worker = _fast_worker(store)
        old = _frozen_camera()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            new = _camera_that_stops(worker, _make_frame(timestamp=5.0))
            mock_build.side_effect = [old, new]

            async def run() -> None:
                async with worker.lifecycle() as camera:
                    await worker.run_loop(camera)

            asyncio.run(run())

        old.disconnect.assert_called_once()
        new.disconnect.assert_called_once()


def _realsense_config(output_type="both", width=8, height=6, depth_range=(0.3, 3.0)):
    from schemas.project_camera import CameraAdapter

    # The schema's minimum resolution is irrelevant to the worker; bypass it to keep frames tiny.
    camera = CameraAdapter.validate_python(
        {
            "driver": "realsense",
            "name": "top",
            "fingerprint": {"serial": "123"},
            "hardware_name": None,
            "payload": {
                "width": 1280,
                "height": 720,
                "fps": 30,
                "output_type": output_type,
                "depth_range_min": depth_range[0],
                "depth_range_max": depth_range[1],
            },
        }
    )
    camera.payload.width, camera.payload.height = width, height
    return camera


def _rgbd_frame(color, depth, timestamp=4.0):
    from utils.realsense_rgbd import pack_rgbd

    frame = MagicMock()
    frame.data = pack_rgbd(color, depth)
    frame.timestamp = timestamp
    return frame


def _run_once(worker, frame):
    camera = MagicMock()

    def read_latest():
        worker.request_stop()
        return frame

    camera.read_latest.side_effect = read_latest
    asyncio.run(worker.run_loop(camera))


class TestRealSenseCameraWorker:
    @pytest.fixture
    def rgbd_store(self):
        from workers.camera_worker import camera_observations

        store = Store.create(FeatureSpec((), camera_observations("top", _realsense_config())), uuid.uuid4().hex)
        yield store
        store.unlink()

    def test_both_writes_color_and_aligned_depth_with_one_timestamp(self, rgbd_store):
        color = np.full((6, 8, 3), 5, np.uint8)
        depth = np.full((6, 8), 1500, np.uint16)
        depth[0, 0], depth[0, 1], depth[0, 2] = 0, 200, 3500  # no data, nearer and farther than the range
        _run_once(CameraWorker(_realsense_config(), rgbd_store, "top"), _rgbd_frame(color, depth))

        color_sample, depth_sample = rgbd_store.read_observation("top"), rgbd_store.read_observation("top_depth")
        assert color_sample.timestamp == depth_sample.timestamp == 4.0
        np.testing.assert_array_equal(color_sample.value, color)
        assert depth_sample.value.dtype == np.uint16
        assert depth_sample.value.shape == (6, 8, 1)
        assert depth_sample.value[0, :3, 0].tolist() == [0, 0, 0]  # outside 0.3-3 m is no data
        assert np.all(depth_sample.value[1:] == 1500)

    def test_depth_at_another_resolution_is_resized_without_inventing_depth(self, rgbd_store):
        depth = np.zeros((3, 4), np.uint16)
        depth[:, 2:] = 1000
        _run_once(
            CameraWorker(_realsense_config(), rgbd_store, "top"), _rgbd_frame(np.zeros((3, 4, 3), np.uint8), depth)
        )

        values = set(np.unique(rgbd_store.read_observation("top_depth").value).tolist())
        assert values == {0, 1000}

    def test_depth_only(self):
        from workers.camera_worker import camera_observations

        config = _realsense_config("depth")
        store = Store.create(FeatureSpec((), camera_observations("top", config)), uuid.uuid4().hex)
        try:
            _run_once(CameraWorker(config, store, "top"), _rgbd_frame(None, np.full((6, 8), 900, np.uint16)))
            assert set(store.spec.observations) == {"top_depth"}
            assert np.all(store.read_observation("top_depth").value == 900)
        finally:
            store.unlink()

    def test_a_publisher_without_the_configured_streams_fails_clearly(self, rgbd_store):
        frame = MagicMock()
        frame.data, frame.timestamp = np.zeros((6, 8, 3), np.uint8), 1.0  # color only, from an older publisher
        camera = MagicMock()
        camera.read_latest.return_value = frame
        with pytest.raises(RuntimeError, match="without the color and depth streams"):
            asyncio.run(CameraWorker(_realsense_config(), rgbd_store, "top").run_loop(camera))
        assert rgbd_store.read_observation("top") is None
        assert rgbd_store.read_observation("top_depth") is None
