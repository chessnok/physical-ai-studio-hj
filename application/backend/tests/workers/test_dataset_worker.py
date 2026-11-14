from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest

from exceptions import RuntimeSessionCommandError
from runtime_session.store import ACTION, STATE, FeatureSpec, ImageSpec, Store
from schemas.runtime_session import SaveEpisodeCommand, StartEpisodeCommand, StatusMessage
from workers.dataset_worker import Recorder


@pytest.fixture
def recorder():
    store = Store.create(FeatureSpec((), {"cam": ImageSpec(2, 2)}), uuid4().hex)
    for key in (STATE, ACTION):
        store.write(key, [], timestamp=1.0)
    store.write_observation("cam", np.zeros((2, 2, 3), dtype=np.uint8), timestamp=1.0)
    recorder = Recorder(MagicMock(), store, publish=MagicMock(), episodes_recorded=1)
    recorder.start(StartEpisodeCommand(request_id="start", task="pick"))
    yield recorder
    store.unlink()


def test_missing_camera_discards_only_the_open_episode(recorder):
    recorder.record_frame()
    recorder.store.invalidate_observation("cam")
    recorder.record_frame()
    recorder.record_frame()

    recorder.mutation.discard_buffer.assert_called_once()
    assert recorder.status().episodes_recorded == 1
    assert not recorder.status().is_recording
    status = recorder.publish.call_args.args[0]
    assert 'camera "cam"' in status.error
    assert "Episode discarded" in status.error
    with pytest.raises(RuntimeSessionCommandError, match="No episode"):
        recorder.save(SaveEpisodeCommand(request_id="save"))
    recorder.mutation.save_episode.assert_not_called()


@pytest.mark.parametrize("state", ["starting", "failed"])
def test_camera_loss_between_recording_ticks_discards_the_episode(recorder, state):
    recorder.record_frame()
    # The slot can already have a fresh frame again when the queued status arrives.
    recorder.camera_status(StatusMessage(component="environment/cam", state=state, error="camera lost"))
    recorder.record_frame()

    recorder.mutation.discard_buffer.assert_called_once()
    assert not recorder.status().is_recording
    assert recorder.publish.call_args.args[0].error is not None

    recorder.start(StartEpisodeCommand(request_id="restart", task="pick"))
    assert recorder.publish.call_args.args[0].error is None
    recorder.record_frame()
    recorder.record_frame()
    assert recorder.save(SaveEpisodeCommand(request_id="save")).episodes_recorded == 2


@pytest.mark.parametrize(
    ("component", "state"),
    [("environment/cam", "running"), ("environment/follower", "starting"), ("environment/other", "failed")],
)
def test_unrelated_status_does_not_discard_an_episode(recorder, component, state):
    recorder.camera_status(StatusMessage(component=component, state=state))

    recorder.mutation.discard_buffer.assert_not_called()
    assert recorder.status().is_recording


def test_camera_loss_while_idle_does_not_discard_or_notify(recorder):
    recorder.discard()
    recorder.publish.reset_mock()
    recorder.mutation.discard_buffer.reset_mock()

    recorder.camera_status(StatusMessage(component="environment/cam", state="starting"))

    recorder.publish.assert_not_called()
    recorder.mutation.discard_buffer.assert_not_called()


def test_a_color_image_and_its_depth_map_are_recorded_from_the_same_frameset():
    store = Store.create(FeatureSpec((), {"top": ImageSpec(2, 2), "top_depth": ImageSpec.depth(2, 2)}), uuid4().hex)
    try:
        for key in (STATE, ACTION):
            store.write(key, [], timestamp=1.0)
        store.write_observation("top", np.full((2, 2, 3), 1, np.uint8), timestamp=1.0)
        store.write_observation("top_depth", np.full((2, 2, 1), 100, np.uint16), timestamp=1.0)
        recorder = Recorder(MagicMock(), store, publish=MagicMock())
        recorder.start(StartEpisodeCommand(request_id="start", task="pick"))

        # The camera writes the next frameset while the recorder reads: it gets the new color image
        # and the old depth map, then the camera writes the new depth map.
        store.write_observation("top", np.full((2, 2, 3), 2, np.uint8), timestamp=2.0)
        original = store.read_observation
        depth_reads = 0

        def read_observation(key):
            nonlocal depth_reads
            sample = original(key)
            if key == "top_depth":
                depth_reads += 1
                if depth_reads == 1:
                    store.write_observation("top_depth", np.full((2, 2, 1), 200, np.uint16), timestamp=2.0)
            return sample

        store.read_observation = read_observation
        recorder.record_frame()

        observation = recorder.mutation.add_frame.call_args.args[0]
        assert np.all(observation["top"] == 2)
        assert np.all(observation["top_depth"] == 200)
        assert depth_reads == 2
    finally:
        store.unlink()


def test_a_depth_only_camera_lost_between_ticks_discards_the_episode():
    store = Store.create(FeatureSpec((), {"top_depth": ImageSpec.depth(2, 2)}), uuid4().hex)
    try:
        recorder = Recorder(MagicMock(), store, publish=MagicMock())
        recorder.start(StartEpisodeCommand(request_id="start", task="pick"))
        recorder.camera_status(StatusMessage(component="environment/top", state="starting", error="lost"))
        recorder.mutation.discard_buffer.assert_called_once()
    finally:
        store.unlink()
