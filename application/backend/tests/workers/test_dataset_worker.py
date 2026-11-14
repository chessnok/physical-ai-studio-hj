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
    for _ in range(3):
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


class _DroppingMutation(MagicMock):
    """A recording whose video encoder drops frames once ``dropped`` is set."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropped: dict[str, int] = {}
        self.problems: list[str] = []

    def dropped_frames(self) -> dict[str, int]:
        return self.dropped

    def saved_episode_problems(self) -> list[str]:
        return self.problems


@pytest.fixture
def dropping_recorder():
    store = Store.create(FeatureSpec((), {"top": ImageSpec(2, 2), "top_depth": ImageSpec.depth(2, 2)}), uuid4().hex)
    for key in (STATE, ACTION):
        store.write(key, [], timestamp=1.0)
    store.write_observation("top", np.zeros((2, 2, 3), np.uint8), timestamp=1.0)
    store.write_observation("top_depth", np.zeros((2, 2, 1), np.uint16), timestamp=1.0)
    recorder = Recorder(_DroppingMutation(), store, publish=MagicMock(), fps=30)
    recorder.start(StartEpisodeCommand(request_id="start", task="pick"))
    yield recorder
    store.unlink()


def test_an_encoder_that_drops_frames_discards_the_episode_and_says_why(dropping_recorder):
    recorder = dropping_recorder
    recorder.record_frame()
    assert recorder.status().is_recording
    recorder.mutation.dropped = {"observation.images.top_depth": 3}

    recorder.record_frame()

    assert not recorder.status().is_recording
    recorder.mutation.discard_buffer.assert_called_once()
    error = recorder.publish.call_args.args[0].error
    assert "dropped frames (3 of top_depth)" in error


def test_an_episode_with_dropped_frames_is_not_saved(dropping_recorder):
    recorder = dropping_recorder
    for _ in range(3):
        recorder.record_frame()
    recorder.mutation.dropped = {"observation.images.top_depth": 1}

    with pytest.raises(RuntimeSessionCommandError, match="could not keep up") as error:
        recorder.save(SaveEpisodeCommand(request_id="save"))

    assert error.value.error_code == "encoder_dropped_frames"
    recorder.mutation.save_episode.assert_not_called()
    recorder.mutation.discard_buffer.assert_called_once()


def test_a_recording_that_falls_behind_its_clock_is_discarded(dropping_recorder, monkeypatch):
    recorder = dropping_recorder
    clock = [100.0]
    monkeypatch.setattr("workers.dataset_worker.time.monotonic", lambda: clock[0])
    recorder.record_frame()
    for _ in range(5):  # on time: one row per 1/30 s
        clock[0] += 1 / 30
        recorder.record_frame()
    assert recorder.status().is_recording

    clock[0] += 1.0  # the computer stalled for a second: 30 ticks missed

    recorder.record_frame()
    assert not recorder.status().is_recording
    assert "behind its 30 fps clock" in recorder.publish.call_args.args[0].error


def test_small_hiccups_over_a_long_episode_are_kept(dropping_recorder, monkeypatch):
    """0.4 s of accumulated hiccups over a one-minute episode (0.7 % fast) must not discard it."""
    recorder = dropping_recorder
    clock = [100.0]
    monkeypatch.setattr("workers.dataset_worker.time.monotonic", lambda: clock[0])
    recorder.record_frame()
    for i in range(1, 1800):  # 60 s at 30 fps
        clock[0] += 1 / 30 + (0.1 if i % 450 == 0 else 0.0)  # four 0.1 s hiccups
        recorder.record_frame()
    assert recorder.status().is_recording


def test_a_saved_episode_whose_videos_do_not_match_is_flagged(dropping_recorder):
    recorder = dropping_recorder
    for _ in range(3):
        recorder.record_frame()
    recorder.mutation.problems = ["top_depth: 2 video frames for 3 rows"]

    status = recorder.save(SaveEpisodeCommand(request_id="save"))

    assert status.episodes_recorded == 1
    assert "do not match its rows (top_depth: 2 video frames for 3 rows)" in status.error
