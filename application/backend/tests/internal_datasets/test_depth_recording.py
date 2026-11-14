"""Depth maps in LeRobot datasets: how they are encoded, and how people look at them."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import av
import numpy as np
import pytest

from internal_datasets.access_mode import DatasetAccessMode
from internal_datasets.lerobot import video_encoding_settings
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from internal_datasets.lerobot.video_encoding_settings import (
    StudioDepthEncoderConfig,
    VideoEncodingSettings,
    depth_features,
)

if TYPE_CHECKING:
    from pathlib import Path

HEIGHT, WIDTH = 64, 96
FEATURES = {
    "action": {"dtype": "float32", "shape": (2,), "names": ["arm.pos", "gripper.pos"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["arm.pos", "gripper.pos"]},
    "observation.images.top": {"dtype": "video", "shape": (HEIGHT, WIDTH, 3), "names": ["height", "width", "channels"]},
    "observation.images.top_depth": {
        "dtype": "video",
        "shape": (HEIGHT, WIDTH, 1),
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": True},
    },
}


def _depth(frame: int) -> np.ndarray:
    depth = np.tile(np.linspace(250, 4000, WIDTH, dtype=np.float32), (HEIGHT, 1)).astype(np.uint16) + frame
    depth[:8, :8] = 0
    return depth[..., None]


def test_depth_is_lossless_12_bit_hevc_with_a_real_time_preset() -> None:
    encoder = StudioDepthEncoderConfig()
    options = encoder.get_codec_options(as_strings=True)
    assert (encoder.vcodec, encoder.pix_fmt) == ("hevc", "gray12le")
    assert options["preset"] == "ultrafast"
    assert "lossless=1" in options["x265-params"]
    assert "crf" not in options
    assert (encoder.depth_min, encoder.depth_max, encoder.use_log) == (0.0, 10.0, True)


def test_a_crf_records_depth_lossy() -> None:
    options = StudioDepthEncoderConfig(crf=8).get_codec_options(as_strings=True)
    assert options["crf"] == "8"
    assert "lossless" not in options["x265-params"]


def test_appending_keeps_the_quantization_a_dataset_was_recorded_with() -> None:
    recorded = {"video.depth_min": 0.01, "video.depth_max": 4.0, "video.shift": 3.5, "video.use_log": False}
    encoder = StudioDepthEncoderConfig.for_recording(None, recorded)
    assert (encoder.depth_min, encoder.depth_max, encoder.use_log) == (0.01, 4.0, False)


def test_only_datasets_with_depth_get_a_real_depth_encoder() -> None:
    settings = VideoEncodingSettings(vcodec="libsvtav1", depth_crf=8)
    rgb_only = settings.to_lerobot_write_kwargs({})
    assert isinstance(rgb_only["depth_encoder"], video_encoding_settings._UnsupportedDepthEncoderConfig)
    rgbd = settings.to_lerobot_write_kwargs(depth_features(FEATURES))
    assert isinstance(rgbd["depth_encoder"], StudioDepthEncoderConfig)
    assert rgbd["depth_encoder"].crf == 8
    assert list(depth_features(FEATURES)) == ["observation.images.top_depth"]


def test_a_backend_without_a_12_bit_hevc_encoder_refuses_depth_clearly() -> None:
    with (
        patch.object(video_encoding_settings, "_is_depth_encoder_usable", return_value=False),
        pytest.raises(RuntimeError, match="Depth recording needs the 'hevc' video encoder"),
    ):
        StudioDepthEncoderConfig()


@pytest.fixture
def rgbd_dataset(tmp_path: Path) -> InternalLeRobotDataset:
    dataset = InternalLeRobotDataset(
        tmp_path / "dataset",
        recording_video_encoding_settings=VideoEncodingSettings(vcodec="libsvtav1", encoder_threads=1),
        access_mode=DatasetAccessMode.RECORDING_MUTATION,
    )
    dataset.create(fps=30, features=FEATURES, robot_type="test")
    dataset.prepare_for_writing()
    for frame in range(6):
        dataset.add_frame(
            {
                "arm.pos": 0.0,
                "gripper.pos": 1.0,
                "top": np.full((HEIGHT, WIDTH, 3), 40 * frame, np.uint8),
                "top_depth": _depth(frame),
            },
            {"arm.pos": 0.0, "gripper.pos": 1.0},
            "test",
        )
    dataset.save_episode()
    dataset.finalize()
    return InternalLeRobotDataset(dataset.path)


def test_episodes_say_which_videos_are_depth(rgbd_dataset: InternalLeRobotDataset) -> None:
    videos = rgbd_dataset.get_episodes()[0].videos
    assert not videos["observation.images.top"].is_depth
    assert videos["observation.images.top_depth"].is_depth
    assert rgbd_dataset.get_depth_keys() == ["observation.images.top_depth"]
    assert rgbd_dataset.get_feature_info("observation.images.top_depth")["video.depth_max"] == 10.0


def test_a_depth_thumbnail_is_colorized(rgbd_dataset: InternalLeRobotDataset) -> None:
    import cv2

    result = rgbd_dataset.get_episode_thumbnail_png(0, "observation.images.top_depth", WIDTH, HEIGHT)
    assert result is not None
    image = cv2.imdecode(np.frombuffer(result[0], np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape == (HEIGHT, WIDTH, 3)
    assert np.all(image[:8, :8] == 0)  # no depth is black
    assert len(np.unique(image.reshape(-1, 3), axis=0)) > 10  # near to far is a gradient


def test_a_depth_preview_plays_in_browsers_with_the_same_timestamps(
    rgbd_dataset: InternalLeRobotDataset, tmp_path: Path
) -> None:
    from internal_datasets.lerobot.depth_video import ensure_depth_preview

    source = rgbd_dataset.get_video_path(0, "top_depth")
    preview = ensure_depth_preview(
        source, tmp_path / "cache", rgbd_dataset.get_feature_info("observation.images.top_depth")
    )
    assert preview == ensure_depth_preview(source, tmp_path / "cache", {})  # cached
    with av.open(str(source)) as original, av.open(str(preview)) as copy:
        original_times = [float(frame.time) for frame in original.decode(video=0)]
        stream = copy.streams.video[0]
        assert stream.codec_context.name in {"h264", "libsvtav1", "av1"}
        frames = list(copy.decode(stream))
        assert [float(frame.time) for frame in frames] == pytest.approx(original_times)
        assert frames[0].to_ndarray(format="rgb24").shape == (HEIGHT, WIDTH, 3)


def test_lerobot_reads_back_the_recorded_millimetres(rgbd_dataset: InternalLeRobotDataset) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    lerobot = LeRobotDataset("depth", root=rgbd_dataset.path)
    assert lerobot.meta.depth_keys == ["observation.images.top_depth"]
    for index in (0, 5):
        depth = lerobot[index]["observation.images.top_depth"].numpy()[0]
        expected = _depth(index)[..., 0].astype(np.float32)
        assert np.all(depth[:8, :8] == 0)
        # 12-bit log codes over 0-10 m: within half a step, under 1 mm up to 1 m and 1.5 mm at 4 m.
        assert np.abs(depth - expected).max() <= 1.5
