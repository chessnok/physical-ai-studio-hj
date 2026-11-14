"""RealSense RGB-D capture: frame packing, mode selection and the pipeline it starts, without hardware."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from physicalai.capture.errors import CaptureError

from utils.realsense_rgbd import (
    RealSenseModeError,
    RealSenseRGBDCamera,
    StreamPlan,
    pack_rgbd,
    packed_channels,
    select_streams,
    unpack_rgbd,
)

# What a D415 offers on USB 3 and on USB 2.1 (rgb8 color, z16 depth), as pyrealsense2 lists it.
USB3_COLOR = {(1920, 1080, 30), (1920, 1080, 15), (1280, 720, 30), (1280, 720, 15), (640, 480, 30), (960, 540, 60)}
USB3_DEPTH = {(1280, 720, 30), (1280, 720, 15), (848, 480, 30), (848, 480, 60), (640, 480, 30), (640, 360, 30)}
USB2_COLOR = {(1280, 720, 6), (1280, 720, 15), (640, 480, 30), (1920, 1080, 8)}
USB2_DEPTH = {(1280, 720, 6), (640, 480, 30), (848, 480, 10)}


class TestPacking:
    def test_color_only_is_the_color_image_itself(self):
        color = np.random.default_rng(0).integers(0, 255, (4, 6, 3), dtype=np.uint8)
        packed = pack_rgbd(color, None)
        assert packed is color
        assert unpack_rgbd(packed) == (color, None)

    @pytest.mark.parametrize("with_color", [True, False])
    def test_depth_survives_packing_bit_exact(self, with_color):
        rng = np.random.default_rng(1)
        color = rng.integers(0, 255, (4, 6, 3), dtype=np.uint8) if with_color else None
        depth = rng.integers(0, 65535, (4, 6), dtype=np.uint16)
        packed = pack_rgbd(color, depth)
        assert packed.dtype == np.uint8
        assert packed.shape == (4, 6, 5 if with_color else 2)
        unpacked_color, unpacked_depth = unpack_rgbd(packed)
        assert unpacked_depth.dtype == np.uint16
        assert unpacked_depth.shape == (4, 6, 1)
        np.testing.assert_array_equal(unpacked_depth[..., 0], depth)
        if with_color:
            np.testing.assert_array_equal(unpacked_color, color)
        else:
            assert unpacked_color is None

    def test_packed_channels_by_output_type(self):
        assert [packed_channels(output) for output in ("color", "depth", "both")] == [3, 2, 5]

    def test_mismatched_color_and_depth_are_refused(self):
        with pytest.raises(ValueError, match="does not match depth"):
            pack_rgbd(np.zeros((4, 6, 3), np.uint8), np.zeros((4, 5), np.uint16))

    def test_other_frames_are_not_unpacked(self):
        with pytest.raises(ValueError, match="Not a packed RealSense frame"):
            unpack_rgbd(np.zeros((4, 6, 4), np.uint8))
        with pytest.raises(ValueError, match="Not a packed RealSense frame"):
            unpack_rgbd(np.zeros((4, 6), np.uint16))


class TestSelectStreams:
    def _select(self, output_type, width=1280, height=720, fps=30, *, usb2=False, depth_size=None):
        return select_streams(
            USB2_COLOR if usb2 else USB3_COLOR,
            USB2_DEPTH if usb2 else USB3_DEPTH,
            width=width,
            height=height,
            fps=fps,
            output_type=output_type,
            depth_size=depth_size,
            usb_type="2.1" if usb2 else "3.2",
            device="RealSense D415 123",
        )

    def test_color_streams_color_only(self):
        assert self._select("color", 1920, 1080, 15) == StreamPlan(color=(1920, 1080, 15), depth=None)

    def test_depth_streams_depth_only_at_the_requested_mode(self):
        assert self._select("depth") == StreamPlan(color=None, depth=(1280, 720, 30))

    def test_both_at_1280x720_30_streams_depth_at_the_color_resolution(self):
        assert self._select("both") == StreamPlan(color=(1280, 720, 30), depth=(1280, 720, 30))

    def test_both_with_a_color_mode_the_depth_sensor_lacks_aligns_its_best_depth_mode(self):
        # Depth is aligned to color, so a 1920x1080 color image gets 1280x720 depth upsampled to it.
        assert self._select("both", 1920, 1080, 30).depth == (1280, 720, 30)
        assert self._select("both", 960, 540, 60).depth == (848, 480, 60)

    def test_explicit_depth_size(self):
        assert self._select("both", depth_size=(848, 480)).depth == (848, 480, 30)
        with pytest.raises(RealSenseModeError, match="no depth mode 640x360@15fps"):
            self._select("both", 1920, 1080, 15, depth_size=(640, 360))

    def test_usb2_refuses_1280x720_30_and_says_why(self):
        with pytest.raises(RealSenseModeError) as error:
            self._select("both", usb2=True)
        message = str(error.value)
        assert "no color mode 1280x720@30fps" in message
        assert "1280x720@15fps" in message  # what it can do instead
        assert "USB 2.1" in message
        assert "USB 3" in message

    def test_usb2_has_no_depth_at_15fps_to_align(self):
        with pytest.raises(RealSenseModeError, match="no depth mode at 15 fps"):
            self._select("both", 1280, 720, 15, usb2=True)

    def test_unknown_output_type(self):
        with pytest.raises(ValueError, match="Unknown RealSense output type"):
            self._select("infrared")


class _FakeFrame:
    def __init__(self, data):
        self._data = data

    def get_data(self):
        return self._data

    def __bool__(self):
        return True


class _FakeFrameset:
    def __init__(self, color, depth):
        self.color, self.depth = color, depth

    def get_color_frame(self):
        return None if self.color is None else _FakeFrame(self.color)

    def get_depth_frame(self):
        return None if self.depth is None else _FakeFrame(self.depth)


class _FakeRealSense:
    """Just enough of pyrealsense2 for one D415 at 1280x720@30 on USB 3."""

    def __init__(self, usb_type="3.2"):
        self.enabled: list[tuple] = []
        self.started = self.stopped = False
        self.aligned = 0
        self.usb_type = usb_type
        self.stream = SimpleNamespace(color="color", depth="depth")
        self.format = SimpleNamespace(rgb8="rgb8", bgr8="bgr8", z16="z16")
        self.camera_info = SimpleNamespace(name="name", serial_number="serial", usb_type_descriptor="usb")
        fake = self

        class Device:
            def supports(self, key):
                return True

            def get_info(self, key):
                return {"name": "Intel RealSense D415", "serial": "123", "usb": fake.usb_type}[key]

            def query_sensors(self):
                profiles = [_profile("color", "rgb8", *mode) for mode in USB3_COLOR]
                profiles += [_profile("depth", "z16", *mode) for mode in USB3_DEPTH]
                profiles.append(_profile("color", "yuyv", 1920, 1080, 60))  # not rgb8: ignored
                return [SimpleNamespace(get_stream_profiles=lambda: profiles)]

        class Config:
            def enable_device(self, serial):
                assert serial == "123"

            def enable_stream(self, *args):
                fake.enabled.append(args)

        class Pipeline:
            def start(self, config):
                fake.started = True

            def stop(self):
                fake.stopped = True

            def wait_for_frames(self, timeout_ms):
                return fake.frameset()

            def poll_for_frames(self):
                return None

        class Align:
            def __init__(self, stream):
                assert stream == "color"

            def process(self, frameset):
                fake.aligned += 1
                return frameset

        self.context = lambda: SimpleNamespace(query_devices=lambda: [Device()])
        self.config, self.pipeline, self.align = Config, Pipeline, Align

    def frameset(self):
        color = np.full((720, 1280, 3), 7, np.uint8)
        depth = np.full((720, 1280), 1234, np.uint16)
        return _FakeFrameset(color, depth)


def _profile(stream, fmt, width, height, fps):
    video = SimpleNamespace(width=lambda: width, height=lambda: height)
    return SimpleNamespace(
        stream_type=lambda: stream, format=lambda: fmt, fps=lambda: fps, as_video_stream_profile=lambda: video
    )


@pytest.fixture
def fake_rs(monkeypatch: pytest.MonkeyPatch):
    # Only this entry: restoring all of sys.modules would drop modules first imported meanwhile.
    fake = _FakeRealSense()
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake)
    return fake


class TestRealSenseRGBDCamera:
    def test_both_streams_color_and_aligned_depth_in_one_frame(self, fake_rs):
        camera = RealSenseRGBDCamera(serial_number="123", width=1280, height=720, fps=30, output_type="both")
        camera.connect()
        assert fake_rs.enabled == [("color", 1280, 720, "rgb8", 30), ("depth", 1280, 720, "z16", 30)]
        frame = camera.read()
        color, depth = unpack_rgbd(frame.data)
        assert np.all(color == 7)
        assert np.all(depth == 1234)
        assert fake_rs.aligned == 2  # the first frameset on connect, then this one
        assert camera.read_latest() is frame  # nothing newer arrived
        camera.disconnect()
        assert fake_rs.stopped

    def test_color_only_streams_no_depth_so_1920x1080_works(self, fake_rs):
        fake_rs.frameset = lambda: _FakeFrameset(np.zeros((1080, 1920, 3), np.uint8), None)
        camera = RealSenseRGBDCamera(serial_number="123", width=1920, height=1080, fps=15, output_type="color")
        camera.connect()
        assert fake_rs.enabled == [("color", 1920, 1080, "rgb8", 15)]
        assert camera.read().data.shape == (1080, 1920, 3)
        assert fake_rs.aligned == 0

    def test_an_unavailable_mode_fails_before_starting_the_pipeline(self, fake_rs):
        camera = RealSenseRGBDCamera(serial_number="123", width=1280, height=720, fps=60, output_type="both")
        with pytest.raises(RealSenseModeError, match="no color mode 1280x720@60fps"):
            camera.connect()
        assert not fake_rs.started
        assert not camera.is_connected

    def test_a_missing_device_names_the_connected_ones(self, fake_rs):
        with pytest.raises(CaptureError, match=r"RealSense 999 is not connected .*123"):
            RealSenseRGBDCamera(serial_number="999").connect()

    def test_query_formats_by_output_type(self, fake_rs):
        assert RealSenseRGBDCamera.query_formats({"serial": "123"}, "depth") == sorted(USB3_DEPTH)
        assert RealSenseRGBDCamera.query_formats("123", "color") == sorted(USB3_COLOR)
        # Color modes that have a depth mode at their fps to align; every fps has one on USB 3.
        assert RealSenseRGBDCamera.query_formats("123", "both") == sorted(USB3_COLOR)
        assert RealSenseRGBDCamera.query_formats("999", "both") == []

    def test_is_built_from_a_config_like_the_publisher_does(self, fake_rs):
        from physicalai.capture.camera import Camera
        from physicalai.config import Config

        camera = Config(
            "utils.realsense_rgbd.RealSenseRGBDCamera",
            {"serial_number": "123", "width": 1280, "height": 720, "fps": 30, "output_type": "depth"},
        ).instantiate(expected_type=Camera)
        assert isinstance(camera, RealSenseRGBDCamera)
        assert camera.output_type == "depth"
