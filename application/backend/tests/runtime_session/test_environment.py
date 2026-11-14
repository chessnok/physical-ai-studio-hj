from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from runtime_session.bus import publish_status
from runtime_session.environment import _enter_concurrently, describe_environment
from runtime_session.store import ImageSpec
from tests.workers.fake_robot import FakeRobot


@pytest.fixture
def factory(mock_robot_client_factory, tmp_path):
    driver = FakeRobot(str(tmp_path / "log"), ["shoulder_pan", "gripper"], [0.0, 0.0])
    mock_robot_client_factory.build_robot_driver = AsyncMock(return_value=(driver, object()))
    return mock_robot_client_factory


async def test_a_slow_device_cannot_overwrite_a_connected_cameras_reconnect_warning() -> None:
    bus = MagicMock()
    camera_connected = asyncio.Event()
    camera_exited = MagicMock()

    @asynccontextmanager
    async def camera():
        camera_connected.set()
        try:
            yield "camera"
        finally:
            camera_exited()

    @asynccontextmanager
    async def slow_device():
        await camera_connected.wait()
        publish_status(bus, "environment/cam", "starting", "no frames; reconnecting")
        yield "follower"

    async with AsyncExitStack() as stack:
        started = await _enter_concurrently(stack, bus, {"cam": camera(), "follower": slow_device()})
        assert started == {"cam": "camera", "follower": "follower"}
        statuses = [call.args[1] for call in bus.publish.call_args_list if call.args[1].component == "environment/cam"]
        assert [status.state for status in statuses] == ["starting", "running", "starting"]
        assert statuses[-1].error == "no frames; reconnecting"
        camera_exited.assert_not_called()
    camera_exited.assert_called_once()


async def test_spec_comes_from_the_driver_and_the_camera_rows(test_environment, factory) -> None:
    description = await describe_environment(test_environment, factory)

    assert description.spec.joint_names == ("shoulder_pan.pos", "gripper.pos")
    assert description.spec.observations == {"grabber": ImageSpec(480, 640), "front": ImageSpec(480, 640)}
    assert description.follower.robot_id == test_environment.robots[0].robot.id
    assert set(description.cameras) == {"grabber", "front"}


async def test_more_than_one_robot_is_refused(test_environment, factory) -> None:
    test_environment.robots.append(test_environment.robots[0])
    with pytest.raises(ValueError, match="has 2 robots"):
        await describe_environment(test_environment, factory)


async def test_cameras_that_would_share_a_key_are_refused(test_environment, factory) -> None:
    test_environment.cameras[1].name = "Grabber"
    with pytest.raises(ValueError, match="'grabber'"):
        await describe_environment(test_environment, factory)


def _realsense_row(name: str, output_type: str) -> dict:
    return {
        "id": "5b8f3a3e-8d5e-4c55-9d43-2f1f2b9c1a01",
        "driver": "realsense",
        "name": name,
        "fingerprint": {"serial": "243622062352"},
        "hardware_name": "Intel RealSense D415",
        "payload": {"width": 1280, "height": 720, "fps": 30, "output_type": output_type},
    }


@pytest.mark.parametrize(
    ("output_type", "slots"),
    [
        ("color", {"top": ImageSpec(720, 1280)}),
        ("depth", {"top_depth": ImageSpec.depth(360, 640)}),
        ("both", {"top": ImageSpec(720, 1280), "top_depth": ImageSpec.depth(360, 640)}),
    ],
)
async def test_a_realsense_camera_records_what_its_output_type_asks(
    test_environment, factory, output_type, slots
) -> None:
    from schemas.project_camera import CameraAdapter

    test_environment.cameras.append(CameraAdapter.validate_python(_realsense_row("Top", output_type)))
    description = await describe_environment(test_environment, factory)

    observations = dict(description.spec.observations)
    assert {key: observations.pop(key) for key in slots} == slots
    assert set(observations) == {"grabber", "front"}
    shown = description.session_environment().cameras[-1]
    assert shown.key == ("top_depth" if output_type == "depth" else "top")
    assert shown.depth_key == (None if output_type == "color" else "top_depth")
    assert (shown.height, shown.width) == ((360, 640) if output_type == "depth" else (720, 1280))


async def test_a_camera_named_like_another_ones_depth_is_refused(test_environment, factory) -> None:
    from schemas.project_camera import CameraAdapter

    test_environment.cameras.append(CameraAdapter.validate_python(_realsense_row("Top", "both")))
    test_environment.cameras[0].name = "top_depth"
    with pytest.raises(ValueError, match="'top_depth'"):
        await describe_environment(test_environment, factory)


def test_a_session_shows_a_cameras_color_and_a_depth_only_cameras_depth() -> None:
    from api.runtime_ws import _shown_frames

    assert _shown_frames(["top", "top_depth", "floor_depth", "arm"]) == ["top", "floor_depth", "arm"]


async def test_one_camera_twice_in_an_environment_is_refused(test_environment, factory) -> None:
    from schemas.project_camera import CameraAdapter

    for output_type, name in (("color", "Top color"), ("depth", "Top depth")):
        row = _realsense_row(name, output_type)
        row["id"] = str(uuid4())
        test_environment.cameras.append(CameraAdapter.validate_python(row))
    with pytest.raises(ValueError, match=r"'Top color' and 'Top depth' are the same device.*Color \+ depth"):
        await describe_environment(test_environment, factory)


async def test_saving_an_environment_with_one_camera_twice_is_refused(test_environment, mocker) -> None:
    from exceptions import CameraDeviceConflictError
    from schemas.environment import Environment
    from schemas.project_camera import CameraAdapter
    from services.environment_service import EnvironmentService

    cameras = [CameraAdapter.validate_python(_realsense_row(name, "color") | {"id": str(uuid4())}) for name in "AB"]
    mocker.patch("services.environment_service.ProjectCameraRepository.get_all", AsyncMock(return_value=cameras))
    service = EnvironmentService(MagicMock(), MagicMock())
    environment = Environment(id=uuid4(), name="Bench", cameras=[{"camera_id": camera.id} for camera in cameras])
    with pytest.raises(CameraDeviceConflictError, match="same device"):
        await service.create_environment(uuid4(), environment)
