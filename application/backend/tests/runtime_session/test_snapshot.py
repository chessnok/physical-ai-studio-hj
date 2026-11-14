"""State snapshots of a live session: what they save, and that taking one changes nothing."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_runtime_session_registry
from main import app
from runtime_session import snapshot, snapshot_index
from runtime_session.environment import describe_environment
from runtime_session.registry import RuntimeSessionRegistry
from runtime_session.session import LoadedEnvironment
from runtime_session.store import ACTION, STATE, Store
from schemas.project_camera import CameraAdapter
from schemas.runtime_session import SessionState, TeleopControlConfig
from settings import get_settings
from tests.workers.fake_robot import FakeRobot
from utils.camera_factory import camera_stream_info_path

from .fixtures import JOINTS, make_environment

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

PROJECT_ID = uuid4()


class TickingRobot(FakeRobot):
    """A driver whose positions convert to servo ticks, ten per unit, like a calibrated one."""

    def _unit_to_ticks(self, values: np.ndarray) -> np.ndarray:
        return np.round(np.asarray(values) * 10).astype(np.int32)


def _color(value: int) -> np.ndarray:
    image = np.zeros((240, 424, 3), np.uint8)
    image[..., 0], image[..., 1] = value, np.arange(424, dtype=np.uint16)[None, :] % 256
    return image


def _depth() -> np.ndarray:
    """The depth slot's frame: aligned to color, decimated by DEPTH_RECORD_DOWNSCALE (2)."""
    depth = np.tile(np.linspace(400, 2500, 212).astype(np.uint16), (120, 1))
    depth[:10, :10] = 0
    return depth[..., None]


class _Session(SimpleNamespace):
    def state(self) -> SessionState:
        return SessionState(session_id=self.id, control=self.control)


@pytest.fixture
async def live(mock_robot_client_factory, tmp_path) -> AsyncIterator[_Session]:
    """A live session: a follower with a leader, a USB camera and a RealSense with color and depth."""

    async def build(robot, _finder):
        return TickingRobot(str(tmp_path / f"{robot.id}.jsonl"), JOINTS, [0.0, 0.0]), object()

    mock_robot_client_factory.build_robot_driver = AsyncMock(side_effect=build)
    row = make_environment(cameras=1)
    row.cameras.append(
        CameraAdapter.validate_python(
            {
                "id": str(uuid4()),
                "driver": "realsense",
                "name": "Top",
                "fingerprint": {"serial": "243622062352"},
                "hardware_name": "Intel RealSense D415",
                "payload": {"width": 424, "height": 240, "fps": 30, "output_type": "both", "depth_range_max": 2.0},
            }
        )
    )
    row.cameras[0].payload.width, row.cameras[0].payload.height = 424, 240
    description = await describe_environment(row, mock_robot_client_factory)
    store = Store.create(description.spec, uuid4().hex)
    store.write(STATE, [12.5, 40.0], timestamp=10.0)
    store.write(ACTION, [13.0, 41.5], timestamp=10.02)
    store.write_observation("cam 0", _color(7), timestamp=9.99)
    store.write_observation("top", _color(9), timestamp=9.98)
    store.write_observation("top_depth", _depth(), timestamp=9.98)
    info_path = camera_stream_info_path(row.cameras[1])
    assert info_path is not None
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(
        json.dumps(
            {
                "color_intrinsics": {"fx": 910.0, "ppx": 212.0, "width": 424, "height": 240},
                "depth_intrinsics": {"fx": 910.0, "ppx": 212.0, "width": 424, "height": 240},
                "depth_scale_m": 0.001,
            }
        )
    )
    loaded = LoadedEnvironment(
        environment=SimpleNamespace(description=description),  # type: ignore[arg-type]
        state=description.session_environment(),
        robot_type="SO101_Follower",
        row=row,
    )
    session = _Session(
        id="session-1",
        project_id=PROJECT_ID,
        project_name="Junction",
        store=store,
        loaded_environment=loaded,
        control=TeleopControlConfig(),
        robot_logs=[tmp_path / f"{robot.robot.id}.jsonl" for robot in row.robots],
    )
    yield session
    store.unlink()


def _stamps(store: Store) -> list[float]:
    return list(store._stamps)


def test_a_snapshot_saves_frames_and_state_without_touching_the_session(live, tmp_path) -> None:
    before = _stamps(live.store)

    result = snapshot.write(snapshot.capture(live), tmp_path / "snapshots", build={"git_sha": "abc"})

    folder = Path(result.path)
    assert folder.parent == tmp_path / "snapshots" / str(PROJECT_ID)
    assert {camera.key: camera.files for camera in result.cameras} == {
        "cam 0": ["cam_0.color.png"],
        "top": ["top.color.png", "top.depth.png", "top.depth_preview.png"],
    }
    assert result.follower and result.leader
    # Lossless: the PNGs hold exactly the store's frames.
    color, depth, preview = (
        cv2.imread(str(folder / name), flag)
        for name, flag in (
            ("top.color.png", cv2.IMREAD_COLOR),
            ("top.depth.png", cv2.IMREAD_UNCHANGED),
            ("top.depth_preview.png", cv2.IMREAD_COLOR),
        )
    )
    assert color is not None and depth is not None and preview is not None
    np.testing.assert_array_equal(cv2.cvtColor(color, cv2.COLOR_BGR2RGB), _color(9))
    assert depth.dtype == np.uint16
    np.testing.assert_array_equal(depth, _depth()[..., 0])
    assert preview.shape == (120, 212, 3)
    assert np.all(preview[:10, :10] == 0)  # no depth is black

    # Read-only: nothing written to the store, nothing sent to a robot.
    assert _stamps(live.store) == before
    assert not any(log.exists() for log in live.robot_logs)


def test_state_json_says_what_was_measured_when(live, tmp_path) -> None:
    result = snapshot.write(snapshot.capture(live), tmp_path, build={"git_sha": "abc"})
    state = json.loads((Path(result.path) / "state.json").read_text())

    assert state["format"] == snapshot.FORMAT
    assert state["app"] == {"git_sha": "abc"}
    assert state["project"] == {"id": str(PROJECT_ID), "name": "Junction"}
    assert state["session"]["control"]["type"] == "teleop"
    follower, leader = state["robots"]["follower"], state["robots"]["leader"]
    assert follower["joint_positions"] == {"shoulder_pan.pos": 12.5, "gripper.pos": 40.0}
    assert follower["sample"]["monotonic"] == 10.0
    assert follower["serial_number"] == "Follower"
    assert follower["raw_ticks"]["values"] == {"shoulder_pan.pos": 125, "gripper.pos": 400}
    assert "not read" in follower["raw_ticks"]["source"]
    # While teleoperating, the leader's reading is the command the follower is sent.
    assert leader["joint_positions"] == {"shoulder_pan.pos": 13.0, "gripper.pos": 41.5}
    assert leader["serial_number"] == "Leader"
    cameras = {camera["key"]: camera for camera in state["cameras"]}
    top = cameras["top"]
    assert top["settings"]["output_type"] == "both"
    assert top["depth"]["unit"] == "mm"
    assert top["depth"]["range_m"] == [0.3, 2.0]
    assert top["depth"]["aligned_to_color"]
    assert top["depth"]["decimation"] == 2
    # The saved depth image's own intrinsics: the aligned depth's, halved.
    assert top["depth"]["intrinsics"] == {"fx": 455.0, "ppx": 106.0, "width": 212, "height": 120}
    assert top["realsense"]["depth_scale_m"] == 0.001
    assert cameras["cam 0"]["realsense"] is None
    assert cameras["cam 0"]["depth"] is None
    assert state["warnings"] == []
    assert (state["defective"], state["comment"]) == (False, "")


def test_without_teleoperation_the_leader_is_not_read(live, tmp_path) -> None:
    live.control = None

    result = snapshot.write(snapshot.capture(live), tmp_path)

    state = json.loads((Path(result.path) / "state.json").read_text())
    assert state["robots"]["leader"]["joint_positions"] is None
    assert "bus traffic" in state["robots"]["leader"]["joint_positions_source"]
    assert not result.leader
    assert any("leader" in warning for warning in result.warnings)


def test_missing_frames_and_intrinsics_are_reported_not_fatal(live, tmp_path) -> None:
    live.store.invalidate_observation("top_depth")
    path = camera_stream_info_path(live.loaded_environment.row.cameras[1])
    assert path is not None
    path.unlink()

    result = snapshot.write(snapshot.capture(live), tmp_path)

    top = next(camera for camera in result.cameras if camera.key == "top")
    assert top.files == ["top.color.png"]
    assert any("no depth frame yet" in warning for warning in result.warnings)
    assert any("no intrinsics" in warning for warning in result.warnings)


def test_snapshots_are_listed_newest_first(live, tmp_path) -> None:
    first = snapshot.write(snapshot.capture(live), tmp_path)
    second = snapshot.write(snapshot.capture(live), tmp_path)
    (tmp_path / str(PROJECT_ID) / "not-a-snapshot").mkdir()

    listed = snapshot.list_snapshots(tmp_path / str(PROJECT_ID))

    assert [item.snapshot_id for item in listed] == sorted([first.snapshot_id, second.snapshot_id], reverse=True)
    assert listed[0].cameras == second.cameras


def test_a_session_without_environment_cannot_be_captured(live) -> None:
    live.loaded_environment = None
    with pytest.raises(ValueError, match="no environment"):
        snapshot.capture(live)


@pytest.fixture
def client(live) -> Iterator[TestClient]:
    registry = RuntimeSessionRegistry()
    registry.add(live)  # type: ignore[arg-type]
    app.dependency_overrides[get_runtime_session_registry] = lambda: registry
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_the_api_captures_the_projects_live_session_and_lists_it(client, live) -> None:
    response = client.post(f"/api/projects/{PROJECT_ID}/state_snapshots", params={"session_id": live.id})

    assert response.status_code == 201
    created = response.json()
    folder = Path(created["path"])
    assert folder.parent == get_settings().storage_dir / "state_snapshots" / str(PROJECT_ID)
    assert (folder / "state.json").is_file()
    assert "studio_version" in json.loads((folder / "state.json").read_text())["app"]
    listed = client.get(f"/api/projects/{PROJECT_ID}/state_snapshots").json()
    assert listed[0]["snapshot_id"] == created["snapshot_id"]


def test_the_api_without_a_live_session_says_so(client) -> None:
    response = client.post(f"/api/projects/{uuid4()}/state_snapshots")

    assert response.status_code == 409
    assert response.json()["error_code"] == "no_live_session"
    assert "Open the environment" in response.json()["message"]


# Labels and the CSV index.

TRICKY_COMMENT = 'scratch, near "edge"\nsecond line; naïve ✓'


def _csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"), newline="")))


def test_a_snapshot_is_saved_with_its_label_and_indexed(live, tmp_path) -> None:
    result = snapshot.write(snapshot.capture(live), tmp_path, defective=True, comment=TRICKY_COMMENT)

    assert (result.defective, result.comment) == (True, TRICKY_COMMENT)
    folder = Path(result.path)
    state = json.loads((folder / "state.json").read_text())
    assert (state["defective"], state["comment"]) == (True, TRICKY_COMMENT)
    (row,) = _csv(folder.parent / snapshot_index.CSV_NAME)
    assert row["snapshot_id"] == result.snapshot_id
    assert (row["defective"], row["comment"]) == ("true", TRICKY_COMMENT)
    assert row["captured_at_local"] == state["captured_at"]["local"]
    assert row["captured_at_utc"] == state["captured_at"]["utc"]
    assert row["environment_id"] == str(live.loaded_environment.row.id)
    assert (row["session_id"], row["control_mode"]) == ("session-1", "teleop")
    assert row["joint_names"] == "shoulder_pan.pos;gripper.pos"
    assert (row["follower_joint_1"], row["follower_joint_2"], row["follower_joint_3"]) == ("12.5", "40.0", "")
    assert (row["leader_joint_1"], row["leader_joint_2"]) == ("13.0", "41.5")
    assert row["camera_names"] == "Cam 0;Top"
    # Paths relative to the CSV that lead to the saved images.
    for column in ("camera_1_color", "camera_2_color", "camera_2_depth", "camera_2_depth_preview"):
        assert (folder.parent / row[column]).is_file(), column
    assert row["camera_1_depth"] == ""
    assert [item.comment for item in snapshot.list_snapshots(folder.parent)] == [TRICKY_COMMENT]


def test_a_snapshot_is_saved_even_if_the_index_cannot_be_written(live, tmp_path, monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(snapshot_index, "append", fail)

    result = snapshot.write(snapshot.capture(live), tmp_path)

    assert (Path(result.path) / "state.json").is_file()
    assert any("snapshots.csv" in warning for warning in result.warnings)
    assert snapshot_index.ensure(Path(result.path).parent)  # the next check adds it


@pytest.fixture
def project(live) -> UUID:
    """A project of its own for the live session, so that no other test's snapshots show up."""
    live.project_id = uuid4()
    return live.project_id


def _url(project: UUID, suffix: str = "") -> str:
    return f"/api/projects/{project}/state_snapshots{suffix}"


def _download(client: TestClient, project: UUID, **params) -> list[dict[str, str]]:
    response = client.get(_url(project, "/snapshots.csv"), params=params)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert response.headers["content-disposition"] == f'attachment; filename="state_snapshots_{project}.csv"'
    reader = csv.DictReader(io.StringIO(response.content.decode("utf-8"), newline=""))
    assert tuple(reader.fieldnames or ()) == snapshot_index.COLUMNS
    return list(reader)


def _capture(client: TestClient, live, project: UUID, **label) -> dict:
    response = client.post(_url(project), params={"session_id": live.id}, json=label or None)
    assert response.status_code == 201, response.text
    return response.json()


def test_the_api_captures_with_a_label_lists_it_and_indexes_it(client, live, project) -> None:
    created = _capture(client, live, project, defective=True, comment=TRICKY_COMMENT)

    assert (created["defective"], created["comment"]) == (True, TRICKY_COMMENT)
    state = json.loads((Path(created["path"]) / "state.json").read_text())
    assert (state["defective"], state["comment"]) == (True, TRICKY_COMMENT)
    listed = client.get(_url(project)).json()
    assert [(item["snapshot_id"], item["defective"], item["comment"]) for item in listed] == [
        (created["snapshot_id"], True, TRICKY_COMMENT)
    ]
    rows = _download(client, project)
    assert [(row["snapshot_id"], row["defective"], row["comment"]) for row in rows] == [
        (created["snapshot_id"], "true", TRICKY_COMMENT)
    ]


def test_a_capture_without_a_label_is_good_and_has_no_comment(client, live, project) -> None:
    created = _capture(client, live, project)

    assert (created["defective"], created["comment"]) == (False, "")
    (row,) = _download(client, project)
    assert (row["defective"], row["comment"]) == ("false", "")


def test_a_comment_is_trimmed_and_may_have_500_characters(client, live, project) -> None:
    created = _capture(client, live, project, comment="  " + "é" * 500 + "\n")

    assert created["comment"] == "é" * 500


@pytest.mark.parametrize(
    "label",
    [
        pytest.param({"comment": "x" * 501}, id="too long"),
        pytest.param({"comment": 5}, id="not text"),
        pytest.param({"comment": "a\x00b"}, id="NUL"),
        pytest.param({"defective": "maybe"}, id="not a bool"),
        pytest.param({"defect": True}, id="unknown field"),
    ],
)
def test_the_api_refuses_a_bad_label_and_saves_nothing(client, live, project, label) -> None:
    response = client.post(_url(project), params={"session_id": live.id}, json=label)

    assert response.status_code == 400
    assert response.json()["error_code"] == "bad_request"
    assert client.get(_url(project)).json() == []


def test_the_api_changes_a_label_later(client, live, project) -> None:
    first = _capture(client, live, project, comment="ok")
    second = _capture(client, live, project, comment="untouched")

    response = client.patch(_url(project, f"/{first['snapshot_id']}"), json={"defective": True})
    assert response.status_code == 200
    assert (response.json()["defective"], response.json()["comment"]) == (True, "ok")
    response = client.patch(_url(project, f"/{first['snapshot_id']}"), json={"comment": f"  {TRICKY_COMMENT} "})
    assert response.status_code == 200
    assert (response.json()["defective"], response.json()["comment"]) == (True, TRICKY_COMMENT)

    state = json.loads((Path(first["path"]) / "state.json").read_text())
    assert (state["defective"], state["comment"]) == (True, TRICKY_COMMENT)
    assert "label_updated_at" in state
    assert state["robots"]["follower"]["joint_positions"] == {"shoulder_pan.pos": 12.5, "gripper.pos": 40.0}
    labels = {row["snapshot_id"]: (row["defective"], row["comment"]) for row in _download(client, project)}
    assert labels == {first["snapshot_id"]: ("true", TRICKY_COMMENT), second["snapshot_id"]: ("false", "untouched")}
    listed = {item["snapshot_id"]: (item["defective"], item["comment"]) for item in client.get(_url(project)).json()}
    assert listed == {first["snapshot_id"]: (True, TRICKY_COMMENT), second["snapshot_id"]: (False, "untouched")}

    response = client.patch(_url(project, f"/{first['snapshot_id']}"), json={"defective": False, "comment": ""})
    assert (response.json()["defective"], response.json()["comment"]) == (False, "")


def test_a_label_change_needs_a_change_and_a_snapshot(client, live, project) -> None:
    created = _capture(client, live, project)

    for body in ({}, {"defective": None, "comment": None}, {"comment": "x" * 501}, {"defective": True, "x": 1}):
        response = client.patch(_url(project, f"/{created['snapshot_id']}"), json=body)
        assert response.status_code == 400, body
    response = client.patch(_url(project, "/20261114-030000-999"), json={"defective": True})
    assert response.status_code == 404
    assert response.json()["error_code"] == "state_snapshot_not_found"
    response = client.patch(_url(uuid4(), f"/{created['snapshot_id']}"), json={"defective": True})
    assert response.status_code == 404
    response = client.patch(_url(project, "/not-a-snapshot"), json={"defective": True})
    assert response.status_code == 400
    assert not client.get(_url(project)).json()[0]["defective"]


def test_the_csv_download_is_rebuilt_when_missing_or_damaged(client, live, project) -> None:
    assert _download(client, project) == []  # a project without snapshots: the header only
    first = _capture(client, live, project, defective=True)
    second = _capture(client, live, project, comment="b, c")
    index = Path(first["path"]).parent / snapshot_index.CSV_NAME

    index.unlink()
    assert [row["snapshot_id"] for row in _download(client, project)] == [first["snapshot_id"], second["snapshot_id"]]

    index.write_bytes(index.read_bytes()[:-30])
    rows = _download(client, project)
    assert [(row["defective"], row["comment"]) for row in rows] == [("true", ""), ("false", "b, c")]

    # Edited by hand: only an explicit rebuild picks that up, since the index still lists every snapshot.
    state_path = Path(second["path"]) / "state.json"
    state_path.write_text(json.dumps(json.loads(state_path.read_text()) | {"comment": "by hand"}))
    assert _download(client, project)[1]["comment"] == "b, c"
    assert _download(client, project, rebuild="true")[1]["comment"] == "by hand"


def test_snapshots_from_before_labels_are_unlabelled_until_labelled(client, live, project) -> None:
    created = _capture(client, live, project)
    state_path = Path(created["path"]) / "state.json"
    old = json.loads(state_path.read_text())
    del old["defective"], old["comment"]  # as the first snapshot feature wrote them
    state_path.write_text(json.dumps(old))
    (state_path.parent.parent / snapshot_index.CSV_NAME).unlink()

    assert client.get(_url(project)).json()[0]["defective"] is None
    assert _download(client, project)[0]["defective"] == ""

    response = client.patch(_url(project, f"/{created['snapshot_id']}"), json={"comment": "seen"})
    assert (response.json()["defective"], response.json()["comment"]) == (None, "seen")
    assert _download(client, project)[0]["defective"] == ""
    response = client.patch(_url(project, f"/{created['snapshot_id']}"), json={"defective": False})
    assert response.json()["defective"] is False
    assert _download(client, project)[0]["defective"] == "false"


def test_the_list_can_be_limited_to_the_newest(client, live, project) -> None:
    created = [_capture(client, live, project, comment=str(index))["snapshot_id"] for index in range(3)]

    listed = client.get(_url(project), params={"limit": 2}).json()

    assert [item["snapshot_id"] for item in listed] == created[:0:-1]
    assert client.get(_url(project), params={"limit": 0}).status_code == 400
