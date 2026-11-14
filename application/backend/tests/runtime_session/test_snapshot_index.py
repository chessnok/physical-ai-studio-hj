"""``snapshots.csv``: one well-formed row per snapshot, whatever the comments hold and however captures overlap."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from runtime_session import snapshot_index
from runtime_session.snapshot_index import COLUMNS, CSV_NAME

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
TRICKY_COMMENT = 'scratch, near "edge"\nsecond line; naïve ✓ 日本'


def _document(snapshot_id: str, **changes: Any) -> dict[str, Any]:
    """A ``state.json`` as a capture writes it: a follower, a leader while teleoperating, two cameras."""
    document: dict[str, Any] = {
        "format": "physicalai-studio/state-snapshot/v1",
        "snapshot_id": snapshot_id,
        "captured_at": {"local": "2026-11-14T03:00:00.123+02:00", "utc": "2026-11-14T01:00:00.123+00:00"},
        "defective": False,
        "comment": "",
        "session": {
            "id": "session-1",
            "control": {"type": "teleop"},
            "dataset": {"dataset_id": "dataset-1", "is_recording": False},
        },
        "environment": {"id": "environment-1", "name": "Bench, left"},
        "robots": {
            "follower": {"joint_positions": dict(zip(JOINTS, [1.5, -20.25, 30.0, 0.0, -99.5, 42.0], strict=True))},
            "leader": {"joint_positions": dict(zip(JOINTS, [2.0, -21.0, 31.0, 0.5, -98.0, 40.0], strict=True))},
        },
        "cameras": [
            {
                "name": "Top",
                "files": {"color": "top.color.png", "depth": "top.depth.png", "depth_preview": "top.depth_preview.png"},
            },
            {"name": "Wrist cam", "files": {"color": "wrist_cam.color.png", "depth": None, "depth_preview": None}},
        ],
        "warnings": [],
    }
    return document | changes


def _save(project_root: Path, snapshot_id: str, **changes: Any) -> dict[str, Any]:
    """A snapshot folder with its ``state.json``, as a capture leaves it before indexing."""
    document = _document(snapshot_id, **changes)
    (project_root / snapshot_id).mkdir(parents=True)
    snapshot_index.write_json(project_root / snapshot_id / "state.json", document)
    return document


def _rows(project_root: Path) -> list[dict[str, str]]:
    data = (project_root / CSV_NAME).read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    assert tuple(reader.fieldnames or ()) == COLUMNS
    return list(reader)


def _snapshot_id(index: int) -> str:
    return f"20261114-0300{index // 1000 % 60:02d}-{index % 1000:03d}"


def test_the_header_is_stable() -> None:
    # Readers pick columns by name; this is the documented order. Only ever append new columns.
    assert COLUMNS[:11] == (
        "snapshot_id",
        "captured_at_local",
        "captured_at_utc",
        "defective",
        "comment",
        "environment_id",
        "environment_name",
        "session_id",
        "control_mode",
        "dataset_id",
        "joint_names",
    )
    assert COLUMNS[11:17] == tuple(f"follower_joint_{n}" for n in range(1, 7))
    assert COLUMNS[17:23] == tuple(f"leader_joint_{n}" for n in range(1, 7))
    assert COLUMNS[23] == "camera_names"
    assert COLUMNS[24:28] == ("camera_1_name", "camera_1_color", "camera_1_depth", "camera_1_depth_preview")
    assert COLUMNS[-1] == "warnings"
    assert len(COLUMNS) == 41
    # Each column is documented in the module docstring (numbered ones by their pattern).
    documented = snapshot_index.__doc__ or ""
    assert all(f"``{column}``" in documented for column in COLUMNS if not any(c.isdigit() for c in column))
    assert all(name in documented for name in ("follower_joint_1", "leader_joint_6", "camera_<n>_depth_preview"))


def test_a_row_holds_the_label_joints_and_image_paths(tmp_path) -> None:
    document = _save(tmp_path, "20261114-030000-123", defective=True, comment="dent")
    snapshot_index.append(tmp_path, document)

    (row,) = _rows(tmp_path)
    assert row["snapshot_id"] == "20261114-030000-123"
    assert row["captured_at_local"] == "2026-11-14T03:00:00.123+02:00"
    assert row["captured_at_utc"] == "2026-11-14T01:00:00.123+00:00"
    assert row["defective"] == "true"
    assert row["comment"] == "dent"
    assert (row["environment_id"], row["environment_name"]) == ("environment-1", "Bench, left")
    assert (row["session_id"], row["control_mode"], row["dataset_id"]) == ("session-1", "teleop", "dataset-1")
    assert row["joint_names"] == ";".join(JOINTS)
    assert [float(row[f"follower_joint_{n}"]) for n in range(1, 7)] == [1.5, -20.25, 30.0, 0.0, -99.5, 42.0]
    assert [float(row[f"leader_joint_{n}"]) for n in range(1, 7)] == [2.0, -21.0, 31.0, 0.5, -98.0, 40.0]
    assert row["camera_names"] == "Top;Wrist cam"
    # Relative to the CSV, so they resolve wherever the folder is copied.
    assert row["camera_1_color"] == "20261114-030000-123/top.color.png"
    assert row["camera_1_depth"] == "20261114-030000-123/top.depth.png"
    assert row["camera_1_depth_preview"] == "20261114-030000-123/top.depth_preview.png"
    assert (row["camera_2_name"], row["camera_2_color"]) == ("Wrist cam", "20261114-030000-123/wrist_cam.color.png")
    assert row["camera_2_depth"] == row["camera_2_depth_preview"] == ""
    assert row["camera_3_name"] == row["camera_3_color"] == ""


def test_an_unknown_leader_and_missing_parts_are_empty_cells(tmp_path) -> None:
    document = _save(
        tmp_path,
        "20261114-030000-123",
        robots={"follower": {"joint_positions": {"gripper.pos": 3.0}}, "leader": {"joint_positions": None}},
        session={"id": "session-1", "control": None, "dataset": None},
        cameras=[],
        warnings=["Camera 'Top' has no frame yet", "The leader's joints are known only while teleoperation runs"],
    )
    snapshot_index.append(tmp_path, document)

    (row,) = _rows(tmp_path)
    assert row["follower_joint_1"] == "3.0"
    assert all(row[f"follower_joint_{n}"] == "" for n in range(2, 7))
    assert all(row[f"leader_joint_{n}"] == "" for n in range(1, 7))
    assert row["control_mode"] == row["dataset_id"] == row["camera_names"] == ""
    assert row["warnings"] == " | ".join(document["warnings"])


def test_comments_with_commas_quotes_newlines_and_non_ascii_survive(tmp_path) -> None:
    first = _save(tmp_path, "20261114-030000-001", comment=TRICKY_COMMENT, defective=True)
    second = _save(tmp_path, "20261114-030000-002", comment='"', environment={"id": "e", "name": "a,b"})
    snapshot_index.append(tmp_path, first)
    snapshot_index.append(tmp_path, second)

    rows = _rows(tmp_path)
    assert [(row["snapshot_id"], row["comment"], row["defective"]) for row in rows] == [
        ("20261114-030000-001", TRICKY_COMMENT, "true"),
        ("20261114-030000-002", '"', "false"),
    ]
    assert rows[1]["environment_name"] == "a,b"
    raw = (tmp_path / CSV_NAME).read_bytes()
    assert "naïve ✓ 日本".encode() in raw  # UTF-8, not escaped
    assert not raw.startswith(b"\xef\xbb\xbf")  # no BOM


def test_appending_twice_adds_one_row(tmp_path) -> None:
    document = _save(tmp_path, "20261114-030000-123")
    snapshot_index.append(tmp_path, document)
    snapshot_index.append(tmp_path, document)

    assert len(_rows(tmp_path)) == 1


def test_a_missing_index_is_rebuilt_from_every_snapshot(tmp_path) -> None:
    # Snapshots from before labels existed have no defective or comment.
    old = _document("20261113-170000-000")
    del old["defective"], old["comment"]
    (tmp_path / old["snapshot_id"]).mkdir()
    (tmp_path / old["snapshot_id"] / "state.json").write_text(json.dumps(old))
    (tmp_path / "20261113-170000-500").mkdir()  # a capture still writing its images: no state.json yet
    new = _save(tmp_path, "20261114-030000-123", defective=True)

    snapshot_index.append(tmp_path, new)

    rows = _rows(tmp_path)
    assert [(row["snapshot_id"], row["defective"], row["comment"]) for row in rows] == [
        ("20261113-170000-000", "", ""),  # not labelled, which is not the same as good
        ("20261114-030000-123", "true", ""),
    ]


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(lambda data: b"", id="empty"),
        pytest.param(lambda data: data.replace(b"snapshot_id", b"id", 1), id="other header"),
        pytest.param(lambda data: data + b"20261114-030000-999,true\n", id="short row"),
        pytest.param(lambda data: data[:-1], id="no line end after the last row"),
        pytest.param(lambda data: data[: len(data) // 2], id="cut off"),
        pytest.param(lambda data: data + b"\xff\xfe\n", id="not UTF-8"),
        pytest.param(lambda data: data + data.split(b"\n", 1)[1], id="a row twice"),
    ],
)
def test_an_unreadable_index_is_rebuilt(tmp_path, damage) -> None:
    for index in range(3):
        snapshot_index.append(tmp_path, _save(tmp_path, _snapshot_id(index), comment=f"a, b\n{index}"))
    path = tmp_path / CSV_NAME
    path.write_bytes(damage(path.read_bytes()))

    assert snapshot_index.ensure(tmp_path)

    assert [(row["snapshot_id"], row["comment"]) for row in _rows(tmp_path)] == [
        (_snapshot_id(index), f"a, b\n{index}") for index in range(3)
    ]
    assert not snapshot_index.ensure(tmp_path)  # now current


def test_an_index_out_of_step_with_the_folders_is_rebuilt(tmp_path) -> None:
    for index in range(3):
        snapshot_index.append(tmp_path, _save(tmp_path, _snapshot_id(index)))
    shutil.rmtree(tmp_path / _snapshot_id(1))  # deleted by hand
    _save(tmp_path, _snapshot_id(7))  # saved, but its append never happened

    assert b"\n" in snapshot_index.contents(tmp_path)

    assert [row["snapshot_id"] for row in _rows(tmp_path)] == [_snapshot_id(0), _snapshot_id(2), _snapshot_id(7)]


def test_a_state_json_of_an_unexpected_shape_still_gets_its_row(tmp_path) -> None:
    good = _save(tmp_path, _snapshot_id(1))
    _save(tmp_path, _snapshot_id(2), robots={"follower": "not a dict"}, cameras="nope", defective=True, comment="dent")
    (tmp_path / _snapshot_id(3)).mkdir()
    (tmp_path / _snapshot_id(3) / "state.json").write_text("{ not json")
    _save(tmp_path, _snapshot_id(4), robots={"follower": {"joint_positions": {"gripper.pos": "high"}}})

    snapshot_index.append(tmp_path, good)

    rows = {row["snapshot_id"]: row for row in _rows(tmp_path)}
    assert list(rows) == [_snapshot_id(1), _snapshot_id(2), _snapshot_id(4)]  # unreadable JSON is left out
    assert "unexpected shape" in rows[_snapshot_id(2)]["warnings"]
    # Its label survives where it is readable.
    assert (rows[_snapshot_id(2)]["defective"], rows[_snapshot_id(2)]["comment"]) == ("true", "dent")
    assert rows[_snapshot_id(4)]["follower_joint_1"] == ""


def test_a_placeholder_row_leaves_an_unreadable_label_empty(tmp_path) -> None:
    _save(tmp_path, _snapshot_id(1), robots={"follower": "not a dict"}, defective="yes", comment=7)

    snapshot_index.rebuild(tmp_path)

    (row,) = _rows(tmp_path)
    assert (row["defective"], row["comment"]) == ("", "")
    assert "unexpected shape" in row["warnings"]


@pytest.mark.parametrize(
    ("label", "cell"),
    [
        pytest.param({"defective": True}, "true", id="defective"),
        pytest.param({"defective": False}, "false", id="good"),
        pytest.param({}, "", id="never labelled"),
        pytest.param({"defective": None}, "", id="null"),
        pytest.param({"defective": "yes"}, "", id="not a bool"),
    ],
)
def test_an_unlabelled_snapshot_is_neither_good_nor_defective(tmp_path, label, cell) -> None:
    document = _document(_snapshot_id(1)) | label
    if "defective" not in label:
        del document["defective"]
    (tmp_path / _snapshot_id(1)).mkdir()
    snapshot_index.write_json(tmp_path / _snapshot_id(1) / "state.json", document)

    snapshot_index.append(tmp_path, document)

    assert _rows(tmp_path)[0]["defective"] == cell


def test_contents_of_a_project_without_snapshots_is_the_header(tmp_path) -> None:
    data = snapshot_index.contents(tmp_path / "no-such-project")

    assert data.decode("utf-8") == ",".join(COLUMNS) + "\n"
    assert not (tmp_path / "no-such-project").exists()


def test_updating_a_label_rewrites_state_json_and_the_index(tmp_path) -> None:
    for index in range(3):
        snapshot_index.append(tmp_path, _save(tmp_path, _snapshot_id(index)))

    document = snapshot_index.update(tmp_path, _snapshot_id(1), {"defective": True, "comment": TRICKY_COMMENT})

    assert document["defective"] is True
    state = json.loads((tmp_path / _snapshot_id(1) / "state.json").read_text())
    assert (state["defective"], state["comment"]) == (True, TRICKY_COMMENT)
    assert state["robots"] == _document("x")["robots"]  # the rest is kept
    rows = _rows(tmp_path)
    assert [(row["defective"], row["comment"]) for row in rows] == [
        ("false", ""),
        ("true", TRICKY_COMMENT),
        ("false", ""),
    ]


def test_a_value_differing_from_state_json_is_rebuilt(tmp_path) -> None:
    for index in range(2):
        snapshot_index.append(tmp_path, _save(tmp_path, _snapshot_id(index), comment="before"))
    state = tmp_path / _snapshot_id(1) / "state.json"
    snapshot_index.write_json(state, json.loads(state.read_text()) | {"defective": True, "comment": "by hand"})

    assert snapshot_index.ensure(tmp_path)

    assert [(row["defective"], row["comment"]) for row in _rows(tmp_path)] == [("false", "before"), ("true", "by hand")]
    assert not snapshot_index.ensure(tmp_path)


def _disk_full(*_args, **_kwargs):
    raise OSError(28, "No space left on device")


def test_an_edit_cut_short_after_state_json_is_repaired_by_the_next_check(tmp_path, monkeypatch) -> None:
    for index in range(2):
        snapshot_index.append(tmp_path, _save(tmp_path, _snapshot_id(index)))
    write = snapshot_index._write_locked
    monkeypatch.setattr(snapshot_index, "_write_locked", _disk_full)

    # The label is changed (state.json is the truth); the index cannot be rewritten yet.
    document = snapshot_index.update(tmp_path, _snapshot_id(1), {"defective": True, "comment": "dent"})

    assert (document["defective"], document["comment"]) == (True, "dent")
    state = json.loads((tmp_path / _snapshot_id(1) / "state.json").read_text())
    assert (state["defective"], state["comment"]) == (True, "dent")
    assert _rows(tmp_path)[1]["defective"] == "false"  # stale, and marked so
    assert (tmp_path / snapshot_index.DIRTY_NAME).exists()

    monkeypatch.setattr(snapshot_index, "_write_locked", write)
    downloaded = snapshot_index.contents(tmp_path)  # what the CSV download serves

    rows = list(csv.DictReader(io.StringIO(downloaded.decode("utf-8"), newline="")))
    assert [(row["defective"], row["comment"]) for row in rows] == [("false", ""), ("true", "dent")]
    assert not (tmp_path / snapshot_index.DIRTY_NAME).exists()


def test_a_capture_after_an_edit_cut_short_repairs_the_index(tmp_path, monkeypatch) -> None:
    snapshot_index.append(tmp_path, _save(tmp_path, _snapshot_id(0)))
    write = snapshot_index._write_locked
    monkeypatch.setattr(snapshot_index, "_write_locked", _disk_full)
    snapshot_index.update(tmp_path, _snapshot_id(0), {"comment": "dent"})
    monkeypatch.setattr(snapshot_index, "_write_locked", write)

    snapshot_index.append(tmp_path, _save(tmp_path, _snapshot_id(1)))  # the next capture

    assert [(row["snapshot_id"], row["comment"]) for row in _rows(tmp_path)] == [
        (_snapshot_id(0), "dent"),
        (_snapshot_id(1), ""),
    ]
    assert not (tmp_path / snapshot_index.DIRTY_NAME).exists()


def test_an_edit_whose_state_json_cannot_be_written_changes_nothing(tmp_path, monkeypatch) -> None:
    snapshot_index.append(tmp_path, _save(tmp_path, _snapshot_id(0), comment="before"))

    monkeypatch.setattr(snapshot_index, "write_json", _disk_full)
    with pytest.raises(OSError, match="No space"):
        snapshot_index.update(tmp_path, _snapshot_id(0), {"comment": "after"})
    monkeypatch.undo()

    assert json.loads((tmp_path / _snapshot_id(0) / "state.json").read_text())["comment"] == "before"
    snapshot_index.ensure(tmp_path)
    assert _rows(tmp_path)[0]["comment"] == "before"
    assert not (tmp_path / snapshot_index.DIRTY_NAME).exists()


@pytest.mark.parametrize("snapshot_id", ["20261114-030000-999", "..", "../other-project/20261114-030000-000"])
def test_updating_a_missing_snapshot_fails(tmp_path, snapshot_id) -> None:
    project = tmp_path / "project"
    _save(project, _snapshot_id(0))
    _save(tmp_path / "other-project", "20261114-030000-000")

    with pytest.raises(FileNotFoundError):
        snapshot_index.update(project, snapshot_id, {"defective": True})


def test_concurrent_captures_edits_and_rebuilds_leave_one_row_each(tmp_path) -> None:
    """Threads, like the API's worker threads: captures append while labels are edited and the index rebuilt."""
    count = 120
    ids = [_snapshot_id(index) for index in range(count)]
    seed = _save(tmp_path, "20261113-000000-000")
    snapshot_index.append(tmp_path, seed)
    stop = threading.Event()

    def capture(snapshot_id: str) -> None:
        snapshot_index.append(tmp_path, _save(tmp_path, snapshot_id, comment=f'{snapshot_id}, "q"\nline 2'))

    def edit() -> None:
        flips = 0
        while not stop.is_set():
            flips += 1
            snapshot_index.update(tmp_path, seed["snapshot_id"], {"defective": flips % 2 == 1, "comment": str(flips)})
            snapshot_index.ensure(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        editor = pool.submit(edit)
        list(pool.map(capture, ids))
        stop.set()
        editor.result()

    rows = _rows(tmp_path)
    assert sorted(row["snapshot_id"] for row in rows) == ["20261113-000000-000", *ids]
    assert all(row["comment"] == f'{row["snapshot_id"]}, "q"\nline 2' for row in rows if row["snapshot_id"] in ids)
    assert not snapshot_index.ensure(tmp_path)


_PROCESS_SCRIPT = """
import sys
from pathlib import Path
from runtime_session import snapshot_index

root, worker, count = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
for index in range(count):
    snapshot_id = f"20261114-03{worker:02d}{index // 1000:02d}-{index % 1000:03d}"
    (root / snapshot_id).mkdir()
    document = {"snapshot_id": snapshot_id, "comment": f"worker {worker}, #{index}\\n\\"ok\\"", "defective": True}
    snapshot_index.write_json(root / snapshot_id / "state.json", document)
    snapshot_index.append(root, document)
    if index % 10 == 0:
        snapshot_index.rebuild(root)
"""


def test_captures_in_several_processes_leave_one_row_each(tmp_path) -> None:
    """Processes, like several backend workers: the lock file serializes them."""
    workers, count = 3, 40
    src = Path(snapshot_index.__file__).resolve().parents[1]
    env = os.environ | {"PYTHONPATH": os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")])}
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _PROCESS_SCRIPT, str(tmp_path), str(worker), str(count)],
            env=env,
            stderr=subprocess.PIPE,
        )
        for worker in range(workers)
    ]
    for process in processes:
        _, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, stderr.decode()

    rows = _rows(tmp_path)
    assert len(rows) == workers * count
    assert len({row["snapshot_id"] for row in rows}) == workers * count
    assert all(row["comment"].endswith('\n"ok"') and row["defective"] == "true" for row in rows)
    assert not snapshot_index.ensure(tmp_path)
