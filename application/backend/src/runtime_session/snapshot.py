"""State snapshots: what a live session holds right now, saved for people to look at.

A snapshot is read-only. It copies what the session's store already holds, the
latest frame of every camera and the latest follower state and action, the way
a recording does, and writes it to ``<storage>/state_snapshots/<project>/<time>/``:

- ``state.json``: when, which project, environment, robots and cameras; joint
  positions with the time each was measured; camera settings, and a RealSense's
  intrinsics and depth scale as its publisher recorded them; its label
  (``defective``, ``comment``), which can be changed later.
- ``<camera>.color.png``: the color image, lossless.
- ``<camera>.depth.png`` (16-bit, millimetres, 0 = no depth) and
  ``<camera>.depth_preview.png`` (colorized) for a camera that records depth.

It never opens a device, sends a robot anything or reads a robot's bus: the
leader's positions are known only while it teleoperates (they are the command
the follower is sent), and servo ticks are derived from the calibration.

Every snapshot also gets a row in the project's ``snapshots.csv``
(:mod:`runtime_session.snapshot_index`).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from loguru import logger

from runtime_session import snapshot_index
from runtime_session.snapshot_index import STATE_FILE
from runtime_session.store import ACTION, STATE, depth_key
from schemas.state_snapshot import SnapshotCamera, StateSnapshot
from utils.depth_colormap import colorize_depth

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from runtime_session.session import LoadedEnvironment, RuntimeSession
    from runtime_session.store import Sample
    from schemas.project_camera import Camera

FORMAT = "physicalai-studio/state-snapshot/v1"


@dataclass
class _Clock:
    """Turns the store's ``time.monotonic()`` stamps into wall-clock times."""

    wall: float = field(default_factory=time.time)
    monotonic: float = field(default_factory=time.monotonic)

    def describe(self, sample: Sample | None) -> dict[str, Any] | None:
        if sample is None:
            return None
        age = self.monotonic - sample.timestamp
        return {
            "monotonic": sample.timestamp,
            "utc": datetime.fromtimestamp(self.wall - age, tz=UTC).isoformat(),
            "age_ms": round(age * 1000, 1),
        }


@dataclass
class Captured:
    """Everything a snapshot holds, copied from the session; writing it touches nothing live."""

    clock: _Clock
    project_id: str
    project_name: str
    session_id: str
    loaded: LoadedEnvironment
    control: dict[str, Any] | None
    dataset: dict[str, Any] | None
    state: Sample | None
    action: Sample | None
    frames: dict[str, Sample | None]


def capture(session: RuntimeSession) -> Captured:
    """Copy a live session's latest state and frames.

    Raises:
        ValueError: The session has no environment loaded, so there is nothing to capture.
    """
    loaded, store = session.loaded_environment, session.store
    if loaded is None or store is None:
        raise ValueError("The session has no environment loaded")
    clock = _Clock()
    state = session.state()
    return Captured(
        clock=clock,
        project_id=str(session.project_id),
        project_name=session.project_name,
        session_id=session.id,
        loaded=loaded,
        control=state.control.model_dump(mode="json") if state.control is not None else None,
        dataset=state.dataset.model_dump(mode="json") if state.dataset is not None else None,
        state=store.read(STATE),
        action=store.read(ACTION),
        frames={key: store.read_observation(key) for key in store.spec.observations},
    )


def write(
    captured: Captured,
    root: Path,
    *,
    build: Mapping[str, Any] | None = None,
    defective: bool = False,
    comment: str = "",
) -> StateSnapshot:
    """Write a captured snapshot under ``root/<project>/<local time>/``, index it, and describe it."""
    captured_at = datetime.fromtimestamp(captured.clock.wall).astimezone()
    folder = _new_folder(root / captured.project_id, captured_at)
    warnings: list[str] = []
    description = captured.loaded.environment.description
    cameras: list[dict[str, Any]] = []
    shown: list[SnapshotCamera] = []
    used_stems: set[str] = set()
    for key, row in description.cameras.items():
        camera, files = _write_camera(folder, key, row, captured, used_stems, warnings)
        cameras.append(camera)
        shown.append(
            SnapshotCamera(
                name=row.name,
                key=key,
                files=files,
                color=camera["color"] is not None,
                depth=camera["depth"] is not None,
            )
        )
    follower, leader = _robots(captured, warnings)
    environment_row = captured.loaded.row
    document = {
        "format": FORMAT,
        "snapshot_id": folder.name,
        "captured_at": {
            "local": captured_at.isoformat(timespec="milliseconds"),
            "utc": captured_at.astimezone(UTC).isoformat(timespec="milliseconds"),
            "monotonic": captured.clock.monotonic,
        },
        "defective": defective,
        "comment": comment,
        "app": dict(build or {}),
        "project": {"id": captured.project_id, "name": captured.project_name},
        "session": {"id": captured.session_id, "control": captured.control, "dataset": captured.dataset},
        "environment": {
            "id": str(environment_row.id) if environment_row is not None else None,
            "name": description.name,
        },
        "robots": {"follower": follower, "leader": leader},
        "action": _joints(description.spec.joint_names, captured.action, captured.clock)
        | {"meaning": "the last command sent to the follower; its time is when it is due, not when it was measured"},
        "cameras": cameras,
        "warnings": warnings,
    }
    snapshot_index.write_json(folder / STATE_FILE, document)
    logger.info("Saved state snapshot {}", folder)
    shown_warnings = list(warnings)
    try:
        snapshot_index.append(folder.parent, document)
    except OSError as error:  # the snapshot is saved; the index is rebuilt when it is next found stale
        logger.exception("Could not add snapshot {} to {}", folder.name, snapshot_index.CSV_NAME)
        shown_warnings.append(f"It is not in {snapshot_index.CSV_NAME} yet ({error}); downloading the CSV adds it.")
    return StateSnapshot(
        snapshot_id=folder.name,
        path=str(folder),
        captured_at=captured_at,
        session_id=captured.session_id,
        environment_name=description.name,
        cameras=shown,
        follower=follower["joint_positions"] is not None,
        leader=leader is not None and leader["joint_positions"] is not None,
        warnings=shown_warnings,
        defective=defective,
        comment=comment,
    )


def list_snapshots(project_root: Path, limit: int | None = None) -> list[StateSnapshot]:
    """The snapshots saved for a project, newest first, at most ``limit``.

    Folders without a readable ``state.json`` are skipped.
    """
    snapshots: list[StateSnapshot] = []
    if not project_root.is_dir():
        return snapshots
    for folder in sorted((path for path in project_root.iterdir() if path.is_dir()), reverse=True):
        if limit is not None and len(snapshots) >= limit:
            break
        try:
            document = json.loads((folder / STATE_FILE).read_text(encoding="utf-8"))
            snapshots.append(_summary(folder, document))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return snapshots


def update_label(
    project_root: Path, snapshot_id: str, *, defective: bool | None = None, comment: str | None = None
) -> StateSnapshot:
    """Change a saved snapshot's label in its ``state.json`` and in ``snapshots.csv``.

    Raises:
        FileNotFoundError: The project has no such snapshot.
    """
    changes: dict[str, Any] = {"label_updated_at": datetime.now(UTC).isoformat(timespec="milliseconds")}
    if defective is not None:
        changes["defective"] = defective
    if comment is not None:
        changes["comment"] = comment
    document, index_error = snapshot_index.update(project_root, snapshot_id, changes)
    summary = _summary(project_root / snapshot_id, document)
    if index_error is not None:
        summary.csv_warning = (
            f"The label is saved, but {snapshot_index.CSV_NAME} could not be rewritten ({index_error}); "
            "the next capture or CSV download brings it up to date."
        )
    return summary


def _summary(folder: Path, document: dict[str, Any]) -> StateSnapshot:
    # Snapshots from before labels existed have none: neither good nor defective, but unlabelled.
    defective, comment = document.get("defective"), document.get("comment")
    robots = document.get("robots") or {}
    leader = robots.get("leader") or {}
    cameras = [
        SnapshotCamera(
            name=camera["name"],
            key=camera["key"],
            files=[name for name in (camera.get("files") or {}).values() if name],
            color=camera.get("color") is not None,
            depth=camera.get("depth") is not None,
        )
        for camera in document.get("cameras", [])
    ]
    return StateSnapshot(
        snapshot_id=folder.name,
        path=str(folder),
        captured_at=datetime.fromisoformat(document["captured_at"]["local"]),
        session_id=(document.get("session") or {}).get("id"),
        environment_name=(document.get("environment") or {}).get("name"),
        cameras=cameras,
        follower=(robots.get("follower") or {}).get("joint_positions") is not None,
        leader=leader.get("joint_positions") is not None,
        warnings=document.get("warnings", []),
        defective=defective if isinstance(defective, bool) else None,
        comment=comment if isinstance(comment, str) else "",
    )


def _new_folder(project_root: Path, captured_at: datetime) -> Path:
    name = captured_at.strftime("%Y%m%d-%H%M%S-") + f"{captured_at.microsecond // 1000:03d}"
    project_root.mkdir(parents=True, exist_ok=True)
    for attempt in range(100):
        folder = project_root / (name if attempt == 0 else f"{name}-{attempt}")
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        return folder
    raise FileExistsError(f"Too many snapshots at {name}")


def _joints(names: tuple[str, ...], sample: Sample | None, clock: _Clock) -> dict[str, Any]:
    return {
        "joint_positions": None if sample is None else dict(zip(names, map(float, sample.value), strict=True)),
        "sample": clock.describe(sample),
    }


def _robots(captured: Captured, warnings: list[str]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    description = captured.loaded.environment.description
    row = captured.loaded.row
    follower_row = row.robots[0].robot if row is not None and row.robots else None
    leader_row = getattr(row.robots[0].tele_operator, "robot", None) if row is not None and row.robots else None
    names = description.spec.joint_names

    follower = _robot(follower_row, description.follower.robot_id)
    follower |= _joints(names, captured.state, captured.clock)
    follower["joint_positions_source"] = "the follower's last measurement in the session"
    if captured.state is None:
        warnings.append("The follower has not reported its joints yet")
    follower["raw_ticks"] = _derived_ticks(description.follower.driver, captured.state, warnings, "follower")

    if description.leader is None:
        return follower, None
    leader = _robot(leader_row, description.leader.robot_id)
    teleop = (captured.control or {}).get("type") == "teleop"
    if teleop and captured.action is not None:
        leader |= _joints(names, captured.action, captured.clock)
        leader["joint_positions_source"] = (
            "the leader's last reading, as teleoperation sent it to the follower; its time is when it is due"
        )
        leader["raw_ticks"] = _derived_ticks(description.leader.driver, captured.action, warnings, "leader")
    else:
        leader |= {"joint_positions": None, "sample": None, "raw_ticks": None}
        leader["joint_positions_source"] = (
            "unknown: the leader is only read while it teleoperates, and reading it now would add bus traffic"
        )
        warnings.append("The leader's joints are known only while teleoperation runs")
    return follower, leader


def _robot(row: Any, robot_id: Any) -> dict[str, Any]:
    payload: Any = getattr(row, "payload", None)
    payload_data = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else {}
    return {
        "id": str(robot_id),
        "name": getattr(row, "name", None),
        "type": getattr(row, "type", None),
        "serial_number": payload_data.get("serial_number"),
        "connection_string": payload_data.get("connection_string"),
        "unit": payload_data.get("unit"),
        # A copy of the calibration the session runs with, as the robot row holds it.
        "calibration": payload_data.get("calibration"),
    }


def _derived_ticks(
    driver_config: Mapping[str, Any], sample: Sample | None, warnings: list[str], role: str
) -> dict[str, Any] | None:
    """Servo ticks for ``sample``, converted with the driver's calibration; never read from the servos."""
    if sample is None:
        return None
    try:
        from physicalai.config import instantiate

        driver: Any = instantiate(dict(driver_config))  # builds it; only connect() touches the device
        to_ticks = getattr(driver, "_unit_to_ticks", None)
        if to_ticks is None:
            warnings.append(f"The {role}'s driver cannot convert positions to servo ticks")
            return None
        ticks = np.asarray(to_ticks(np.asarray(sample.value, dtype=np.float64)))
    except Exception as error:  # optional: the snapshot is worth more than its ticks
        warnings.append(f"The {role}'s servo ticks could not be derived: {error}")
        return None
    return {
        "values": dict(zip(sample.names, (int(tick) for tick in ticks), strict=True)),
        "source": "derived from the positions with the calibration; the servos were not read, to keep their bus free",
    }


def _write_camera(
    folder: Path,
    key: str,
    row: Camera,
    captured: Captured,
    used_stems: set[str],
    warnings: list[str],
) -> tuple[dict[str, Any], list[str]]:
    stem = key.replace(" ", "_")
    while stem in used_stems:
        stem += "_"
    used_stems.add(stem)
    payload = row.payload.model_dump(mode="json")
    camera: dict[str, Any] = {
        "name": row.name,
        "key": key,
        "id": str(row.id),
        "driver": row.driver,
        "hardware_name": row.hardware_name,
        "fingerprint": row.fingerprint,
        "settings": payload,
        "files": {"color": None, "depth": None, "depth_preview": None},
        "color": None,
        "depth": None,
    }
    files: list[str] = []
    color_frame = captured.frames.get(key)
    if key in captured.frames:
        if color_frame is None:
            warnings.append(f"Camera {row.name!r} has no frame yet")
        else:
            name = f"{stem}.color.png"
            _write_png(folder / name, cv2.cvtColor(np.ascontiguousarray(color_frame.value), cv2.COLOR_RGB2BGR))
            camera["files"]["color"] = name
            camera["color"] = {"shape": list(color_frame.value.shape), "sample": captured.clock.describe(color_frame)}
            files.append(name)
    depth_slot = depth_key(key)
    if depth_slot in captured.frames:
        depth_frame = captured.frames[depth_slot]
        if depth_frame is None:
            warnings.append(f"Camera {row.name!r} has no depth frame yet")
        else:
            depth = np.ascontiguousarray(depth_frame.value.reshape(depth_frame.value.shape[:2]))
            near, far = payload.get("depth_range_min"), payload.get("depth_range_max")
            preview = colorize_depth(depth, *((near * 1000, far * 1000) if near and far else (None, None)))
            names = (f"{stem}.depth.png", f"{stem}.depth_preview.png")
            _write_png(folder / names[0], depth)
            _write_png(folder / names[1], cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            camera["files"]["depth"], camera["files"]["depth_preview"] = names
            # Recorded decimated (DEPTH_RECORD_DOWNSCALE): depth pixel (u, v) is color pixel (s*u, s*v).
            scale = max(1, round(row.payload.model_dump().get("height", depth.shape[0]) / depth.shape[0]))
            camera["depth"] = {
                "shape": list(depth.shape),
                "dtype": "uint16",
                "unit": "mm",
                "zero_means": "no depth (not measured, or outside the camera's depth range)",
                "range_m": [near, far],
                "aligned_to_color": key in captured.frames or row.payload.model_dump().get("output_type") == "both",
                "decimation": scale,
                "pixel_to_color": f"depth pixel (u, v) is color pixel ({scale}*u, {scale}*v)",
                "valid_fraction": round(float(np.count_nonzero(depth)) / depth.size, 4),
                "sample": captured.clock.describe(depth_frame),
            }
            files.extend(names)
    camera["realsense"] = _realsense_info(row, warnings)
    if camera["depth"] is not None and camera["realsense"] is not None:
        camera["depth"]["intrinsics"] = _recorded_depth_intrinsics(camera["realsense"], camera["depth"]["decimation"])
    return camera, files


def _recorded_depth_intrinsics(info: dict[str, Any], scale: int) -> dict[str, Any] | None:
    """The intrinsics of the depth image as saved: the stream's depth intrinsics, decimated by ``scale``."""
    intrinsics = info.get("depth_intrinsics")
    if not isinstance(intrinsics, dict):
        return None
    scaled = dict(intrinsics)
    for name in ("fx", "fy", "ppx", "ppy"):
        if isinstance(scaled.get(name), int | float):
            scaled[name] = scaled[name] / scale
    for name in ("width", "height"):
        if isinstance(scaled.get(name), int):
            scaled[name] = -(-scaled[name] // scale)
    return scaled


def _realsense_info(row: Camera, warnings: list[str]) -> dict[str, Any] | None:
    """The intrinsics and depth scale the camera's publisher recorded when it connected."""
    if row.driver != "realsense":
        return None
    from utils.camera_factory import camera_stream_info_path
    from utils.realsense_rgbd import read_stream_info

    path = camera_stream_info_path(row)
    info = read_stream_info(path) if path is not None else None
    if info is None:
        warnings.append(f"Camera {row.name!r}: no intrinsics recorded by its publisher ({path})")
    return info


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Could not write {path}")
