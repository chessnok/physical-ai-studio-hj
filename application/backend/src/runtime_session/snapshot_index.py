"""``snapshots.csv``: one row per state snapshot of a project, with its label, to sort and filter them.

Kept next to the snapshot folders, at ``<storage>/state_snapshots/<project>/snapshots.csv``: UTF-8
(no BOM), comma-separated, a header row, ``\\n`` line ends, written with :mod:`csv` (a comment with
commas, quotes or line breaks is quoted, so read it with a CSV reader, not line by line).

Each snapshot's ``state.json`` is the source of truth; the index is derived from it. A capture
appends its row; editing a label, or finding the index missing, unreadable or different from what
the ``state.json`` files say, rewrites it from all of them (:func:`rebuild`). Every change holds the
project's lock (threads, and processes through ``snapshots.csv.lock``) and lands whole: an appended
row in one write, a rebuilt index by an atomic rename. An edit marks the index stale
(``snapshots.csv.dirty``) before it touches a ``state.json``, so that an edit cut short (a crash, a
full disk) is repaired by the next capture or download.

Columns, in this order (stable; new columns are only ever added at the end):

- ``snapshot_id``: the snapshot's folder name, its local capture time ``YYYYmmdd-HHMMSS-mmm``.
- ``captured_at_local``, ``captured_at_utc``: ISO 8601 with milliseconds and UTC offset.
- ``defective``: ``true``, ``false``, or empty: not labelled (taken before labels existed; label it
  with the API or the snapshot list).
- ``comment``: free text as entered (at most 500 characters), or empty.
- ``environment_id``, ``environment_name``, ``session_id``.
- ``control_mode``: what drove the follower: ``teleop``, ``model``, …; empty when it was held still.
- ``dataset_id``: the dataset the session had loaded, or empty.
- ``joint_names``: the joints of the columns below, ``;``-separated (SO-101: ``shoulder_pan.pos`` …
  ``gripper.pos``).
- ``follower_joint_1`` … ``follower_joint_6``: the follower's positions as the session measured
  them, in the units it records into datasets (normalized for an SO-101).
- ``leader_joint_1`` … ``leader_joint_6``: the leader's, same units; empty unless it was teleoperating.
- ``camera_names``: every camera's name, ``;``-separated.
- ``camera_<n>_name``, ``camera_<n>_color``, ``camera_<n>_depth``, ``camera_<n>_depth_preview`` for
  ``n`` = 1 … 4: the environment's first four cameras and their image files, relative to this file
  (``<snapshot_id>/<camera>.color.png``); empty where a camera or file is missing.
- ``warnings``: what could not be captured, ``" | "``-separated.

Rows are oldest first after a rebuild; appended rows follow in the order captures finished.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import shutil
import threading
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

CSV_NAME = "snapshots.csv"
LOCK_NAME = f"{CSV_NAME}.lock"
DIRTY_NAME = f"{CSV_NAME}.dirty"
"""Present while an edit may have changed a ``state.json`` that the index does not show yet."""
STATE_FILE = "state.json"
JOINT_COLUMNS = 6
"""Joint columns per robot: an SO-101 has six; a robot with more has the rest in ``state.json`` only."""
CAMERA_SLOTS = 4
"""Cameras with file columns; more are listed in ``camera_names``, their files in ``state.json``."""
CAMERA_FILES = ("color", "depth", "depth_preview")

COLUMNS: tuple[str, ...] = (
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
    *(f"follower_joint_{index}" for index in range(1, JOINT_COLUMNS + 1)),
    *(f"leader_joint_{index}" for index in range(1, JOINT_COLUMNS + 1)),
    "camera_names",
    *(f"camera_{slot}_{field}" for slot in range(1, CAMERA_SLOTS + 1) for field in ("name", *CAMERA_FILES)),
    "warnings",
)

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


@contextlib.contextmanager
def locked(project_root: Path) -> Iterator[None]:
    """Hold the project's index: across threads, and across processes through a lock file."""
    project_root.mkdir(parents=True, exist_ok=True)
    with _locks_guard:
        lock = _locks.setdefault(str(project_root.resolve()), threading.Lock())
    with lock, (project_root / LOCK_NAME).open("a") as lock_file:
        try:
            import fcntl
        except ImportError:  # no fcntl (Windows): the thread lock still serializes this process
            pass
        else:
            fcntl.flock(lock_file, fcntl.LOCK_EX)  # released when the file closes
        yield


def row(document: dict[str, Any]) -> list[str]:
    """The index row of a snapshot's ``state.json``."""
    snapshot_id = str(document.get("snapshot_id", ""))
    captured = document.get("captured_at") or {}
    session = document.get("session") or {}
    environment = document.get("environment") or {}
    robots = document.get("robots") or {}
    follower = (robots.get("follower") or {}).get("joint_positions") or {}
    leader = (robots.get("leader") or {}).get("joint_positions") or {}
    names = list(follower) or list(leader)
    cameras = [camera for camera in document.get("cameras") or [] if isinstance(camera, dict)]
    values: dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "captured_at_local": captured.get("local") or "",
        "captured_at_utc": captured.get("utc") or "",
        "defective": _label(document.get("defective")),
        "comment": _comment(document.get("comment")),
        "environment_id": environment.get("id") or "",
        "environment_name": environment.get("name") or "",
        "session_id": session.get("id") or "",
        "control_mode": (session.get("control") or {}).get("type") or "",
        "dataset_id": (session.get("dataset") or {}).get("dataset_id") or "",
        "joint_names": ";".join(map(str, names)),
        "camera_names": ";".join(str(camera.get("name", "")) for camera in cameras),
        "warnings": " | ".join(map(str, document.get("warnings") or [])),
    }
    for index, name in enumerate(names[:JOINT_COLUMNS], start=1):
        values[f"follower_joint_{index}"] = _number(follower.get(name))
        values[f"leader_joint_{index}"] = _number(leader.get(name))
    for slot, camera in enumerate(cameras[:CAMERA_SLOTS], start=1):
        files = camera.get("files") or {}
        values[f"camera_{slot}_name"] = camera.get("name", "")
        for kind in CAMERA_FILES:
            values[f"camera_{slot}_{kind}"] = f"{snapshot_id}/{files[kind]}" if files.get(kind) else ""
    return [str(values.get(column, "")) for column in COLUMNS]


def append(project_root: Path, document: dict[str, Any]) -> None:
    """Add a new snapshot's row, once its ``state.json`` is written.

    The whole index is rebuilt instead if it is missing or unreadable, and nothing is added if
    the row is already there (a concurrent rebuild may have picked the snapshot up).
    """
    with locked(project_root):
        path = project_root / CSV_NAME
        rows = _current_rows(project_root)
        if rows is None:
            _rebuild_locked(project_root)
            return
        if str(document.get("snapshot_id")) in {line[0] for line in rows}:
            return
        # A whole row in one write, under the lock: no reader or writer sees half of it.
        with path.open("ab") as index:
            index.write(_encode([_row_or_placeholder(document)], header=False))
            index.flush()
            os.fsync(index.fileno())


def rebuild(project_root: Path) -> int:
    """Write the index anew from every snapshot's ``state.json``. Returns its rows."""
    with locked(project_root):
        return _rebuild_locked(project_root)


def ensure(project_root: Path) -> bool:
    """Rebuild the index unless it holds exactly the rows the snapshots' ``state.json`` files make.

    So it is rebuilt if it is missing, unreadable, marked stale, lists a snapshot twice or not at
    all, or any value differs (a label edited by hand, or an edit cut short).

    Returns:
        Whether it was rebuilt.
    """
    if not project_root.is_dir():
        return False
    with locked(project_root):
        return _ensure_locked(project_root)


def contents(project_root: Path, *, rebuild_first: bool = False) -> bytes:
    """The index, checked (:func:`ensure`) or rebuilt first; just the header if there are no snapshots."""
    if not project_root.is_dir():
        return _encode([], header=True)
    with locked(project_root):
        if rebuild_first:
            _rebuild_locked(project_root)
        else:
            _ensure_locked(project_root)
        return (project_root / CSV_NAME).read_bytes()


def update(project_root: Path, snapshot_id: str, changes: dict[str, Any]) -> tuple[dict[str, Any], OSError | None]:
    """Change fields of a snapshot's ``state.json`` (such as its label), then rebuild the index.

    The index is marked stale first. If it cannot be rebuilt once ``state.json`` is changed, the
    change stands and the index stays marked, for the next capture or download to rebuild it.

    Returns:
        The updated ``state.json``, and why the index could not be rebuilt, if it could not.

    Raises:
        FileNotFoundError: There is no such snapshot.
        OSError: ``state.json`` could not be changed; it is as it was.
    """
    folder = project_root / snapshot_id
    if folder.resolve().parent != project_root.resolve() or not (folder / STATE_FILE).is_file():
        raise FileNotFoundError(folder)
    with locked(project_root):
        document = json.loads((folder / STATE_FILE).read_text(encoding="utf-8"))
        document.update(changes)
        (project_root / DIRTY_NAME).touch()
        write_json(folder / STATE_FILE, document)
        try:
            _rebuild_locked(project_root)
        except OSError as error:
            logger.exception("Changed {} but could not rebuild {}; the next check will", folder.name, CSV_NAME)
            return document, error
        return document, None


def trash(project_root: Path, snapshot_id: str, trash_dir: Path) -> tuple[Path, OSError | None]:
    """Move a snapshot's folder into ``trash_dir`` (not deleted: it can be moved back), then rebuild the index.

    Returns:
        Where the folder is now, and why the index could not be rebuilt, if it could not; the next
        capture or download rebuilds it then.

    Raises:
        FileNotFoundError: There is no such snapshot.
    """
    folder = project_root / snapshot_id
    if folder.resolve().parent != project_root.resolve():
        raise FileNotFoundError(folder)
    with locked(project_root):
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        destination = trash_dir / snapshot_id
        for attempt in range(1, 100):
            if not destination.exists():
                break
            destination = trash_dir / f"{snapshot_id}-{attempt}"
        trash_dir.mkdir(parents=True, exist_ok=True)
        (project_root / DIRTY_NAME).touch()
        shutil.move(folder, destination)
        logger.info("Moved state snapshot {} to {}", folder, destination)
        try:
            _rebuild_locked(project_root)
        except OSError as error:
            logger.exception("Moved {} but could not rebuild {}; the next check will", snapshot_id, CSV_NAME)
            return destination, error
        return destination, None


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Write a ``state.json`` atomically, so that a rebuild never reads half of one."""
    partial = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    partial.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    partial.replace(path)


def _number(value: Any) -> str:
    try:
        return "" if value is None else repr(float(value))
    except (TypeError, ValueError):
        return ""


def _label(defective: Any) -> str:
    """``true``, ``false``, or empty for a snapshot that was never labelled."""
    if isinstance(defective, bool):
        return "true" if defective else "false"
    return ""


def _comment(comment: Any) -> str:
    return comment if isinstance(comment, str) else ""


def _row_or_placeholder(document: dict[str, Any]) -> list[str]:
    """:func:`row`, or for a ``state.json`` it cannot make sense of, its id, label and why."""
    try:
        return row(document)
    except (AttributeError, TypeError, ValueError) as error:
        logger.warning("Snapshot {} has an unexpected state.json: {}", document.get("snapshot_id"), error)
        values = {
            "snapshot_id": str(document.get("snapshot_id", "")),
            "defective": _label(document.get("defective")),
            "comment": _comment(document.get("comment")),
            "warnings": f"state.json has an unexpected shape: {error}",
        }
        return [values.get(column, "") for column in COLUMNS]


def _encode(rows: list[list[str]], *, header: bool) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    if header:
        writer.writerow(COLUMNS)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _documents(project_root: Path) -> list[tuple[str, dict[str, Any]]]:
    """Every snapshot's folder name and ``state.json``, oldest first; unreadable ones are left out."""
    documents = []
    for folder in sorted(path for path in project_root.iterdir() if path.is_dir()):
        try:
            document = json.loads((folder / STATE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(document, dict):
            documents.append((folder.name, document | {"snapshot_id": folder.name}))
    return documents


def _ensure_locked(project_root: Path) -> bool:
    rows = _current_rows(project_root)
    expected = [_row_or_placeholder(document) for _, document in _documents(project_root)]
    # In any order: appended rows follow the order captures finished in.
    if rows is not None and sorted(rows) == sorted(expected):
        return False
    _write_locked(project_root, expected)
    return True


def _rebuild_locked(project_root: Path) -> int:
    rows = [_row_or_placeholder(document) for _, document in _documents(project_root)]
    _write_locked(project_root, rows)
    return len(rows)


def _write_locked(project_root: Path, rows: list[list[str]]) -> None:
    path = project_root / CSV_NAME
    partial = path.with_name(f".{CSV_NAME}.{os.getpid()}.{threading.get_ident()}.tmp")
    partial.write_bytes(_encode(rows, header=True))
    partial.replace(path)  # atomic: a reader sees the old index or the new one
    (project_root / DIRTY_NAME).unlink(missing_ok=True)
    logger.info("Rebuilt {} with {} snapshot(s)", path, len(rows))


def _current_rows(project_root: Path) -> list[list[str]] | None:
    """The index's rows, or None if it is marked stale, missing, or not exactly as this module writes it.

    Not as written: another header, a row of another width, an id twice, or no line end after the
    last row (an append would run into it).
    """
    if (project_root / DIRTY_NAME).exists():
        return None
    path = project_root / CSV_NAME
    try:
        data = path.read_bytes()
        rows = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    if not data.endswith(b"\n") or not rows or tuple(rows[0]) != COLUMNS:
        return None
    if any(len(line) != len(COLUMNS) for line in rows[1:]):
        return None
    ids = [line[0] for line in rows[1:]]
    return rows[1:] if len(set(ids)) == len(ids) else None
