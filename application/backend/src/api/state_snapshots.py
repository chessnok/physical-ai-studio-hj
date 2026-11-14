"""State snapshots of a project's live runtime session: capture, list, label, and the CSV index.

See :mod:`runtime_session.snapshot`. Capturing reads only what the session's store
already holds: it opens no device, commands no robot and leaves a recording alone.
Each snapshot has a label (``defective``, ``comment``), given when it is captured and
changeable later; ``snapshots.csv`` (:mod:`runtime_session.snapshot_index`) lists them all.
"""

import asyncio
import http
from pathlib import Path as FilePath
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Path, Query, Response, status

from api.dependencies import RuntimeSessionRegistryDep
from build_info import _run_git, get_build_info_async
from exceptions import BaseException as StudioError
from exceptions import RuntimeSessionCommandError
from runtime_session import snapshot, snapshot_index
from runtime_session.registry import RuntimeSessionRegistry
from runtime_session.session import RuntimeSession
from schemas.state_snapshot import StateSnapshot, StateSnapshotLabel, StateSnapshotLabelUpdate
from settings import get_settings

router = APIRouter(prefix="/api/projects/{project_id}/state_snapshots", tags=["State Snapshots"])

SNAPSHOT_ID_PATTERN = r"^\d{8}-\d{6}-\d{3}(-\d{1,2})?$"
"""A snapshot folder's name: ``YYYYmmdd-HHMMSS-mmm``, with ``-<n>`` if several were taken in one millisecond."""


def _project_root(project_id: UUID) -> FilePath:
    return get_settings().storage_dir / "state_snapshots" / str(project_id)


def _live_session(registry: RuntimeSessionRegistry, project_id: UUID, session_id: str | None) -> RuntimeSession:
    """The project's session to capture: ``session_id``, or the one live session with an environment.

    Raises:
        RuntimeSessionCommandError: No such session, none is live, or several are and none was named.
    """
    if session_id is not None:
        session = registry.get(session_id)
        if session is None or session.project_id != project_id:
            raise RuntimeSessionCommandError(
                "That live session has ended. Open the environment or the recording page again.",
                error_code="no_live_session",
            )
        candidates = [session]
    else:
        candidates = [session for session in registry.sessions() if session.project_id == project_id]
    live = [session for session in candidates if session.loaded_environment is not None and session.store is not None]
    if not live:
        raise RuntimeSessionCommandError(
            "No live session with an environment to snapshot. Open the environment or the recording page first.",
            error_code="no_live_session",
        )
    if len(live) > 1:
        raise RuntimeSessionCommandError(
            "Several sessions of this project are live; say which one to snapshot.", error_code="ambiguous_session"
        )
    return live[0]


@router.post("", status_code=status.HTTP_201_CREATED)
async def capture_state_snapshot(
    project_id: UUID,
    registry: RuntimeSessionRegistryDep,
    session_id: Annotated[
        str | None, Query(description="The live session to capture; the project's only one if omitted")
    ] = None,
    label: Annotated[
        StateSnapshotLabel | None, Body(description="The snapshot's label; not defective, no comment if omitted")
    ] = None,
) -> StateSnapshot:
    """Save the live session's latest camera frames and robot state, with a label, and add it to the CSV.

    Read-only: it copies what the session's store holds, so it neither opens a device a second
    time nor commands a robot, and a recording goes on undisturbed.
    """
    label = label or StateSnapshotLabel()
    session = _live_session(registry, project_id, session_id)
    # Copied at once, on the event loop, so the frames and joints are of one moment; written in a thread.
    captured = snapshot.capture(session)
    build = await get_build_info_async()
    branch = await asyncio.to_thread(_run_git, "rev-parse", "--abbrev-ref", "HEAD")
    app = {"studio_version": get_settings().version, **build.__dict__, "git_branch": branch}
    root = get_settings().storage_dir / "state_snapshots"
    return await asyncio.to_thread(
        snapshot.write, captured, root, build=app, defective=label.defective, comment=label.comment
    )


@router.get("")
async def list_state_snapshots(
    project_id: UUID,
    limit: Annotated[int | None, Query(ge=1, description="At most this many, the newest")] = None,
) -> list[StateSnapshot]:
    """The project's state snapshots with their labels, newest first."""
    return await asyncio.to_thread(snapshot.list_snapshots, _project_root(project_id), limit)


@router.get(
    "/snapshots.csv",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}, "description": "The CSV index, one row per snapshot"}},
)
async def download_state_snapshots_csv(
    project_id: UUID,
    rebuild: Annotated[
        bool, Query(description="Rewrite it from every snapshot's state.json first, even if it looks current")
    ] = False,
) -> Response:
    """The project's ``snapshots.csv``: one row per snapshot, with its label, times, joints and image files.

    It is checked first, and rewritten from the snapshots' ``state.json`` if it is missing,
    unreadable or does not list every snapshot exactly once. Just the header if there are none.
    """
    data = await asyncio.to_thread(snapshot_index.contents, _project_root(project_id), rebuild_first=rebuild)
    return Response(
        content=data,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="state_snapshots_{project_id}.csv"'},
    )


@router.patch("/{snapshot_id}")
async def update_state_snapshot(
    project_id: UUID,
    snapshot_id: Annotated[str, Path(pattern=SNAPSHOT_ID_PATTERN, description="The snapshot's folder name")],
    changes: StateSnapshotLabelUpdate,
) -> StateSnapshot:
    """Change a snapshot's label (``defective``, ``comment`` or both) in its ``state.json`` and the CSV."""
    try:
        return await asyncio.to_thread(
            snapshot.update_label,
            _project_root(project_id),
            snapshot_id,
            defective=changes.defective,
            comment=changes.comment,
        )
    except FileNotFoundError:
        raise StudioError(
            f"This project has no state snapshot {snapshot_id}.",
            error_code="state_snapshot_not_found",
            http_status=http.HTTPStatus.NOT_FOUND,
        ) from None
