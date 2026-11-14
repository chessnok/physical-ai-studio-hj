"""State snapshots of a project's live runtime session: capture one, list them.

See :mod:`runtime_session.snapshot`. Capturing reads only what the session's store
already holds: it opens no device, commands no robot and leaves a recording alone.
"""

import asyncio
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, status

from api.dependencies import RuntimeSessionRegistryDep
from build_info import _run_git, get_build_info_async
from exceptions import RuntimeSessionCommandError
from runtime_session import snapshot
from runtime_session.registry import RuntimeSessionRegistry
from runtime_session.session import RuntimeSession
from schemas.state_snapshot import StateSnapshot
from settings import get_settings

router = APIRouter(prefix="/api/projects/{project_id}/state_snapshots", tags=["State Snapshots"])


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
) -> StateSnapshot:
    """Save the live session's latest camera frames and robot state.

    Read-only: it copies what the session's store holds, so it neither opens a device a second
    time nor commands a robot, and a recording goes on undisturbed.
    """
    session = _live_session(registry, project_id, session_id)
    # Copied at once, on the event loop, so the frames and joints are of one moment; written in a thread.
    captured = snapshot.capture(session)
    build = await get_build_info_async()
    branch = await asyncio.to_thread(_run_git, "rev-parse", "--abbrev-ref", "HEAD")
    app = {"studio_version": get_settings().version, **build.__dict__, "git_branch": branch}
    root = get_settings().storage_dir / "state_snapshots"
    return await asyncio.to_thread(snapshot.write, captured, root, build=app)


@router.get("")
async def list_state_snapshots(project_id: UUID) -> list[StateSnapshot]:
    """The project's state snapshots, newest first."""
    root = get_settings().storage_dir / "state_snapshots" / str(project_id)
    return await asyncio.to_thread(snapshot.list_snapshots, root)
