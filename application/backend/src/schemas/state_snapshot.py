"""A state snapshot: a live session's latest camera frames and robot state, saved to disk."""

from datetime import datetime

from pydantic import BaseModel, Field


class SnapshotCamera(BaseModel):
    name: str
    key: str
    """The camera's store and dataset key."""
    files: list[str] = Field(description="Image files saved for it, relative to the snapshot folder")
    color: bool
    depth: bool


class StateSnapshot(BaseModel):
    snapshot_id: str = Field(description="The snapshot's folder name: its local capture time, YYYYmmdd-HHMMSS-mmm")
    path: str = Field(description="The snapshot folder on the backend's machine")
    captured_at: datetime
    session_id: str | None
    environment_name: str | None
    cameras: list[SnapshotCamera]
    follower: bool = Field(description="Whether the follower's joint positions were captured")
    leader: bool = Field(description="Whether the leader's joint positions were captured")
    warnings: list[str] = Field(default_factory=list, description="What could not be captured, and why")
