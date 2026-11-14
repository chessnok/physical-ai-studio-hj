"""A state snapshot: a live session's latest camera frames and robot state, saved to disk."""

from datetime import datetime
from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_validator

COMMENT_MAX_LENGTH = 500


def _printable(comment: str) -> str:
    if "\x00" in comment:
        raise ValueError("The comment must not contain NUL characters")
    return comment


SnapshotComment = Annotated[
    str,
    StringConstraints(strip_whitespace=True, max_length=COMMENT_MAX_LENGTH),
    AfterValidator(_printable),
    Field(description=f"Free text, at most {COMMENT_MAX_LENGTH} characters; leading and trailing spaces are dropped"),
]


class StateSnapshotLabel(BaseModel):
    """How a new snapshot is labelled: what it shows, for anomaly detection, and a note."""

    model_config = ConfigDict(extra="forbid")

    defective: bool = Field(default=False, description="Whether the snapshot shows a defective part")
    comment: SnapshotComment = ""


class StateSnapshotLabelUpdate(BaseModel):
    """A change to a saved snapshot's label: the fields given, at least one."""

    model_config = ConfigDict(extra="forbid")

    defective: bool | None = Field(default=None, description="Whether the snapshot shows a defective part")
    comment: SnapshotComment | None = None

    @model_validator(mode="after")
    def _changes_something(self) -> Self:
        if self.defective is None and self.comment is None:
            raise ValueError("Give defective, comment or both")
        return self


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
    defective: bool | None = Field(
        default=None,
        description="Whether the snapshot shows a defective part; null if it was never labelled "
        "(taken before labels existed)",
    )
    comment: str = Field(default="", description="The note saved with the snapshot")
    csv_warning: str | None = Field(
        default=None,
        description="Set when a label change is saved but snapshots.csv could not be rewritten yet; "
        "the next capture or CSV download brings it up to date",
    )
