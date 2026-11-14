from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from schemas import Episode, EpisodeInfo

if TYPE_CHECKING:
    from internal_datasets.mutations.recording_mutation import RecordingMutation


class EpisodeNotFoundError(LookupError):
    """The dataset has no episode with the given index."""


class DatasetClient(ABC):
    type: str
    has_episodes: bool = False

    @property
    @abstractmethod
    def exists_on_disk(self) -> bool:
        """Whether the dataset exists on disk."""

    @abstractmethod
    def prepare_for_writing(self) -> None:
        """Processes for writing episodes."""

    @abstractmethod
    def get_episodes(self) -> list[Episode]:
        """Get episodes of dataset."""

    @abstractmethod
    def get_episode_infos(self) -> list[EpisodeInfo]:
        """Get episode summaries without heavy frame/action data."""

    @abstractmethod
    def find_episode(self, episode_index: int) -> Episode | None:
        """Find episode by index or return None."""

    @abstractmethod
    def get_tasks(self) -> list[str]:
        """Get Tasks in dataset."""

    @abstractmethod
    def get_video_path(self, episode: int, camera: str) -> Path:
        """Get Video path of specific episode and camera."""

    @abstractmethod
    def get_video_keys(self) -> list[str]:
        """Get the video keys used to record the dataset"""

    def dropped_video_frames(self) -> dict[str, int]:
        """Frames the video encoders dropped in the open episode, by video key."""
        return {}

    def last_episode_video_problems(self) -> list[str]:
        """How the last saved episode's videos do not match its rows; empty when they do."""
        return []

    def get_depth_keys(self) -> list[str]:
        """The video keys that hold depth maps rather than color images."""
        return []

    def get_feature_info(self, key: str) -> dict:  # noqa: ARG002
        """The ``info`` LeRobot recorded for a feature, such as a depth video's quantization."""
        return {}

    @abstractmethod
    def get_episode_thumbnail_png(
        self,
        episode_index: int,
        video_key: str,
        width: int = 320,
        height: int = 240,
    ) -> tuple[bytes, Path] | None:
        """Build a PNG thumbnail and return bytes with source video path."""

    @abstractmethod
    def create(self, fps: int, features: dict, robot_type: str) -> None:
        """Create dataset."""

    @abstractmethod
    def delete_episodes(self, episode_indices: list[int], output_path: Path) -> "DatasetClient":
        """Copy over repo without given episode_indices to output_path."""

    @abstractmethod
    def set_episode_task(self, episode_index: int, task: str) -> None:
        """Make ``task`` the only task of the episode, in place.

        Raises:
            EpisodeNotFoundError: The dataset has no such episode.
        """

    @abstractmethod
    def add_frame(self, obs: dict, act: dict, task: str) -> None:
        """Add frame to recording buffer."""

    @abstractmethod
    def save_episode(self) -> None:
        """Save current recording buffer as episode."""

    @abstractmethod
    def discard_buffer(self) -> None:
        """Discard current recording buffer."""

    @abstractmethod
    def teardown(self) -> None:
        """Clean up dataset and delete if no episodes."""

    @abstractmethod
    def delete(self) -> None:
        """Delete dataset."""

    @abstractmethod
    def finalize(self) -> None:
        """Finalize changes to dataset."""

    @abstractmethod
    def overwrite(self, source: "DatasetClient") -> None:
        """Overwrite dataset with given dataset."""

    @abstractmethod
    def start_recording_mutation(self, fps: int, features: dict, robot_type: str) -> "RecordingMutation":
        """Start recording mutation."""
