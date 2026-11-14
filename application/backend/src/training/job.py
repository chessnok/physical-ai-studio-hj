# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""One training run, described by a spec and executed by one function.

:class:`TrainingJobSpec` is the complete configuration of a training run —
 policy, epoch budget, batch size, precision, device — and nothing else. It holds
no paths, no job identity, and no transport details, so the same spec can be
built in-process or sent over HTTP to a remote trainer and mean the same thing
in both places.

:func:`run_training_job` executes a spec: it builds the datamodule and policy,
runs Lightning, saves the final checkpoint into ``output_dir``, and exports the
policy to every backend it supports. Where the data comes from, where the
result goes, how progress is reported, and how cancellation is signalled are all
arguments — the runner owns none of that policy.

Example:
    >>> spec = TrainingJobSpec(policy="act", max_epochs=5, batch_size=8)
    >>> run_training_job(  # doctest: +SKIP
    ...     spec,
    ...     dataset_root="/data/snapshot",
    ...     output_dir="/models/abc",
    ...     cache_dir="/cache/abc",
    ...     report=lambda progress, message, extra: None,
    ...     should_stop=lambda: False,
    ... )
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

if TYPE_CHECKING:
    from physicalai.policies.base import Policy
    from physicalai.train.callbacks import ReportFn, StopFn

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "model.ckpt"
"""Filename of the ordinary (flow-matching) checkpoint."""

SNAPFLOW_CHECKPOINT_PREFIX = "snapflow-"
"""Prefix :class:`~physicalai.train.callbacks.SnapFlowPhaseCallback` applies to
``ModelCheckpoint`` filenames once distillation starts, so phase-2 checkpoints
land beside phase-1's instead of overwriting them."""

SNAPFLOW_CHECKPOINT_NAME = f"{SNAPFLOW_CHECKPOINT_PREFIX}{CHECKPOINT_NAME}"
"""Filename of the distilled checkpoint, written only once a SnapFlow run
reaches its phase boundary. Preferred over :data:`CHECKPOINT_NAME` by export,
resume, and download whenever it exists; see :func:`resolve_checkpoint`."""

SNAPFLOW_POLICIES = frozenset({"pi05", "smolvla"})
"""Policies that implement ``enable_snapflow()`` and can be distilled.

SnapFlow is a flow-matching technique; only the flow-matching policies mix in
:class:`~physicalai.policies.mixins.SnapFlowPolicyMixin`.
"""

EXPORTS_DIRNAME = "exports"
"""Subdirectory of ``output_dir`` holding one directory per export backend."""

_DATASET_REPO_ID = "snapshot"
"""Placeholder repo id: datasets are always loaded from a local root here."""

PRETRAINED_BASE_CHECKPOINTS: dict[str, str] = {
    "molmoact2": "allenai/MolmoAct2",
    "pi05": "lerobot/pi05_base",
    "rldx1": "RLWRLD/RLDX-1-PT",
    "smolvla": "lerobot/smolvla_base",
    "xr0": "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain",
}
"""Hub checkpoints used to initialize policies that only fine-tune from pretrained weights."""

PEFT_POLICIES = frozenset({"pi05", "molmoact2"})
"""Policies whose ``Config`` mixes in ``physicalai.policies.mixins.peft.PeftConfigMixin`` and
support LoRA/DoRA fine-tuning."""

EARLY_STOPPING_SUMMARY_NAME = "early_stopping.json"
"""What early stopping did, written into ``output_dir`` beside the checkpoint
whenever a run has early stopping enabled. It travels inside the model
artifact, so Studio reads the same file after a local, Modal, AWS Batch, or
remote-trainer run (see :func:`read_early_stopping_summary`)."""

EARLY_STOPPING_UNSUPPORTED_MESSAGE = (
    "The remote trainer does not support early stopping: it runs trainer code from before the feature "
    "existed. Redeploy or update the trainer (for Modal: deploy the trainer again from this Studio), "
    "or train without early stopping."
)
"""Shown instead of the raw validation error an older trainer returns for the
early-stopping fields (see :func:`explain_trainer_rejection`)."""


def _is_unset(value: object) -> bool:
    """Serialization filter: the early-stopping fields are sent only when early stopping is on."""
    return value is None


def _is_zero(value: float) -> bool:
    return value == 0.0


class RunOptions(BaseModel):
    """Runtime-only controls which are not part of the training payload."""

    resume_from: Path | str | None = None
    hf_token: SecretStr | None = None


class TrainingJobSpec(BaseModel):
    """Everything needed to train one policy, and nothing else.

    Deliberately free of paths, identifiers, and transport concerns: this is
    the shared contract between an in-process runner and a remote trainer, so
    it must describe *what* to train rather than *where*. Runner-local
    concerns (dataset location, output location, resume checkpoint) are
    arguments to :func:`run_training_job`.

    Example:
        >>> spec = TrainingJobSpec(policy="act", max_epochs=5)
        >>> spec.precision
        'bf16-mixed'
    """

    model_config = ConfigDict(extra="forbid")

    policy: str = Field(description="Policy name, e.g. 'act', 'pi05', 'molmoact2', or 'smolvla'.")
    policy_source: Literal["physicalai", "lerobot"] = Field(
        default="physicalai",
        description="Which implementation of the policy to train.",
    )
    max_epochs: int = Field(
        default=5,
        ge=1,
        description=(
            "Total training epoch budget. When SnapFlow distillation is enabled this already "
            "includes the distillation phase (see snapflow_start_epoch)."
        ),
    )
    batch_size: int = Field(default=8, ge=1, description="Training batch size.")
    num_workers: int | Literal["auto"] = Field(default="auto", description="Dataloader worker count.")
    val_split: float = Field(default=0.1, ge=0.0, lt=1.0, description="Fraction of episodes held out for validation.")
    precision: str = Field(default="bf16-mixed", description="Lightning precision, e.g. '32-true' or 'bf16-mixed'.")
    compile_model: bool = Field(default=False, description="Whether to torch.compile the policy forward pass.")
    augment_images: bool = Field(
        default=False,
        description="Whether to augment training images with the default pipeline.",
    )
    auto_scale_batch_size: bool = Field(default=False, description="Whether to search for the largest fitting batch.")
    image_key_reorder_map: dict[str, int] = Field(
        default_factory=dict,
        description="Dataset image key -> camera slot index, for policies with a fixed camera order.",
    )
    num_cameras: int = Field(
        default=0,
        ge=0,
        description="Camera slots the policy reads; unfilled slots become masked empty images. 0 keeps the batch's.",
    )
    export_backends: list[str] | None = Field(
        default=None,
        description="Export formats to produce. None exports every backend the policy supports.",
    )
    snapflow_start_epoch: int | None = Field(
        default=None,
        ge=1,
        description=("Epoch at which to switch from flow matching to SnapFlow self-distillation."),
    )
    device_type: str | None = Field(
        default=None,
        description="Accelerator to train on ('xpu', 'cuda', 'cpu', ...). None auto-detects.",
    )
    device_index: int | None = Field(
        default=None,
        ge=0,
        description="Zero-based index of the accelerator to train on. None lets Lightning pick one.",
    )
    lora_enabled: bool = Field(
        default=False,
        description=(
            "Fine-tune with LoRA/DoRA instead of full fine-tuning: freezes the base model and "
            f"trains small low-rank adapters. Only available for {sorted(PEFT_POLICIES)}."
        ),
    )
    lora_rank: int = Field(default=32, ge=1, le=256, description="LoRA rank. Ignored unless lora_enabled.")
    lora_alpha: int | None = Field(
        default=None,
        ge=1,
        description="LoRA scaling numerator (scaling = lora_alpha / lora_rank). None defaults to lora_rank.",
    )
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0, description="Dropout applied to LoRA adapter inputs.")
    lora_use_dora: bool = Field(
        default=False,
        description="Use DoRA (Weight-Decomposed LoRA) instead of plain LoRA. Ignored unless lora_enabled.",
    )
    # Both early-stopping fields are left out of the serialized spec while early
    # stopping is off, so a spec without it still validates on a trainer that
    # predates the feature (this model forbids unknown fields).
    early_stopping_patience: int | None = Field(
        default=None,
        ge=1,
        exclude_if=_is_unset,
        description=(
            "Stop when val/loss has not improved for this many validation epochs in a row. "
            "None disables early stopping. Needs val_split > 0. On a SnapFlow run it applies to the "
            "flow-matching phase only: stopping it starts distillation early."
        ),
    )
    early_stopping_min_delta: float = Field(
        default=0.0,
        ge=0.0,
        exclude_if=_is_zero,
        description="Smallest val/loss decrease that counts as an improvement. Ignored unless early stopping is on.",
    )
    run_options: RunOptions = Field(default_factory=RunOptions)

    @model_validator(mode="after")
    def validate_lora(self) -> TrainingJobSpec:
        """Reject a LoRA request the run cannot honour, and DoRA requested without LoRA."""
        if self.lora_use_dora and not self.lora_enabled:
            msg = "lora_use_dora requires lora_enabled; DoRA is a variant of LoRA, not a standalone mode."
            raise ValueError(msg)
        if not self.lora_enabled:
            return self
        if self.policy_source != "physicalai":
            msg = (
                f"LoRA/DoRA fine-tuning requires policy_source='physicalai', got {self.policy_source!r}: "
                "lerobot-wrapped policies do not accept lora_* constructor arguments."
            )
            raise ValueError(msg)
        if self.policy.lower() not in PEFT_POLICIES:
            msg = (
                f"LoRA/DoRA fine-tuning is not available for policy {self.policy!r}; "
                f"supported policies are {sorted(PEFT_POLICIES)}."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_snapflow(self) -> TrainingJobSpec:
        """Reject a SnapFlow boundary the run cannot honour.

        Checked here rather than at the trainer so a bad combination fails at
        submission.

        Returns:
            The validated spec.

        Raises:
            ValueError: If the policy cannot be distilled, or the boundary
                leaves no epochs for the distillation phase.
        """
        if self.snapflow_start_epoch is None:
            return self
        if self.policy.lower() not in SNAPFLOW_POLICIES:
            msg = (
                f"SnapFlow distillation is not supported for policy {self.policy!r}; "
                f"supported policies are {sorted(SNAPFLOW_POLICIES)}."
            )
            raise ValueError(msg)
        if self.snapflow_start_epoch >= self.max_epochs:
            msg = (
                f"snapflow_start_epoch ({self.snapflow_start_epoch}) must be below max_epochs "
                f"({self.max_epochs}) so at least one epoch is spent distilling."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_early_stopping(self) -> TrainingJobSpec:
        """Reject early stopping without a validation split; drop a min_delta that has no effect.

        Returns:
            The validated spec.

        Raises:
            ValueError: If early stopping is on but no episodes are held out
                for the ``val/loss`` it monitors.
        """
        if self.early_stopping_patience is None:
            # Ignored without early stopping; zeroed so it is not sent over the wire either.
            self.early_stopping_min_delta = 0.0
            return self
        if self.val_split <= 0:
            msg = (
                "Early stopping watches val/loss, which needs a validation split: "
                "set val_split above 0 or turn early stopping off."
            )
            raise ValueError(msg)
        return self


class EarlyStoppingSummary(BaseModel):
    """What early stopping did in one run (contents of :data:`EARLY_STOPPING_SUMMARY_NAME`).

    Epoch numbers are 1-based, as a person counts them. On a SnapFlow run
    early stopping watches the flow-matching phase only; ``stopped_after_epoch``
    then ends that phase, and distillation still runs its full budget
    (``distillation_epochs``) right after it.

    Example:
        >>> summary = EarlyStoppingSummary(
        ...     patience=3, min_delta=0.0, max_epochs=30, epochs_trained=12,
        ...     stopped_after_epoch=12, best_epoch=9, best_val_loss=0.5194,
        ... )
        >>> summary.message
        'stopped early after epoch 12 of 30; best epoch 9 (val/loss 0.5194)'
    """

    model_config = ConfigDict(extra="ignore")

    patience: int
    min_delta: float
    max_epochs: int = Field(description="Planned epoch budget, distillation included.")
    epochs_trained: int = Field(description="Epochs that actually ran, distillation included.")
    stopped_after_epoch: int | None = Field(
        default=None,
        description="Last epoch trained before early stopping ended the run (or phase); None if it never did.",
    )
    reason: str | None = Field(default=None, description="Lightning's stopping reason, e.g. 'patience_exhausted'.")
    best_epoch: int | None = Field(default=None, description="Epoch with the lowest val/loss in the watched phase.")
    best_val_loss: float | None = None
    flow_matching_epochs: int | None = Field(
        default=None, description="SnapFlow runs: planned flow-matching epochs before distillation."
    )
    distillation_epochs: int | None = Field(default=None, description="SnapFlow runs: distillation epochs.")

    @property
    def stopped_early(self) -> bool:
        return self.stopped_after_epoch is not None

    @property
    def message(self) -> str | None:
        """One line for the job status, or None when early stopping never triggered."""
        if self.stopped_after_epoch is None:
            return None
        best = (
            f"best epoch {self.best_epoch} (val/loss {self.best_val_loss:.4g})"
            if self.best_epoch is not None and self.best_val_loss is not None
            else "no finite val/loss"
        )
        cause = "; val/loss was not finite" if self.reason == "non_finite_metric" else ""
        if self.distillation_epochs is not None:
            planned = self.flow_matching_epochs if self.flow_matching_epochs is not None else self.max_epochs
            return (
                f"flow matching stopped early after epoch {self.stopped_after_epoch} of {planned}{cause}; "
                f"{best}; then distilled for {self.distillation_epochs} epochs"
            )
        return f"stopped early after epoch {self.stopped_after_epoch} of {self.max_epochs}{cause}; {best}"


def read_early_stopping_summary(model_dir: Path | str) -> EarlyStoppingSummary | None:
    """Read a finished model's early-stopping summary.

    Returns:
        The summary, or None when the run had early stopping off or the file is
        missing or unreadable (logged; a summary is informational only).
    """
    path = Path(model_dir) / EARLY_STOPPING_SUMMARY_NAME
    if not path.is_file():
        return None
    try:
        return EarlyStoppingSummary.model_validate_json(path.read_text())
    except (OSError, ValueError):
        logger.warning("Ignoring unreadable early-stopping summary %s", path, exc_info=True)
        return None


def completion_message(model_dir: Path | str) -> str:
    """Status line for a successfully finished training job.

    Returns:
        ``"Training finished"``, extended with what early stopping did when it
        ended the run (or its flow-matching phase) early.
    """
    summary = read_early_stopping_summary(model_dir)
    detail = summary.message if summary is not None else None
    return f"Training finished: {detail}" if detail else "Training finished"


def explain_trainer_rejection(error_text: str | None) -> str | None:
    """Recognize an older trainer rejecting the early-stopping fields of a spec.

    A trainer that predates early stopping validates the spec with
    ``extra="forbid"`` and fails with a generic pydantic error naming the
    field. That only happens when early stopping is on, since the fields are
    not serialized otherwise.

    Args:
        error_text: The trainer's failure message or HTTP error body.

    Returns:
        :data:`EARLY_STOPPING_UNSUPPORTED_MESSAGE` for that rejection, else None.
    """
    if not error_text or "early_stopping_" not in error_text:
        return None
    if "extra_forbidden" in error_text or "Extra inputs are not permitted" in error_text:
        return EARLY_STOPPING_UNSUPPORTED_MESSAGE
    return None


def build_policy(spec: TrainingJobSpec, *, resume_from: Path | str | None = None) -> Policy:
    """Construct the policy a spec describes.

    Args:
        spec: The training configuration.
        resume_from: Checkpoint to continue training from. When None the policy
            is initialized fresh (from its pretrained base weights where the
            policy requires them, see :data:`PRETRAINED_BASE_CHECKPOINTS`).

    Returns:
        The policy, compiled when ``spec.compile_model`` is set.
    """
    if resume_from is not None:
        if spec.image_key_reorder_map or spec.num_cameras > 0:
            logger.warning("Ignoring camera layout: a resumed policy keeps the layout stored in its checkpoint")
        return _load_policy_from_checkpoint(spec, Path(resume_from))

    from physicalai.policies import get_policy

    kwargs: dict[str, Any] = {"compile_model": spec.compile_model}
    if spec.policy_source == "physicalai":
        pretrained = PRETRAINED_BASE_CHECKPOINTS.get(spec.policy.lower())
        if pretrained is not None:
            kwargs["pretrained_name_or_path"] = pretrained
    kwargs.update(_camera_layout_kwargs(spec))
    if spec.lora_enabled:
        kwargs.update(
            lora_enabled=True,
            lora_rank=spec.lora_rank,
            lora_alpha=spec.lora_alpha,
            lora_dropout=spec.lora_dropout,
            lora_use_dora=spec.lora_use_dora,
        )
    return get_policy(spec.policy, source=spec.policy_source, **kwargs)


_CAMERA_LAYOUT_PARAMS = ("image_key_reorder_map", "num_cameras")


def _camera_layout_kwargs(spec: TrainingJobSpec) -> dict[str, Any]:
    """Camera-slot arguments for a policy that reads its cameras in a fixed order.

    Only some policies take these (SmolVLA does; ACT has no notion of a camera
    slot), so they are passed only where the constructor accepts them — a spec
    carrying a mapping for a policy that ignores camera order trains without it
    rather than failing on an unexpected keyword.

    Returns:
        The camera kwargs to construct the policy with, empty when the spec asks
        for no particular layout or the policy has no such parameters.
    """
    if not spec.image_key_reorder_map and spec.num_cameras <= 0:
        return {}

    if spec.policy_source != "physicalai":
        logger.warning("Ignoring camera layout: %s policies are not configured through it", spec.policy_source)
        return {}

    import inspect

    from physicalai.policies import get_physicalai_policy_class

    parameters = inspect.signature(get_physicalai_policy_class(spec.policy).__init__).parameters
    if any(name not in parameters for name in _CAMERA_LAYOUT_PARAMS):
        logger.warning("Ignoring camera layout: policy '%s' does not read its cameras in a fixed order", spec.policy)
        return {}

    return {"image_key_reorder_map": spec.image_key_reorder_map, "num_cameras": spec.num_cameras}


def _has_depth_features(dataset_root: Path) -> bool:
    """Whether the LeRobot dataset at ``dataset_root`` has depth-map features (``info.is_depth_map``)."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return False
    features = json.loads(info_path.read_text()).get("features", {})
    return any(
        (feature.get("info") or {}).get("is_depth_map") or (feature.get("info") or {}).get("video.is_depth_map")
        for feature in features.values()
    )


@contextlib.contextmanager
def _hf_token_env(token: SecretStr | None):
    """Scope ``HF_TOKEN`` to one training run, restoring whatever was there before.

    ``os.environ`` is process-global; parallel trainer jobs run in separate
    processes (see `trainer.queue_worker`) so their tokens cannot leak across
    jobs. Restore an operator's existing value after each run. A no-op when
    ``token`` is ``None``.
    """
    if token is None:
        yield
        return
    previous = os.environ.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = token.get_secret_value()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = previous


def run_training_job(
    spec: TrainingJobSpec,
    *,
    dataset_root: Path | str,
    output_dir: Path | str,
    cache_dir: Path | str,
    report: ReportFn,
    should_stop: StopFn,
) -> None:
    """Train one policy end to end: fit, checkpoint, and export.

    On success ``output_dir`` holds a checkpoint (:data:`CHECKPOINT_NAME`),
    the Lightning CSV logs, and an :data:`EXPORTS_DIRNAME` directory per
    successful export backend. That checkpoint is normally the best epoch,
    written by the ``ModelCheckpoint`` callback during ``fit``; this function
    only saves a checkpoint of its own as a fallback when the callback did
    not produce one (e.g. the monitored metric never logged), so it never
    overwrites a best checkpoint with the final-epoch weights (see
    :func:`_ensure_checkpoint_exists`).

    When ``spec.snapflow_start_epoch`` is set the run has two phases: standard
    flow matching, then SnapFlow self-distillation. Both phases keep their own
    checkpoint and export reloads whichever one :func:`resolve_checkpoint`
    picks, so the exported artifact (and therefore inference) always matches
    the checkpoint a distilled model ships.

    When ``spec.early_stopping_patience`` is set, training ends once
    ``val/loss`` has not improved for that many epochs, and
    :data:`EARLY_STOPPING_SUMMARY_NAME` records what happened. The published
    checkpoint is still the best epoch, as with any run. On a SnapFlow run an
    early stop ends only the flow-matching phase; distillation then starts at
    the next epoch and runs its full budget.

    Cancellation is cooperative. ``should_stop`` is polled throughout training
    via :class:`~physicalai.train.callbacks.ProgressReportingCallback`; when it
    returns True this function stops after the fit loop unwinds and returns
    *without* writing a checkpoint or exporting, since a partially trained run
    has no artifact worth keeping. Callers distinguish a canceled run from a
    completed one by consulting ``should_stop`` again after this returns.

    Before export, the trainer, datamodule, and their references back into the
    policy are dropped and accelerator memory is released (see
    :func:`_detach_trainer` and :func:`_release_memory`): the trainer's
    optimizer state and dataloaders would otherwise stay resident throughout
    export, and export conversion (OpenVINO in particular) can need several GB
    on top of that.

    Args:
        spec: What to train.
        dataset_root: Local root of the LeRobot dataset to train on.
        output_dir: Destination for the trained model. Replaced if it exists.
        cache_dir: Scratch directory for checkpoints and logs during training;
            moved to ``output_dir`` on success.
        report: Telemetry sink, called with ``(progress, message, extra_info)``.
        should_stop: Cooperative cancellation probe.
    """
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from physicalai.data import LeRobotDataModule
    from physicalai.train.callbacks import ProgressReportingCallback
    from physicalai.train.trainer import Trainer
    from physicalai.transforms import DefaultImageAugmentations

    from training.device import resolve_accelerator, resolve_devices, resolve_strategy

    with _hf_token_env(spec.run_options.hf_token):
        accelerator = resolve_accelerator(spec.device_type)
        output_dir, cache_dir = Path(output_dir), Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # No policy Studio trains takes depth maps: train on the color cameras of an RGB-D dataset.
        # Passed only when there is depth, so RGB datasets also train with an older library.
        depth_options: dict[str, Any] = {"exclude_depth": True} if _has_depth_features(Path(dataset_root)) else {}
        datamodule = LeRobotDataModule(
            repo_id=_DATASET_REPO_ID,
            root=str(dataset_root),
            train_batch_size=spec.batch_size,
            num_workers=spec.num_workers,
            val_split=spec.val_split,
            # Applied to the train split only; the datamodule leaves validation
            # images untouched so eval loss stays comparable across runs.
            image_transforms=DefaultImageAugmentations() if spec.augment_images else None,
            **depth_options,
        )
        policy = build_policy(spec, resume_from=spec.run_options.resume_from)

        checkpoint_callback = ModelCheckpoint(
            dirpath=cache_dir,
            filename=Path(CHECKPOINT_NAME).stem,
            save_top_k=1,
            monitor="val/loss",
            mode="min",
        )
        callbacks: list[Any] = [checkpoint_callback, ProgressReportingCallback(report=report, should_stop=should_stop)]
        snapflow_callback = _build_snapflow_callback(spec)
        if snapflow_callback is not None:
            callbacks.append(snapflow_callback)
        early_stopping_callback = _build_early_stopping_callback(spec, snapflow_callback)
        if early_stopping_callback is not None:
            callbacks.append(early_stopping_callback)

        trainer = Trainer(
            logger=CSVLogger(cache_dir.parent, name=cache_dir.stem),
            callbacks=callbacks,
            accelerator=accelerator,
            strategy=resolve_strategy(spec.device_type),
            devices=resolve_devices(spec.device_index),
            max_epochs=spec.max_epochs,
            auto_scale_batch_size=spec.auto_scale_batch_size,
            precision=spec.precision,
            check_val_every_n_epoch=1,
        )

        report(0, "Training model", {})
        trainer.fit(model=policy, datamodule=datamodule)
        if should_stop():
            logger.info("Training canceled; skipping checkpoint and export")
            return

        if early_stopping_callback is not None:
            _record_early_stopping(early_stopping_callback, trainer, cache_dir, report)
        _finalize_checkpoints(trainer, cache_dir, snapflow_callback=snapflow_callback)
        _publish(cache_dir, output_dir)

        _detach_trainer(policy, trainer)
        del trainer, datamodule, policy
        _release_memory()
        export_policy = _export_policy(spec, output_dir)
        _export(export_policy, output_dir, report, spec.export_backends)


def _build_snapflow_callback(spec: TrainingJobSpec) -> Any:
    """Build the SnapFlow phase-transition callback a spec asks for.

    Returns:
        A :class:`~physicalai.train.callbacks.SnapFlowPhaseCallback` when the
        spec sets a boundary, or ``None`` for an ordinary flow-matching run.
    """
    if spec.snapflow_start_epoch is None:
        return None

    from physicalai.train.callbacks import SnapFlowPhaseCallback

    logger.info("SnapFlow distillation starts at epoch %d of %d", spec.snapflow_start_epoch, spec.max_epochs)
    return SnapFlowPhaseCallback(
        start_epoch=spec.snapflow_start_epoch,
        checkpoint_prefix=SNAPFLOW_CHECKPOINT_PREFIX,
    )


def _build_early_stopping_callback(spec: TrainingJobSpec, snapflow_callback: Any) -> Any:
    """Build the early-stopping callback a spec asks for.

    Args:
        spec: The training configuration.
        snapflow_callback: This run's ``SnapFlowPhaseCallback``, or None. When
            given, early stopping watches the flow-matching phase only (see
            :class:`training.early_stopping.ValLossEarlyStopping`).

    Returns:
        A :class:`~training.early_stopping.ValLossEarlyStopping`, or None when
        early stopping is off.
    """
    if spec.early_stopping_patience is None:
        return None

    from training.early_stopping import ValLossEarlyStopping

    logger.info(
        "Early stopping on val/loss: patience %d, min_delta %g%s",
        spec.early_stopping_patience,
        spec.early_stopping_min_delta,
        " (flow-matching phase only)" if snapflow_callback is not None else "",
    )
    return ValLossEarlyStopping(
        patience=spec.early_stopping_patience,
        min_delta=spec.early_stopping_min_delta,
        snapflow_callback=snapflow_callback,
    )


def _record_early_stopping(callback: Any, trainer: Any, cache_dir: Path, report: ReportFn) -> None:
    """Write the early-stopping summary into the model directory and report an early stop.

    Best-effort: the summary is informational, so a failure to build or write
    it is logged and never fails a run that trained successfully.
    """
    try:
        summary: EarlyStoppingSummary = callback.summary(trainer)
        (cache_dir / EARLY_STOPPING_SUMMARY_NAME).write_text(summary.model_dump_json(indent=2))
    except Exception:
        logger.warning("Could not record the early-stopping summary", exc_info=True)
        return
    if summary.message is not None:
        logger.info("Early stopping: %s", summary.message)
        # Training is over; what is left (checkpoint, export) reports from 99 on.
        report(99, summary.message[0].upper() + summary.message[1:], {})


def _load_policy_from_checkpoint(spec: TrainingJobSpec, checkpoint: Path) -> Policy:
    """Restore a policy from a checkpoint, applying ``spec.compile_model``.

    Returns:
        The restored policy.
    """
    if spec.policy_source == "lerobot":
        from physicalai.policies.lerobot import LeRobotPolicy

        policy: Policy = LeRobotPolicy.load_from_checkpoint(checkpoint)
    else:
        from physicalai.policies import get_physicalai_policy_class

        policy_class = get_physicalai_policy_class(spec.policy)
        policy = policy_class.load_from_checkpoint(str(checkpoint))

    if spec.compile_model:
        import torch

        compile_mode = getattr(policy.config, "compile_mode", "default")
        if spec.policy.lower() == "pi05" and compile_mode == "max-autotune":
            compile_mode = "default"
        policy.forward = torch.compile(policy.forward, mode=compile_mode)  # type: ignore[method-assign]
    return policy


def _ensure_checkpoint_exists(trainer: Any, cache_dir: Path) -> None:
    """Save a final checkpoint only if the ``ModelCheckpoint`` callback did not already write one.

    The callback is the preferred source of :data:`CHECKPOINT_NAME`: it tracks
    the best monitored epoch, not just the last one. But it only saves when
    its monitored metric was actually logged (e.g. a run with no validation
    split never logs ``val/loss``), so this is a fallback for that case —
    saving the trainer's current (final-epoch) weights — rather than a second
    source of truth that could overwrite the callback's best checkpoint.
    """
    checkpoint = cache_dir / CHECKPOINT_NAME
    if not checkpoint.is_file():
        logger.warning("ModelCheckpoint callback did not write a checkpoint; saving final weights instead")
        trainer.save_checkpoint(checkpoint)


def _finalize_checkpoints(trainer: Any, cache_dir: Path, *, snapflow_callback: Any) -> None:
    """Make sure every checkpoint this run should ship actually exists on disk.

    :data:`CHECKPOINT_NAME` is always ensured (see
    :func:`_ensure_checkpoint_exists`). When the run reached the SnapFlow
    phase, :data:`SNAPFLOW_CHECKPOINT_NAME` is ensured too: normally
    ``SnapFlowPhaseCallback`` already wrote it, but a run with nothing for the
    monitored metric to log in phase 2 (e.g. no validation split) would
    otherwise ship no distilled checkpoint at all despite having trained one.

    Args:
        trainer: The Lightning trainer that just finished fitting.
        cache_dir: Directory holding this run's checkpoints.
        snapflow_callback: The ``SnapFlowPhaseCallback`` for this run, or
            ``None`` if the run was ordinary flow matching.
    """
    _ensure_checkpoint_exists(trainer, cache_dir)

    if snapflow_callback is None or not snapflow_callback.state_dict().get("activated"):
        return

    snapflow_checkpoint = cache_dir / SNAPFLOW_CHECKPOINT_NAME
    if not snapflow_checkpoint.is_file():
        logger.warning("SnapFlow phase saved no checkpoint; saving final distilled weights instead")
        trainer.save_checkpoint(snapflow_checkpoint)


def _publish(cache_dir: Path, output_dir: Path) -> None:
    """Move the finished training cache into its final location."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.move(str(cache_dir), str(output_dir))


def resolve_checkpoint(model_dir: Path | str) -> Path:
    """Return the checkpoint that represents a model, preferring the distilled one.

    Args:
        model_dir: A model's directory (``output_dir`` from a finished run).

    Returns:
        ``model_dir / SNAPFLOW_CHECKPOINT_NAME`` if it exists, else
        ``model_dir / CHECKPOINT_NAME`` unconditionally (its own existence is
        not checked; callers that need to know should check themselves).
    """
    model_dir = Path(model_dir)
    snapflow_checkpoint = model_dir / SNAPFLOW_CHECKPOINT_NAME
    if snapflow_checkpoint.is_file():
        return snapflow_checkpoint
    return model_dir / CHECKPOINT_NAME


def _export_policy(spec: TrainingJobSpec, output_dir: Path) -> Policy:
    """Return the selected checkpoint reloaded for export.

    Export must match ``resolve_checkpoint`` (and therefore resume/download),
    rather than the final-epoch weights left in memory.  This matters for every
    policy: an ordinary ``ModelCheckpoint`` selects the best validation epoch,
    while the live policy contains the final epoch.  For SnapFlow,
    ``resolve_checkpoint`` additionally selects the distilled checkpoint.

    Returns:
        The policy instance to hand to the export backends.
    """
    resolved = resolve_checkpoint(output_dir)
    logger.info("Reloading selected checkpoint %s for export", resolved.name)
    uncompiled = spec.model_copy(update={"compile_model": False})
    return _load_policy_from_checkpoint(uncompiled, resolved)


def _detach_trainer(policy: Policy, trainer: Any) -> None:
    """Break the trainer<->policy<->datamodule reference cycle before export.

    Lightning wires ``policy._trainer = trainer``, ``trainer.datamodule =
    datamodule``, and ``trainer.strategy._lightning_module = policy`` during
    ``fit``, and never undoes it. ``policy`` is the object that was just trained,
    so it holds that ``_trainer`` reference, which keeps the trainer, and
    everything it holds (optimizer state, dataloaders, the strategy), alive
    and reachable no matter how many local names ``run_training_job`` deletes.
    ``gc.collect()`` only reclaims *unreachable* cycles, so without this the
    memory release below is a no-op on the object it matters most for.

    Best-effort: a failure here must not abort the job.
    """
    try:
        policy._trainer = None
        if getattr(trainer, "strategy", None) is not None:
            trainer.strategy._lightning_module = None
        trainer.datamodule = None
    except Exception:
        logger.warning("Failed to detach trainer before export; continuing anyway", exc_info=True)


def _release_memory() -> None:
    """Return held host and device memory to the OS/allocator before the next stage.

    Export conversion (OpenVINO in particular) allocates several GB above the
    model's resident size, on top of whatever the just-finished trainer and
    dataloaders still hold. A garbage-collection pass plus clearing every
    available accelerator's cache prevents that peak from stacking on top of
    memory this process no longer needs. Best-effort: a failure here must not
    abort the job.
    """
    gc.collect()
    try:
        import torch

        if torch.xpu.is_available():
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if torch.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("Could not release device cache: %s", exc)


def _backend_name(backend: Any) -> str:
    """Return the wire name of an export backend, which may be an enum or a string."""
    return backend.value if hasattr(backend, "value") else str(backend)


def _export(policy: Policy, output_dir: Path, report: ReportFn, requested: list[str] | None = None) -> None:
    """Export the policy to the requested backends, or to every one it supports.

    A requested backend the policy cannot produce is skipped rather than
    treated as an error: the caller asks for formats it wants, and which of
    them a policy can trace is the policy's to say.
    """
    from physicalai.export import ExportablePolicyMixin

    if not isinstance(policy, ExportablePolicyMixin):
        logger.info("Skipping export: policy does not support export backends")
        return

    supported = policy.get_supported_export_backends()
    backends = supported if requested is None else [b for b in supported if _backend_name(b) in requested]

    if requested is not None:
        unsupported = sorted(set(requested) - {_backend_name(b) for b in supported})
        if unsupported:
            logger.warning("Skipping unsupported export backends: %s", ", ".join(unsupported))
    if not backends:
        logger.info("Skipping export: no requested backend is supported by this policy")
        return

    for backend in backends:
        name = _backend_name(backend)
        try:
            logger.info("Exporting model to %s format", name)
            report(99, f"Exporting to {name} format", {})
            policy.export(output_dir / EXPORTS_DIRNAME / name, backend=backend)
        except ImportError as exc:
            # An optional backend dependency is missing (e.g. executorch on xpu
            # builds). Skip it without a traceback so the run isn't mistaken
            # for a failure; the remaining backends still export.
            logger.warning("Skipping %s export: optional dependency missing (%s)", name, exc)
        except Exception:
            # Export is best-effort: one failing backend must not abort the job.
            logger.exception("Failed exporting model to %s format", name)
        finally:
            # Conversion peaks well above the model's resident size, so release
            # after every backend — not just before the first one.
            _release_memory()
