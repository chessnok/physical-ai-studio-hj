# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Early stopping on ``val/loss`` for a Studio training run.

Imports Lightning at module level, so :mod:`training.job` imports this module
only for a run that asks for early stopping.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks import EarlyStopping

from training.job import EarlyStoppingSummary

if TYPE_CHECKING:
    import lightning.pytorch as pl

logger = logging.getLogger(__name__)

MONITOR = "val/loss"
"""The metric early stopping watches: the same one ``ModelCheckpoint`` ranks
epochs by, so the epoch it stops after is judged by what gets published."""


class ValLossEarlyStopping(EarlyStopping):
    """Lightning's ``EarlyStopping`` on ``val/loss``, aware of SnapFlow and able to report what it did.

    On top of the stock callback it remembers the epoch with the lowest
    ``val/loss`` (the epoch ``ModelCheckpoint`` publishes) for
    :meth:`summary`.

    On a SnapFlow run it watches the flow-matching phase only:

    - ``val/loss`` is not comparable across the phase boundary (which is also
      why ``SnapFlowPhaseCallback`` resets ``ModelCheckpoint``'s best there), so
      no check runs during distillation.
    - Stopping during flow matching would skip distillation altogether. When
      patience runs out in that phase, the stop is withdrawn and the phase
      boundary moves to the next epoch instead: distillation starts from the
      best flow-matching checkpoint (``SnapFlowPhaseCallback`` restores it) and
      still runs its full number of epochs, and the trainer's epoch budget
      shrinks by the flow-matching epochs that were skipped.

    A run whose validation never logs ``val/loss`` trains to the end instead
    of failing (``strict=False``); Lightning warns about it.

    Args:
        patience: Validation epochs without improvement before stopping.
        min_delta: Smallest ``val/loss`` decrease that counts as an improvement.
        snapflow_callback: The run's ``SnapFlowPhaseCallback``, or None.
    """

    def __init__(self, *, patience: int, min_delta: float = 0.0, snapflow_callback: Any = None) -> None:
        super().__init__(
            monitor=MONITOR,
            mode="min",
            patience=patience,
            min_delta=min_delta,
            strict=False,
            verbose=True,
        )
        self._requested_min_delta = min_delta
        self._snapflow = snapflow_callback
        self._planned_max_epochs: int | None = None
        self._planned_flow_matching_epochs: int | None = (
            snapflow_callback.start_epoch if snapflow_callback is not None else None
        )
        self.best_epoch: int | None = None
        """Zero-based epoch with the lowest finite ``val/loss`` (in the watched phase)."""
        self.best_val_loss: float | None = None
        self.stopped_after_epoch: int | None = None
        """One-based epoch after which early stopping ended the run (or the flow-matching phase)."""
        self.stop_reason: str | None = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Remember the planned epoch budget before a SnapFlow hand-over can shrink it."""
        super().setup(trainer, pl_module, stage)
        if self._planned_max_epochs is None:
            self._planned_max_epochs = trainer.max_epochs

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Track the best epoch, then let Lightning check (when it checks after validation)."""
        if self._is_distilling(trainer):
            return
        if not trainer.sanity_checking:
            self._track_best(trainer)
        stop_was_requested = trainer.should_stop
        super().on_validation_end(trainer, pl_module)
        self._after_check(trainer, stop_was_requested)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Let Lightning check (when it checks at epoch end, Studio's case)."""
        if self._is_distilling(trainer):
            return
        stop_was_requested = trainer.should_stop
        super().on_train_epoch_end(trainer, pl_module)
        self._after_check(trainer, stop_was_requested)

    def _is_distilling(self, trainer: pl.Trainer) -> bool:
        return self._snapflow is not None and trainer.current_epoch >= self._snapflow.start_epoch

    def _track_best(self, trainer: pl.Trainer) -> None:
        value = trainer.callback_metrics.get(MONITOR)
        if value is None:
            return
        loss = float(value)
        if math.isfinite(loss) and (self.best_val_loss is None or loss < self.best_val_loss):
            self.best_val_loss = loss
            self.best_epoch = trainer.current_epoch

    def _after_check(self, trainer: pl.Trainer, stop_was_requested: bool) -> None:
        """Record a stop this callback just decided; on a SnapFlow run, turn it into a phase change.

        Args:
            trainer: The active trainer.
            stop_was_requested: ``trainer.should_stop`` before the check. A stop
                requested by someone else (cancellation) is left alone.
        """
        if stop_was_requested or not trainer.should_stop:
            return
        reason = getattr(self, "stopping_reason", None)
        reason_name = reason.name.lower() if reason is not None and hasattr(reason, "name") else None
        epochs_done = trainer.current_epoch + 1

        if self._snapflow is None:
            self.stopped_after_epoch = epochs_done
            self.stop_reason = reason_name
            return

        # SnapFlow: end the flow-matching phase, not the run.
        trainer.should_stop = False
        boundary = self._snapflow.start_epoch
        if epochs_done >= boundary:
            return  # Flow matching ends after this epoch anyway.
        distillation_epochs = (trainer.max_epochs or boundary) - boundary
        self._snapflow.start_epoch = epochs_done
        trainer.fit_loop.max_epochs = epochs_done + distillation_epochs
        self.stopped_after_epoch = epochs_done
        self.stop_reason = reason_name
        logger.info(
            "Early stopping ended flow matching after epoch %d of %d; distilling for %d epochs from epoch %d",
            epochs_done,
            boundary,
            distillation_epochs,
            epochs_done + 1,
        )

    def summary(self, trainer: pl.Trainer) -> EarlyStoppingSummary:
        """Describe what early stopping did, for :data:`training.job.EARLY_STOPPING_SUMMARY_NAME`.

        Args:
            trainer: The trainer, after ``fit`` returned.

        Returns:
            The summary, with 1-based epoch numbers.
        """
        max_epochs = self._planned_max_epochs if self._planned_max_epochs is not None else (trainer.max_epochs or 0)
        flow_matching_epochs = self._planned_flow_matching_epochs
        return EarlyStoppingSummary(
            patience=self.patience,
            min_delta=self._requested_min_delta,
            max_epochs=max_epochs,
            epochs_trained=trainer.current_epoch,
            stopped_after_epoch=self.stopped_after_epoch,
            reason=self.stop_reason,
            best_epoch=self.best_epoch + 1 if self.best_epoch is not None else None,
            best_val_loss=self.best_val_loss,
            flow_matching_epochs=flow_matching_epochs,
            distillation_epochs=max_epochs - flow_matching_epochs if flow_matching_epochs is not None else None,
        )
