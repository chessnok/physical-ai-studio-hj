# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Early stopping in real (tiny) Lightning runs.

The model's ``val/loss`` is scripted per epoch, so a plateau, a too-small
improvement, or a rise during SnapFlow distillation happens exactly where a
test puts it. Everything else is the real thing: Lightning's fit loop, its
``ModelCheckpoint``, the library's ``ProgressReportingCallback`` and
``SnapFlowPhaseCallback``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from physicalai.train.callbacks import ProgressReportingCallback, SnapFlowPhaseCallback
from torch.utils.data import DataLoader, TensorDataset

from training.early_stopping import ValLossEarlyStopping
from training.job import EarlyStoppingSummary

if TYPE_CHECKING:
    from pathlib import Path


class _ScriptedValLoss(pl.LightningModule):
    """Trains a real (tiny) layer but reports a scripted val/loss for each epoch."""

    def __init__(self, val_losses: list[float]) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)
        self.val_losses = val_losses

    def training_step(self, batch: list[torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self.layer(batch[0]).pow(2).mean()

    def validation_step(self, _batch: list[torch.Tensor], _batch_idx: int) -> None:
        loss = self.val_losses[min(self.current_epoch, len(self.val_losses) - 1)]
        self.log("val/loss", torch.tensor(loss))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(TensorDataset(torch.randn(8, 2)), batch_size=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(TensorDataset(torch.randn(4, 2)), batch_size=4)


class _DistillableScriptedValLoss(_ScriptedValLoss):
    """Accepts ``enable_snapflow`` like the flow-matching policies, recording when it happened."""

    def __init__(self, val_losses: list[float]) -> None:
        super().__init__(val_losses)
        self.distillation_started_at: int | None = None

    def enable_snapflow(self, *, alpha: float, lambda_: float, num_inference_steps: int) -> None:
        self.distillation_started_at = self.current_epoch


def _fit(
    model: pl.LightningModule,
    tmp_path: Path,
    *,
    max_epochs: int,
    patience: int,
    min_delta: float = 0.0,
    snapflow_start_epoch: int | None = None,
    report: list[int] | None = None,
    stop_after_steps: int | None = None,
) -> tuple[pl.Trainer, ValLossEarlyStopping]:
    """Fit ``model`` with the callbacks ``run_training_job`` uses."""
    checkpoint = ModelCheckpoint(dirpath=tmp_path, filename="model", save_top_k=1, monitor="val/loss", mode="min")
    progress: list[int] = report if report is not None else []
    trainer_ref: list[pl.Trainer] = []

    def should_stop() -> bool:
        return stop_after_steps is not None and trainer_ref[0].global_step >= stop_after_steps

    callbacks: list[pl.Callback] = [
        checkpoint,
        ProgressReportingCallback(
            report=lambda value, _message, _extra: progress.append(value), should_stop=should_stop
        ),
    ]
    snapflow = None
    if snapflow_start_epoch is not None:
        snapflow = SnapFlowPhaseCallback(start_epoch=snapflow_start_epoch, checkpoint_prefix="snapflow-")
        callbacks.append(snapflow)
    early_stopping = ValLossEarlyStopping(patience=patience, min_delta=min_delta, snapflow_callback=snapflow)
    callbacks.append(early_stopping)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        default_root_dir=tmp_path,
    )
    trainer_ref.append(trainer)
    trainer.fit(model)
    return trainer, early_stopping


class TestPlainRun:
    def test_a_plateau_stops_the_run_after_patience_epochs(self, tmp_path: Path) -> None:
        # Best at epoch 3 (0.7); epochs 4 and 5 do not beat it.
        model = _ScriptedValLoss([1.0, 0.8, 0.7, 0.7, 0.75, 0.6, 0.5])

        trainer, callback = _fit(model, tmp_path, max_epochs=20, patience=2)

        assert trainer.current_epoch == 5
        summary = callback.summary(trainer)
        assert isinstance(summary, EarlyStoppingSummary)
        assert summary.model_dump() == {
            "patience": 2,
            "min_delta": 0.0,
            "max_epochs": 20,
            "epochs_trained": 5,
            "stopped_after_epoch": 5,
            "reason": "patience_exhausted",
            "best_epoch": 3,
            "best_val_loss": pytest.approx(0.7),
            "flow_matching_epochs": None,
            "distillation_epochs": None,
        }
        assert summary.message == "stopped early after epoch 5 of 20; best epoch 3 (val/loss 0.7)"

    def test_the_best_epoch_is_the_published_checkpoint(self, tmp_path: Path) -> None:
        """The summary names the epoch ModelCheckpoint kept, which is what gets exported."""
        model = _ScriptedValLoss([1.0, 0.8, 0.7, 0.7, 0.75])

        trainer, callback = _fit(model, tmp_path, max_epochs=20, patience=2)

        checkpoint = torch.load(tmp_path / "model.ckpt", map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] + 1 == callback.summary(trainer).best_epoch == 3

    def test_improvements_below_min_delta_do_not_count(self, tmp_path: Path) -> None:
        """Each epoch is a bit better, but never by 0.1, so patience runs out after 2 epochs."""
        model = _ScriptedValLoss([1.0, 0.95, 0.92, 0.9, 0.85])

        trainer, callback = _fit(model, tmp_path, max_epochs=20, patience=2, min_delta=0.1)

        summary = callback.summary(trainer)
        assert (summary.stopped_after_epoch, trainer.current_epoch) == (3, 3)
        # The lowest val/loss still names the best epoch (it is the one checkpointed).
        assert (summary.best_epoch, summary.best_val_loss) == (3, pytest.approx(0.92))
        assert summary.min_delta == 0.1

    def test_a_run_that_keeps_improving_trains_every_epoch(self, tmp_path: Path) -> None:
        model = _ScriptedValLoss([1.0, 0.9, 0.8, 0.7])

        trainer, callback = _fit(model, tmp_path, max_epochs=4, patience=2)

        summary = callback.summary(trainer)
        assert trainer.current_epoch == 4
        assert summary.stopped_early is False
        assert summary.message is None
        assert (summary.epochs_trained, summary.best_epoch) == (4, 4)

    def test_a_non_finite_val_loss_stops_the_run_and_says_why(self, tmp_path: Path) -> None:
        model = _ScriptedValLoss([1.0, 0.8, float("nan")])

        trainer, callback = _fit(model, tmp_path, max_epochs=20, patience=5)

        summary = callback.summary(trainer)
        assert (summary.stopped_after_epoch, summary.reason) == (3, "non_finite_metric")
        assert (summary.best_epoch, summary.best_val_loss) == (2, pytest.approx(0.8))
        assert summary.message is not None
        assert "val/loss was not finite" in summary.message

    def test_progress_stays_in_range_and_never_goes_back(self, tmp_path: Path) -> None:
        """An early stop leaves progress where training stopped; the runner reports what follows."""
        model = _ScriptedValLoss([1.0, 0.8, 0.8, 0.8])
        progress: list[int] = []

        _fit(model, tmp_path, max_epochs=10, patience=2, report=progress)

        assert progress
        assert all(0 <= value <= 100 for value in progress)
        assert progress == sorted(progress)
        # 4 of 10 epochs ran: no claim of completion before the job finishes.
        assert max(progress) < 100

    def test_a_cancellation_is_not_reported_as_an_early_stop(self, tmp_path: Path) -> None:
        model = _ScriptedValLoss([1.0, 0.9, 0.8, 0.7, 0.6])

        trainer, callback = _fit(model, tmp_path, max_epochs=5, patience=1, stop_after_steps=4)

        assert trainer.current_epoch < 5
        assert callback.summary(trainer).stopped_early is False


class TestSnapFlowRun:
    """Early stopping watches flow matching only; distillation always runs its full budget."""

    def test_a_plateau_starts_distillation_early_and_distillation_runs_in_full(self, tmp_path: Path) -> None:
        # Flow matching planned for 8 epochs, then 3 of distillation. Best at epoch 3, so patience 2
        # ends flow matching after epoch 5. Distillation's val/loss rises (not comparable to flow
        # matching's) and must not stop it.
        model = _DistillableScriptedValLoss([1.0, 0.8, 0.7, 0.7, 0.7, 2.0, 3.0, 4.0])
        progress: list[int] = []

        trainer, callback = _fit(model, tmp_path, max_epochs=11, patience=2, snapflow_start_epoch=8, report=progress)

        assert model.distillation_started_at == 5
        assert trainer.current_epoch == 8
        summary = callback.summary(trainer)
        assert summary.model_dump() == {
            "patience": 2,
            "min_delta": 0.0,
            "max_epochs": 11,
            "epochs_trained": 8,
            "stopped_after_epoch": 5,
            "reason": "patience_exhausted",
            "best_epoch": 3,
            "best_val_loss": pytest.approx(0.7),
            "flow_matching_epochs": 8,
            "distillation_epochs": 3,
        }
        assert summary.message == (
            "flow matching stopped early after epoch 5 of 8; best epoch 3 (val/loss 0.7); then distilled for 3 epochs"
        )
        # The distilled checkpoint is written beside the flow-matching one, as in a full run.
        assert (tmp_path / "model.ckpt").is_file()
        assert (tmp_path / "snapflow-model.ckpt").is_file()
        assert all(0 <= value <= 100 for value in progress)
        assert progress == sorted(progress)

    def test_without_a_plateau_distillation_starts_as_planned(self, tmp_path: Path) -> None:
        model = _DistillableScriptedValLoss([1.0, 0.9, 0.8, 5.0, 6.0])

        trainer, callback = _fit(model, tmp_path, max_epochs=5, patience=1, snapflow_start_epoch=3)

        assert model.distillation_started_at == 3
        assert trainer.current_epoch == 5
        assert callback.summary(trainer).stopped_early is False

    def test_a_plateau_reached_in_the_last_flow_matching_epoch_changes_nothing(self, tmp_path: Path) -> None:
        model = _DistillableScriptedValLoss([1.0, 0.9, 0.9, 0.9, 1.0, 1.0])

        trainer, callback = _fit(model, tmp_path, max_epochs=6, patience=2, snapflow_start_epoch=4)

        assert model.distillation_started_at == 4
        assert trainer.current_epoch == 6
        assert callback.summary(trainer).stopped_early is False
