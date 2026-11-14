# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared training job spec and runner.

``run_training_job`` is the single training path used both in-process by the
studio and by the standalone trainer service, so what is asserted here is the
contract both depend on: how a spec becomes a policy and a ``Trainer``, where
artifacts land, and what a canceled run leaves behind. Lightning's own fit loop
is mocked out; it is not under test.
"""

from __future__ import annotations

import gc
import inspect
import os
import weakref
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from physicalai.export import ExportablePolicyMixin, ExportBackend
from pydantic import SecretStr, ValidationError
from tests.old_trainer import EARLY_STOPPING_FIELDS, old_trainer_rejection

from api.policies import _available_policy_classes
from training import RunOptions, TrainingJobSpec
from training.job import (
    CHECKPOINT_NAME,
    EARLY_STOPPING_SUMMARY_NAME,
    EARLY_STOPPING_UNSUPPORTED_MESSAGE,
    EXPORTS_DIRNAME,
    PEFT_POLICIES,
    PRETRAINED_BASE_CHECKPOINTS,
    SNAPFLOW_CHECKPOINT_NAME,
    EarlyStoppingSummary,
    _load_policy_from_checkpoint,
    build_policy,
    completion_message,
    explain_trainer_rejection,
    read_early_stopping_summary,
    resolve_checkpoint,
    run_training_job,
)

JOB = "training.job"


class TestTrainingJobSpec:
    """The spec is a wire format as well as a local config, so both matter."""

    def test_only_the_policy_is_required(self) -> None:
        spec = TrainingJobSpec(policy="act")

        assert (spec.policy_source, spec.max_epochs, spec.batch_size) == ("physicalai", 5, 8)
        assert (spec.num_workers, spec.val_split, spec.precision) == ("auto", 0.1, "bf16-mixed")
        assert (spec.compile_model, spec.auto_scale_batch_size) == (False, False)
        assert spec.augment_images is False
        assert (spec.device_type, spec.device_index) == (None, None)

    def test_unknown_field_is_rejected(self) -> None:
        """A stray field over the wire is a version mismatch, not a value to drop."""
        with pytest.raises(ValidationError):
            TrainingJobSpec(policy="act", learning_rate=0.1)

    @pytest.mark.parametrize(
        "invalid",
        [
            {"max_epochs": 0},
            {"batch_size": 0},
            {"val_split": 1.0},
            {"val_split": -0.1},
            {"device_index": -1},
            {"policy_source": "elsewhere"},
        ],
    )
    def test_out_of_range_values_are_rejected(self, invalid: dict) -> None:
        with pytest.raises(ValidationError):
            TrainingJobSpec(policy="act", **invalid)

    def test_spec_round_trips_through_json(self) -> None:
        """Remote submission sends the spec as JSON; it must survive the trip."""
        spec = TrainingJobSpec(policy="pi05", max_epochs=5, num_workers=4, device_type="xpu", device_index=1)

        assert TrainingJobSpec.model_validate_json(spec.model_dump_json()) == spec

    def test_lora_disabled_by_default(self) -> None:
        spec = TrainingJobSpec(policy="act")

        assert spec.lora_enabled is False
        assert (spec.lora_rank, spec.lora_alpha, spec.lora_dropout, spec.lora_use_dora) == (32, None, 0.05, False)

    @pytest.mark.parametrize("policy_name", sorted(PEFT_POLICIES))
    def test_lora_is_accepted_for_peft_capable_policies(self, policy_name: str) -> None:
        spec = TrainingJobSpec(policy=policy_name, lora_enabled=True, lora_rank=16, lora_use_dora=True)

        assert (spec.lora_enabled, spec.lora_rank, spec.lora_use_dora) == (True, 16, True)

    def test_lora_is_rejected_for_a_policy_without_peft_support(self) -> None:
        with pytest.raises(ValidationError):
            TrainingJobSpec(policy="act", lora_enabled=True)

    def test_lora_is_rejected_for_a_lerobot_sourced_policy(self) -> None:
        with pytest.raises(ValidationError):
            TrainingJobSpec(policy="pi05", policy_source="lerobot", lora_enabled=True)

    def test_flow_matching_is_the_default(self) -> None:
        assert TrainingJobSpec(policy="pi05").snapflow_start_epoch is None

    @pytest.mark.parametrize("policy", ["pi05", "smolvla"])
    def test_flow_matching_policies_accept_a_distillation_boundary(self, policy: str) -> None:
        spec = TrainingJobSpec(policy=policy, max_epochs=8, snapflow_start_epoch=5)

        assert spec.snapflow_start_epoch == 5

    @pytest.mark.parametrize("policy", ["act"])
    def test_policies_without_a_flow_matching_sampler_cannot_be_distilled(self, policy: str) -> None:
        """Only the SnapFlowPolicyMixin policies implement enable_snapflow()."""
        with pytest.raises(ValidationError, match="not supported for policy"):
            TrainingJobSpec(policy=policy, max_epochs=8, snapflow_start_epoch=5)

    @pytest.mark.parametrize("start_epoch", [5, 6])
    def test_boundary_must_leave_an_epoch_to_distill(self, start_epoch: int) -> None:
        """A boundary at or past the budget would train a teacher and distil nothing."""
        with pytest.raises(ValidationError, match="must be below max_epochs"):
            TrainingJobSpec(policy="pi05", max_epochs=5, snapflow_start_epoch=start_epoch)

    def test_boundary_must_leave_an_epoch_to_train_the_teacher(self) -> None:
        """Distilling from step zero distils an untrained policy."""
        with pytest.raises(ValidationError):
            TrainingJobSpec(policy="pi05", max_epochs=5, snapflow_start_epoch=0)


class TestEarlyStoppingSpec:
    """Early stopping is opt-in, and a spec without it must still reach a trainer that predates it."""

    def test_early_stopping_is_off_by_default(self) -> None:
        spec = TrainingJobSpec(policy="act")

        assert (spec.early_stopping_patience, spec.early_stopping_min_delta) == (None, 0.0)

    def test_an_ordinary_spec_serializes_without_early_stopping_fields(self) -> None:
        dumped = TrainingJobSpec(policy="act").model_dump(mode="json")

        assert not set(EARLY_STOPPING_FIELDS) & set(dumped)

    def test_an_ordinary_spec_validates_on_a_trainer_without_early_stopping(self) -> None:
        """The Modal app, AWS Batch image, or trainer service may run older code."""
        spec = TrainingJobSpec(policy="pi05", max_epochs=8, snapflow_start_epoch=5, lora_enabled=True)

        assert old_trainer_rejection(spec.model_dump_json(exclude={"run_options"})) is None

    def test_an_early_stopping_spec_is_rejected_by_such_a_trainer_and_explained(self) -> None:
        """Never silently trained without it: the old trainer refuses, and Studio says why."""
        spec = TrainingJobSpec(policy="act", early_stopping_patience=5)

        rejection = old_trainer_rejection(spec.model_dump_json(exclude={"run_options"}))

        assert rejection is not None
        assert explain_trainer_rejection(rejection) == EARLY_STOPPING_UNSUPPORTED_MESSAGE

    @pytest.mark.parametrize(
        "error_text",
        [
            None,
            "",
            "CUDA out of memory",
            "1 validation error for TrainingJobSpec\nlearning_rate\n  Extra inputs are not permitted",
        ],
    )
    def test_other_failures_are_left_alone(self, error_text: str | None) -> None:
        assert explain_trainer_rejection(error_text) is None

    def test_an_early_stopping_spec_round_trips_through_json(self) -> None:
        spec = TrainingJobSpec(policy="act", early_stopping_patience=4, early_stopping_min_delta=0.01)

        dumped = spec.model_dump(mode="json")
        assert (dumped["early_stopping_patience"], dumped["early_stopping_min_delta"]) == (4, 0.01)
        assert TrainingJobSpec.model_validate_json(spec.model_dump_json()) == spec

    def test_a_zero_min_delta_is_not_sent(self) -> None:
        """0 is the default on both ends; leaving it out keeps the wire minimal."""
        dumped = TrainingJobSpec(policy="act", early_stopping_patience=4).model_dump(mode="json")

        assert dumped["early_stopping_patience"] == 4
        assert "early_stopping_min_delta" not in dumped

    def test_min_delta_without_patience_is_dropped(self) -> None:
        """It has no effect without early stopping, so it must not make the spec unsendable to an old trainer."""
        spec = TrainingJobSpec(policy="act", early_stopping_min_delta=0.5)

        assert spec.early_stopping_min_delta == 0.0
        assert old_trainer_rejection(spec.model_dump_json(exclude={"run_options"})) is None

    def test_early_stopping_needs_a_validation_split(self) -> None:
        with pytest.raises(ValidationError, match="needs a validation split"):
            TrainingJobSpec(policy="act", val_split=0.0, early_stopping_patience=3)

    def test_no_validation_split_is_fine_without_early_stopping(self) -> None:
        assert TrainingJobSpec(policy="act", val_split=0.0).early_stopping_patience is None

    @pytest.mark.parametrize("invalid", [{"early_stopping_patience": 0}, {"early_stopping_min_delta": -0.1}])
    def test_out_of_range_early_stopping_values_are_rejected(self, invalid: dict) -> None:
        with pytest.raises(ValidationError):
            TrainingJobSpec(policy="act", **{"early_stopping_patience": 3, **invalid})


class TestEarlyStoppingSummary:
    @staticmethod
    def _write(model_dir: Path, **fields: Any) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        summary = EarlyStoppingSummary(patience=3, min_delta=0.0, max_epochs=30, epochs_trained=12, **fields)
        (model_dir / EARLY_STOPPING_SUMMARY_NAME).write_text(summary.model_dump_json())

    def test_a_model_without_a_summary_simply_finished(self, tmp_path: Path) -> None:
        assert read_early_stopping_summary(tmp_path) is None
        assert completion_message(tmp_path) == "Training finished"

    def test_an_early_stop_is_named_in_the_completion_message(self, tmp_path: Path) -> None:
        self._write(tmp_path, stopped_after_epoch=12, reason="patience_exhausted", best_epoch=9, best_val_loss=0.51943)

        assert completion_message(tmp_path) == (
            "Training finished: stopped early after epoch 12 of 30; best epoch 9 (val/loss 0.5194)"
        )

    def test_a_run_early_stopping_never_ended_simply_finished(self, tmp_path: Path) -> None:
        self._write(tmp_path, best_epoch=30, best_val_loss=0.4)

        assert completion_message(tmp_path) == "Training finished"

    def test_a_snapflow_run_says_distillation_followed(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            stopped_after_epoch=6,
            best_epoch=4,
            best_val_loss=0.3,
            flow_matching_epochs=27,
            distillation_epochs=3,
        )

        assert completion_message(tmp_path) == (
            "Training finished: flow matching stopped early after epoch 6 of 27; best epoch 4 (val/loss 0.3); "
            "then distilled for 3 epochs"
        )

    def test_an_unreadable_summary_does_not_fail_the_job(self, tmp_path: Path) -> None:
        (tmp_path / EARLY_STOPPING_SUMMARY_NAME).write_text("{not json")

        assert completion_message(tmp_path) == "Training finished"


class TestBuildPolicy:
    def test_studio_trainable_policy_constructors_accept_wizard_arguments(self) -> None:
        """Bind Studio's actual kwargs to opt-in constructors without downloading weights."""
        available = _available_policy_classes()
        assert available
        for name, (cls, _) in available.items():
            specs = [
                TrainingJobSpec(policy=name),
                TrainingJobSpec(policy=name, image_key_reorder_map={"camera": 0}, num_cameras=1),
            ]
            if name in PEFT_POLICIES:
                specs.append(TrainingJobSpec(policy=name, lora_enabled=True))
            for spec in specs:
                with patch("physicalai.policies.get_policy") as get_policy:
                    build_policy(spec)

                assert get_policy.call_args.args == (name,)
                kwargs = {key: value for key, value in get_policy.call_args.kwargs.items() if key != "source"}
                try:
                    inspect.signature(cls).bind(**kwargs)
                except TypeError as exc:
                    pytest.fail(f"{name} cannot be constructed by Studio: {exc}")

    def test_fresh_policy_is_built_from_the_spec(self) -> None:
        with patch("physicalai.policies.get_policy") as get_policy:
            policy = build_policy(TrainingJobSpec(policy="act", compile_model=True))

        assert policy is get_policy.return_value
        get_policy.assert_called_once_with("act", source="physicalai", compile_model=True)

    @pytest.mark.parametrize("policy_name", sorted(PRETRAINED_BASE_CHECKPOINTS))
    def test_finetune_only_policies_start_from_pretrained_weights(self, policy_name: str) -> None:
        """These policies have no from-scratch initialization worth training."""
        with patch("physicalai.policies.get_policy") as get_policy:
            build_policy(TrainingJobSpec(policy=policy_name))

        assert get_policy.call_args.kwargs["pretrained_name_or_path"] == PRETRAINED_BASE_CHECKPOINTS[policy_name]

    def test_camera_layout_is_passed_to_a_policy_that_reads_a_fixed_order(self) -> None:
        """SmolVLA takes the dataset's cameras in the slots the mapping names."""
        spec = TrainingJobSpec(
            policy="smolvla",
            image_key_reorder_map={"observation.images.top": 0, "observation.images.wrist": 1},
            num_cameras=3,
        )

        with patch("physicalai.policies.get_policy") as get_policy:
            build_policy(spec)

        assert get_policy.call_args.kwargs["image_key_reorder_map"] == spec.image_key_reorder_map
        assert get_policy.call_args.kwargs["num_cameras"] == 3

    def test_camera_layout_is_dropped_for_a_policy_without_camera_slots(self) -> None:
        """ACT's constructor has no such parameters; passing them would be a TypeError."""
        spec = TrainingJobSpec(policy="act", image_key_reorder_map={"observation.images.top": 0}, num_cameras=2)

        with patch("physicalai.policies.get_policy") as get_policy:
            build_policy(spec)

        assert "image_key_reorder_map" not in get_policy.call_args.kwargs
        assert "num_cameras" not in get_policy.call_args.kwargs

    def test_no_camera_layout_is_passed_when_none_is_asked_for(self) -> None:
        with patch("physicalai.policies.get_policy") as get_policy:
            build_policy(TrainingJobSpec(policy="smolvla"))

        assert "image_key_reorder_map" not in get_policy.call_args.kwargs

    def test_lerobot_policies_are_left_to_lerobots_own_defaults(self) -> None:
        with patch("physicalai.policies.get_policy") as get_policy:
            build_policy(TrainingJobSpec(policy="smolvla", policy_source="lerobot"))

        assert "pretrained_name_or_path" not in get_policy.call_args.kwargs

    def test_lora_disabled_run_gets_no_lora_kwargs(self) -> None:
        with patch("physicalai.policies.get_policy") as get_policy:
            build_policy(TrainingJobSpec(policy="pi05"))

        for key in ("lora_enabled", "lora_rank", "lora_alpha", "lora_dropout", "lora_use_dora"):
            assert key not in get_policy.call_args.kwargs

    def test_lora_enabled_run_passes_lora_kwargs(self) -> None:
        """Learning-rate scaling for LoRA/DoRA is the library's job (PeftConfigMixin.lora_lr_scale),
        not the backend's; the backend only forwards the hyperparameters it owns."""
        with patch("physicalai.policies.get_policy") as get_policy:
            build_policy(
                TrainingJobSpec(
                    policy="pi05", lora_enabled=True, lora_rank=16, lora_alpha=32, lora_dropout=0.1, lora_use_dora=True
                )
            )

        kwargs = get_policy.call_args.kwargs
        assert (kwargs["lora_enabled"], kwargs["lora_rank"], kwargs["lora_alpha"]) == (True, 16, 32)
        assert (kwargs["lora_dropout"], kwargs["lora_use_dora"]) == (0.1, True)
        assert "optimizer_lr" not in kwargs

    def test_resume_loads_the_checkpoint_instead_of_a_new_policy(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / CHECKPOINT_NAME
        policy_class = MagicMock()

        with patch("physicalai.policies.get_physicalai_policy_class", return_value=policy_class):
            policy = build_policy(TrainingJobSpec(policy="act"), resume_from=checkpoint)

        assert policy is policy_class.load_from_checkpoint.return_value
        policy_class.load_from_checkpoint.assert_called_once_with(str(checkpoint))

    def test_lerobot_policies_are_resumed_through_the_wrapper(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / CHECKPOINT_NAME

        with patch("physicalai.policies.lerobot.LeRobotPolicy") as wrapper:
            policy = build_policy(TrainingJobSpec(policy="act", policy_source="lerobot"), resume_from=checkpoint)

        assert policy is wrapper.load_from_checkpoint.return_value
        wrapper.load_from_checkpoint.assert_called_once_with(checkpoint)


class _ExportablePolicy(ExportablePolicyMixin):
    """A policy that records the export calls the runner makes."""

    def __init__(self, backends: list[ExportBackend], *, failing: set[ExportBackend] | None = None) -> None:
        self._backends = backends
        self._failing = failing or set()
        self.exported: list[tuple[str, ExportBackend]] = []

    def get_supported_export_backends(self) -> list[ExportBackend]:  # type: ignore[override]
        return self._backends

    def export(self, output_path, backend, input_sample=None, **export_kwargs) -> None:  # type: ignore[override]
        if backend in self._failing:
            msg = f"{backend} unavailable"
            raise RuntimeError(msg)
        self.exported.append((str(output_path), backend))


def _wire_mocked_fit(trainer: Any, checkpoint: Path, *, model: Any, datamodule: Any) -> None:
    """Simulate the trainer references Lightning leaves attached after fit."""
    model._trainer = trainer
    trainer.datamodule = datamodule
    trainer.strategy._lightning_module = model
    checkpoint.write_text("x")


def _assert_resources_released_before_reload(
    trainer_ref: weakref.ReferenceType[Any],
    policy_ref: list[weakref.ReferenceType[_ExportablePolicy] | None],
    release_memory: MagicMock,
    exported_policy: _ExportablePolicy,
    *_args: Any,
    **_kwargs: Any,
) -> _ExportablePolicy:
    """Ensure checkpoint construction starts after training resources are released."""
    trainer = trainer_ref()
    assert trainer is not None
    assert policy_ref[0] is not None and policy_ref[0]() is None
    assert trainer.datamodule is None
    assert trainer.strategy._lightning_module is None
    release_memory.assert_called_once_with()
    return exported_policy


def _run(
    spec: TrainingJobSpec,
    tmp_path: Path,
    *,
    policy: object | None = None,
    should_stop: bool = False,
    report: MagicMock | None = None,
    write_checkpoint: bool = True,
    reach_snapflow_phase: bool = False,
    write_distilled_checkpoint: bool = True,
    checkpoint_loader: MagicMock | None = None,
) -> MagicMock:
    """Run a job with the datamodule and Lightning trainer mocked out.

    ``Trainer`` is mocked, so its real ``ModelCheckpoint`` callback never runs
    and never writes a checkpoint on its own; ``write_checkpoint`` simulates
    that side effect of a successful ``fit`` so callers don't have to.
    ``trainer.save_checkpoint`` is wired to write the file too, so tests that
    exercise the fallback (``write_checkpoint=False``) still end up with a
    real checkpoint on disk once the runner calls it.

    ``reach_snapflow_phase`` simulates the other side effect a real fit would
    have on a distillation run: ``SnapFlowPhaseCallback`` flips itself to
    activated at the boundary and the prefixed phase-2 checkpoint appears
    beside phase 1's. ``write_distilled_checkpoint`` suppresses the latter, for
    the case where distillation ran but the monitored metric never logged.

    Returns:
        The patched ``Trainer`` class, so tests can assert how it was configured.
    """
    cache_dir = tmp_path / "cache" / "job"

    def fake_fit(*_args: object, **_kwargs: object) -> None:
        if write_checkpoint:
            (cache_dir / CHECKPOINT_NAME).write_text("flow-matching")
        if not reach_snapflow_phase:
            return
        for callback in trainer_class.call_args.kwargs["callbacks"]:
            if type(callback).__name__ == "SnapFlowPhaseCallback":
                callback._activated = True
        if write_distilled_checkpoint:
            (cache_dir / SNAPFLOW_CHECKPOINT_NAME).write_text("distilled")

    def fake_save_checkpoint(path: str | Path, *_args: object, **_kwargs: object) -> None:
        Path(path).write_text("checkpoint")

    loader_patch = (
        nullcontext(checkpoint_loader)
        if checkpoint_loader is not None
        else patch(
            f"{JOB}._load_policy_from_checkpoint",
            return_value=policy if policy is not None else MagicMock(),
        )
    )

    with (
        patch("physicalai.data.LeRobotDataModule"),
        patch(f"{JOB}.build_policy", return_value=policy if policy is not None else MagicMock()),
        loader_patch,
        patch("physicalai.train.trainer.Trainer") as trainer_class,
    ):
        trainer_class.return_value.fit.side_effect = fake_fit
        trainer_class.return_value.save_checkpoint.side_effect = fake_save_checkpoint
        run_training_job(
            spec,
            dataset_root=tmp_path / "snapshot",
            output_dir=tmp_path / "model",
            cache_dir=cache_dir,
            report=report or MagicMock(),
            should_stop=lambda: should_stop,
        )
    return trainer_class


class TestRunTrainingJob:
    def test_hf_token_is_set_during_the_run_and_cleared_after(self, tmp_path: Path, monkeypatch) -> None:
        """The token is scoped to this run: visible while training, gone once it returns."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        spec = TrainingJobSpec(policy="act", run_options=RunOptions(hf_token=SecretStr("hf-secret")))
        seen_during_fit = {}

        def fake_fit(*_args: object, **_kwargs: object) -> None:
            seen_during_fit["HF_TOKEN"] = os.environ.get("HF_TOKEN")
            (tmp_path / "cache" / "job" / CHECKPOINT_NAME).write_text("checkpoint")

        with (
            patch("physicalai.data.LeRobotDataModule"),
            patch(f"{JOB}.build_policy", return_value=MagicMock()),
            patch(f"{JOB}._load_policy_from_checkpoint", return_value=MagicMock()),
            patch("physicalai.train.trainer.Trainer") as trainer_class,
        ):
            trainer_class.return_value.fit.side_effect = fake_fit
            run_training_job(
                spec,
                dataset_root=tmp_path / "snapshot",
                output_dir=tmp_path / "model",
                cache_dir=tmp_path / "cache" / "job",
                report=MagicMock(),
                should_stop=lambda: False,
            )

        assert seen_during_fit["HF_TOKEN"] == "hf-secret"
        assert "HF_TOKEN" not in os.environ

    def test_hf_token_restores_a_prior_process_environment_value(self, tmp_path: Path, monkeypatch) -> None:
        """A token set on the trainer's own process (not per-job) is restored, not clobbered."""
        monkeypatch.setenv("HF_TOKEN", "operator-set-token")
        spec = TrainingJobSpec(policy="act", run_options=RunOptions(hf_token=SecretStr("job-secret")))

        _run(spec, tmp_path)

        assert os.environ["HF_TOKEN"] == "operator-set-token"

    def test_no_hf_token_leaves_environment_untouched(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        spec = TrainingJobSpec(policy="act")

        _run(spec, tmp_path)

        assert "HF_TOKEN" not in os.environ

    def test_trainer_is_configured_from_the_spec(self, tmp_path: Path) -> None:
        spec = TrainingJobSpec(
            policy="act",
            max_epochs=5,
            precision="32-true",
            auto_scale_batch_size=True,
            device_type="cpu",
            device_index=1,
        )

        trainer_class = _run(spec, tmp_path)

        kwargs = trainer_class.call_args.kwargs
        assert kwargs["max_epochs"] == 5
        assert kwargs["precision"] == "32-true"
        assert kwargs["auto_scale_batch_size"] is True
        assert (kwargs["accelerator"], kwargs["strategy"], kwargs["devices"]) == ("cpu", "auto", [1])

    def test_xpu_gets_its_single_device_strategy(self, tmp_path: Path) -> None:
        """Device resolution is shared with the trainer service; assert it is used."""
        trainer_class = _run(TrainingJobSpec(policy="act", device_type="xpu"), tmp_path)

        kwargs = trainer_class.call_args.kwargs
        assert (kwargs["accelerator"], kwargs["strategy"], kwargs["devices"]) == ("xpu", "xpu_single", 1)

    def test_dataset_is_loaded_from_the_local_root(self, tmp_path: Path) -> None:
        spec = TrainingJobSpec(policy="act", batch_size=16, num_workers=2, val_split=0.25)
        cache_dir = tmp_path / "cache" / "job"

        with (
            patch("physicalai.data.LeRobotDataModule") as datamodule,
            patch(f"{JOB}.build_policy"),
            patch(f"{JOB}._load_policy_from_checkpoint", return_value=MagicMock()),
            patch("physicalai.train.trainer.Trainer") as trainer_class,
        ):
            trainer_class.return_value.fit.side_effect = lambda *a, **k: (cache_dir / CHECKPOINT_NAME).write_text("x")
            run_training_job(
                spec,
                dataset_root=tmp_path / "snapshot",
                output_dir=tmp_path / "model",
                cache_dir=cache_dir,
                report=MagicMock(),
                should_stop=lambda: False,
            )

        kwargs = datamodule.call_args.kwargs
        assert kwargs["root"] == str(tmp_path / "snapshot")
        assert (kwargs["train_batch_size"], kwargs["num_workers"], kwargs["val_split"]) == (16, 2, 0.25)

    @pytest.mark.parametrize("augment_images", [True, False])
    def test_image_augmentation_is_opt_in(self, tmp_path: Path, augment_images: bool) -> None:
        """The GUI checkbox is the only thing standing between off and the default pipeline."""
        from physicalai.transforms import DefaultImageAugmentations

        spec = TrainingJobSpec(policy="act", augment_images=augment_images)
        cache_dir = tmp_path / "cache" / "job"

        with (
            patch("physicalai.data.LeRobotDataModule") as datamodule,
            patch(f"{JOB}.build_policy"),
            patch(f"{JOB}._load_policy_from_checkpoint", return_value=MagicMock()),
            patch("physicalai.train.trainer.Trainer") as trainer_class,
        ):
            trainer_class.return_value.fit.side_effect = lambda *a, **k: (cache_dir / CHECKPOINT_NAME).write_text("x")
            run_training_job(
                spec,
                dataset_root=tmp_path / "snapshot",
                output_dir=tmp_path / "model",
                cache_dir=cache_dir,
                report=MagicMock(),
                should_stop=lambda: False,
            )

        transforms = datamodule.call_args.kwargs["image_transforms"]
        if augment_images:
            assert isinstance(transforms, DefaultImageAugmentations)
        else:
            assert transforms is None

    def test_completed_run_publishes_the_cache_as_the_model_directory(self, tmp_path: Path) -> None:
        """The final checkpoint comes solely from the ModelCheckpoint callback, not an explicit save."""
        trainer_class = _run(TrainingJobSpec(policy="act"), tmp_path)

        trainer = trainer_class.return_value
        trainer.save_checkpoint.assert_not_called()
        assert (tmp_path / "model").is_dir()
        assert (tmp_path / "model" / CHECKPOINT_NAME).is_file()
        assert not (tmp_path / "cache" / "job").exists()

    def test_a_model_checkpoint_callback_is_configured_to_keep_the_best_epoch(self, tmp_path: Path) -> None:
        """The callback is the sole source of the published checkpoint; assert it stays configured that way."""
        from lightning.pytorch.callbacks import ModelCheckpoint

        trainer_class = _run(TrainingJobSpec(policy="act"), tmp_path)

        callbacks = trainer_class.call_args.kwargs["callbacks"]
        checkpoint_callbacks = [c for c in callbacks if isinstance(c, ModelCheckpoint)]
        assert len(checkpoint_callbacks) == 1
        callback = checkpoint_callbacks[0]
        assert callback.dirpath == str(tmp_path / "cache" / "job")
        assert (callback.monitor, callback.mode, callback.save_top_k) == ("val/loss", "min", 1)

    def test_missing_checkpoint_from_the_callback_falls_back_to_an_explicit_save(self, tmp_path: Path) -> None:
        """A run with nothing for the callback to monitor (e.g. no validation split) still gets a checkpoint."""
        trainer_class = _run(TrainingJobSpec(policy="act"), tmp_path, write_checkpoint=False)

        trainer = trainer_class.return_value
        trainer.save_checkpoint.assert_called_once_with(tmp_path / "cache" / "job" / CHECKPOINT_NAME)
        assert (tmp_path / "model" / CHECKPOINT_NAME).is_file()

    def test_completed_run_replaces_an_existing_model_directory(self, tmp_path: Path) -> None:
        """Retraining into the same directory must not merge with the old model."""
        output_dir = tmp_path / "model"
        output_dir.mkdir()
        (output_dir / "stale.txt").write_text("old")

        _run(TrainingJobSpec(policy="act"), tmp_path)

        assert not (output_dir / "stale.txt").exists()

    def test_canceled_run_leaves_no_model_behind(self, tmp_path: Path) -> None:
        """A partially trained policy is not an artifact worth keeping."""
        trainer_class = _run(TrainingJobSpec(policy="act"), tmp_path, should_stop=True)

        trainer_class.return_value.save_checkpoint.assert_not_called()
        assert not (tmp_path / "model").exists()

    def test_training_start_is_reported(self, tmp_path: Path) -> None:
        report = MagicMock()

        _run(TrainingJobSpec(policy="act"), tmp_path, report=report)

        assert report.call_args_list[0].args == (0, "Training model", {})

    def test_policy_is_exported_to_every_supported_backend(self, tmp_path: Path) -> None:
        policy = _ExportablePolicy([ExportBackend.TORCH, ExportBackend.OPENVINO])
        report = MagicMock()

        _run(TrainingJobSpec(policy="act"), tmp_path, policy=policy, report=report)

        exports = tmp_path / "model" / EXPORTS_DIRNAME
        assert policy.exported == [
            (str(exports / "torch"), ExportBackend.TORCH),
            (str(exports / "openvino"), ExportBackend.OPENVINO),
        ]
        assert (99, "Exporting to torch format", {}) in [call.args for call in report.call_args_list]

    def test_only_the_requested_export_backends_are_produced(self, tmp_path: Path) -> None:
        policy = _ExportablePolicy([ExportBackend.TORCH, ExportBackend.OPENVINO])

        _run(TrainingJobSpec(policy="act", export_backends=["openvino"]), tmp_path, policy=policy)

        assert [backend for _, backend in policy.exported] == [ExportBackend.OPENVINO]

    def test_a_requested_backend_the_policy_cannot_produce_is_skipped(self, tmp_path: Path) -> None:
        """Which formats a policy can trace is the policy's to say, not the caller's."""
        policy = _ExportablePolicy([ExportBackend.TORCH])

        _run(TrainingJobSpec(policy="smolvla", export_backends=["torch", "onnx"]), tmp_path, policy=policy)

        assert [backend for _, backend in policy.exported] == [ExportBackend.TORCH]

    def test_requesting_no_supported_backend_exports_nothing(self, tmp_path: Path) -> None:
        policy = _ExportablePolicy([ExportBackend.TORCH])

        _run(TrainingJobSpec(policy="act", export_backends=["onnx"]), tmp_path, policy=policy)

        assert policy.exported == []

    def test_a_failing_export_backend_does_not_fail_the_job(self, tmp_path: Path) -> None:
        """Weights are already saved by then; one bad backend must not lose them."""
        policy = _ExportablePolicy(
            [ExportBackend.TORCH, ExportBackend.OPENVINO],
            failing={ExportBackend.TORCH},
        )

        _run(TrainingJobSpec(policy="act"), tmp_path, policy=policy)

        assert [backend for _, backend in policy.exported] == [ExportBackend.OPENVINO]

    @pytest.mark.parametrize("policy_name", ["act", "smolvla", "pi05"])
    def test_compiled_policy_reloads_the_selected_checkpoint_for_export(self, policy_name: str, tmp_path: Path) -> None:
        """Exports must use the selected checkpoint even when training was compiled."""
        policy = _ExportablePolicy([ExportBackend.OPENVINO])

        with patch(f"{JOB}._load_policy_from_checkpoint") as reload:
            reload.return_value = policy
            _run(
                TrainingJobSpec(policy=policy_name, compile_model=True),
                tmp_path,
                policy=policy,
                checkpoint_loader=reload,
            )

        reload.assert_called_once()
        assert reload.call_args.args[1] == tmp_path / "model" / CHECKPOINT_NAME
        assert reload.call_args.args[0].compile_model is False
        assert [backend for _, backend in policy.exported] == [ExportBackend.OPENVINO]

    def test_checkpoint_reload_failure_fails_the_job(self, tmp_path: Path) -> None:
        """Do not silently export final-epoch weights when the selected checkpoint fails to load."""
        with (
            patch(f"{JOB}._load_policy_from_checkpoint", side_effect=RuntimeError("reload failed")) as reload,
            pytest.raises(RuntimeError, match="reload failed"),
        ):
            _run(TrainingJobSpec(policy="act"), tmp_path, checkpoint_loader=reload)

        reload.assert_called_once()

    def test_trainer_and_datamodule_are_released_before_export(self, tmp_path: Path) -> None:
        """Release training resources and the live policy before checkpoint reload.

        Lightning wires ``policy._trainer = trainer`` and ``trainer.datamodule =
        datamodule`` during ``fit`` and never undoes it; that link is simulated
        here since the real ``Trainer`` is mocked out. Without breaking it, the
        trained policy keeps the whole trainer graph reachable and no amount of
        ``gc.collect()`` can reclaim it. Checkpoint reconstruction must happen
        only after those references and the trained policy are released.
        """
        cache_dir = tmp_path / "cache" / "job"
        trained_policy_ref: list[weakref.ReferenceType[_ExportablePolicy] | None] = [None]
        exported_policy = _ExportablePolicy([ExportBackend.TORCH])

        with (
            patch("physicalai.data.LeRobotDataModule") as datamodule_class,
            patch(f"{JOB}.build_policy") as build_policy_mock,
            patch("physicalai.train.trainer.Trainer") as trainer_class,
            patch(f"{JOB}._release_memory", side_effect=gc.collect) as release_memory,
            patch(f"{JOB}._load_policy_from_checkpoint") as reload,
        ):
            trainer_instance = trainer_class.return_value
            datamodule_instance = datamodule_class.return_value

            def build_test_policy(*args: Any, **kwargs: Any) -> _ExportablePolicy:
                policy = _ExportablePolicy([ExportBackend.TORCH])
                trained_policy_ref[0] = weakref.ref(policy)
                return policy

            build_policy_mock.side_effect = build_test_policy

            # Simulate what Lightning's real `trainer.fit(policy, datamodule)` wires up,
            # including writing the checkpoint via the ModelCheckpoint callback.
            # A plain function avoids unittest.mock retaining fit's policy
            # argument in call history after training has returned.
            trainer_instance.fit = partial(
                _wire_mocked_fit,
                trainer_instance,
                cache_dir / CHECKPOINT_NAME,
            )

            trainer_ref = weakref.ref(trainer_instance)
            datamodule_ref = weakref.ref(datamodule_instance)

            reload.side_effect = partial(
                _assert_resources_released_before_reload,
                trainer_ref,
                trained_policy_ref,
                release_memory,
                exported_policy,
            )

            run_training_job(
                TrainingJobSpec(policy="act"),
                dataset_root=tmp_path / "snapshot",
                output_dir=tmp_path / "model",
                cache_dir=cache_dir,
                report=MagicMock(),
                should_stop=lambda: False,
            )

            reload.assert_called_once()

        # Drop the test's own strong refs before collecting, mirroring what
        # run_training_job does internally with its local `trainer`/`datamodule`.
        del trainer_instance, datamodule_instance, trainer_class, datamodule_class, reload, release_memory
        gc.collect()

        assert trainer_ref() is None, "trainer is still reachable; the reference cycle was not broken"
        assert datamodule_ref() is None, "datamodule is still reachable; the reference cycle was not broken"
        assert trained_policy_ref[0] is not None and trained_policy_ref[0]() is None


SNAPFLOW_SPEC = TrainingJobSpec(policy="pi05", max_epochs=8, snapflow_start_epoch=5)


class TestEarlyStoppingRun:
    """How ``run_training_job`` wires early stopping; the stopping itself is in test_early_stopping.py."""

    @staticmethod
    def _early_stopping_callback(trainer_class: MagicMock) -> Any | None:
        from training.early_stopping import ValLossEarlyStopping

        callbacks = trainer_class.call_args.kwargs["callbacks"]
        return next((c for c in callbacks if isinstance(c, ValLossEarlyStopping)), None)

    @staticmethod
    def _stopped_summary() -> EarlyStoppingSummary:
        return EarlyStoppingSummary(
            patience=2,
            min_delta=0.0,
            max_epochs=20,
            epochs_trained=7,
            stopped_after_epoch=7,
            reason="patience_exhausted",
            best_epoch=5,
            best_val_loss=0.25,
        )

    def test_an_ordinary_run_gets_no_early_stopping(self, tmp_path: Path) -> None:
        from lightning.pytorch.callbacks import EarlyStopping

        trainer_class = _run(TrainingJobSpec(policy="act"), tmp_path)

        callbacks = trainer_class.call_args.kwargs["callbacks"]
        assert not any(isinstance(callback, EarlyStopping) for callback in callbacks)
        assert not (tmp_path / "model" / EARLY_STOPPING_SUMMARY_NAME).exists()

    def test_early_stopping_watches_val_loss_with_the_spec_settings(self, tmp_path: Path) -> None:
        spec = TrainingJobSpec(policy="act", early_stopping_patience=4, early_stopping_min_delta=0.01)

        with patch("training.early_stopping.ValLossEarlyStopping.summary", return_value=self._stopped_summary()):
            trainer_class = _run(spec, tmp_path)

        callback = self._early_stopping_callback(trainer_class)
        assert callback is not None
        assert (callback.monitor, callback.mode, callback.patience) == ("val/loss", "min", 4)
        # Lightning stores min_delta signed by mode; the magnitude is what was asked for.
        assert abs(callback.min_delta) == pytest.approx(0.01)
        # It ranks epochs by the metric the published checkpoint is chosen by.
        checkpoint = next(
            c for c in trainer_class.call_args.kwargs["callbacks"] if type(c).__name__ == "ModelCheckpoint"
        )
        assert (checkpoint.monitor, checkpoint.mode) == (callback.monitor, callback.mode)

    def test_on_a_snapflow_run_it_is_tied_to_the_phase_callback(self, tmp_path: Path) -> None:
        spec = TrainingJobSpec(policy="pi05", max_epochs=8, snapflow_start_epoch=5, early_stopping_patience=2)

        with patch("training.early_stopping.ValLossEarlyStopping.summary", return_value=self._stopped_summary()):
            trainer_class = _run(spec, tmp_path)

        callback = self._early_stopping_callback(trainer_class)
        snapflow = next(
            c for c in trainer_class.call_args.kwargs["callbacks"] if type(c).__name__ == "SnapFlowPhaseCallback"
        )
        assert callback is not None
        assert callback._snapflow is snapflow

    def test_an_early_stop_is_recorded_in_the_model_and_reported(self, tmp_path: Path) -> None:
        report = MagicMock()
        spec = TrainingJobSpec(policy="act", early_stopping_patience=2)

        with patch("training.early_stopping.ValLossEarlyStopping.summary", return_value=self._stopped_summary()):
            _run(spec, tmp_path, report=report)

        assert read_early_stopping_summary(tmp_path / "model") == self._stopped_summary()
        report.assert_any_call(99, "Stopped early after epoch 7 of 20; best epoch 5 (val/loss 0.25)", {})
        # The checkpoint and export still follow, as for a run that used all its epochs.
        assert (tmp_path / "model" / CHECKPOINT_NAME).is_file()

    def test_a_run_early_stopping_did_not_end_records_that_too(self, tmp_path: Path) -> None:
        report = MagicMock()
        summary = self._stopped_summary().model_copy(update={"stopped_after_epoch": None, "reason": None})

        with patch("training.early_stopping.ValLossEarlyStopping.summary", return_value=summary):
            _run(TrainingJobSpec(policy="act", early_stopping_patience=2), tmp_path, report=report)

        assert read_early_stopping_summary(tmp_path / "model") == summary
        assert not any("Stopped early" in str(call.args[1]) for call in report.call_args_list)

    def test_a_summary_that_cannot_be_built_does_not_fail_the_run(self, tmp_path: Path) -> None:
        with patch("training.early_stopping.ValLossEarlyStopping.summary", side_effect=RuntimeError("boom")):
            _run(TrainingJobSpec(policy="act", early_stopping_patience=2), tmp_path)

        assert (tmp_path / "model" / CHECKPOINT_NAME).is_file()
        assert not (tmp_path / "model" / EARLY_STOPPING_SUMMARY_NAME).exists()

    def test_a_canceled_run_records_nothing(self, tmp_path: Path) -> None:
        with patch("training.early_stopping.ValLossEarlyStopping.summary") as summary:
            _run(TrainingJobSpec(policy="act", early_stopping_patience=2), tmp_path, should_stop=True)

        summary.assert_not_called()


class TestSnapFlowDistillation:
    """A distillation run keeps two checkpoints, and export/resolve pick the distilled one."""

    @staticmethod
    def _snapflow_callback(trainer_class: MagicMock) -> Any | None:
        callbacks = trainer_class.call_args.kwargs["callbacks"]
        return next((c for c in callbacks if type(c).__name__ == "SnapFlowPhaseCallback"), None)

    def test_a_flow_matching_run_gets_no_phase_callback(self, tmp_path: Path) -> None:
        trainer_class = _run(TrainingJobSpec(policy="pi05"), tmp_path)

        assert self._snapflow_callback(trainer_class) is None

    def test_the_phase_callback_fires_at_the_boundary_the_spec_asks_for(self, tmp_path: Path) -> None:
        trainer_class = _run(SNAPFLOW_SPEC, tmp_path, reach_snapflow_phase=True)

        callback = self._snapflow_callback(trainer_class)
        assert callback is not None
        assert (callback.start_epoch, callback.start_step) == (5, None)
        # Phase-2 checkpoints must land beside phase 1's rather than replace
        # them; the prefix is what SNAPFLOW_CHECKPOINT_NAME relies on.
        assert callback.checkpoint_prefix == "snapflow-"

    def test_both_checkpoints_are_kept_side_by_side(self, tmp_path: Path) -> None:
        """Neither file is renamed or deleted: a distilled model still ships its teacher."""
        _run(SNAPFLOW_SPEC, tmp_path, reach_snapflow_phase=True)

        assert (tmp_path / "model" / CHECKPOINT_NAME).read_text() == "flow-matching"
        assert (tmp_path / "model" / SNAPFLOW_CHECKPOINT_NAME).read_text() == "distilled"

    def test_a_distillation_phase_that_saved_nothing_falls_back_to_the_live_weights(self, tmp_path: Path) -> None:
        """The run did distil, so it must not ship with no distilled checkpoint at all."""
        trainer_class = _run(
            SNAPFLOW_SPEC,
            tmp_path,
            reach_snapflow_phase=True,
            write_distilled_checkpoint=False,
        )

        trainer_class.return_value.save_checkpoint.assert_called_once_with(
            tmp_path / "cache" / "job" / SNAPFLOW_CHECKPOINT_NAME
        )
        assert (tmp_path / "model" / CHECKPOINT_NAME).read_text() == "flow-matching"
        assert (tmp_path / "model" / SNAPFLOW_CHECKPOINT_NAME).is_file()

    def test_a_run_that_never_reached_the_boundary_ships_only_the_flow_matching_checkpoint(
        self, tmp_path: Path
    ) -> None:
        """No distillation happened (e.g. max_epochs was lowered on a resume), so nothing is fabricated."""
        _run(SNAPFLOW_SPEC, tmp_path, reach_snapflow_phase=False)

        assert (tmp_path / "model" / CHECKPOINT_NAME).read_text() == "flow-matching"
        assert not (tmp_path / "model" / SNAPFLOW_CHECKPOINT_NAME).exists()

    def test_exports_reload_from_the_distilled_checkpoint(self, tmp_path: Path) -> None:
        """The live end-of-fit policy is the final epoch, not necessarily the best
        distilled one on disk; export must reload to match what resolve_checkpoint
        (and therefore resume/download) would pick."""
        policy = _ExportablePolicy([ExportBackend.OPENVINO])

        with patch(f"{JOB}._load_policy_from_checkpoint") as reload:
            _run(
                SNAPFLOW_SPEC,
                tmp_path,
                policy=policy,
                reach_snapflow_phase=True,
                checkpoint_loader=reload,
            )

        reload.assert_called_once()
        assert reload.call_args.args[1] == tmp_path / "model" / SNAPFLOW_CHECKPOINT_NAME

    def test_a_flow_matching_run_reloads_the_selected_checkpoint_for_export(self, tmp_path: Path) -> None:
        policy = _ExportablePolicy([ExportBackend.OPENVINO])

        with patch(f"{JOB}._load_policy_from_checkpoint") as reload:
            reload.return_value = policy
            _run(TrainingJobSpec(policy="pi05"), tmp_path, policy=policy, checkpoint_loader=reload)

        reload.assert_called_once()
        assert reload.call_args.args[1] == tmp_path / "model" / CHECKPOINT_NAME
        assert [backend for _, backend in policy.exported] == [ExportBackend.OPENVINO]


class TestResolveCheckpoint:
    def test_prefers_the_distilled_checkpoint_when_present(self, tmp_path: Path) -> None:
        (tmp_path / CHECKPOINT_NAME).write_text("flow-matching")
        (tmp_path / SNAPFLOW_CHECKPOINT_NAME).write_text("distilled")

        assert resolve_checkpoint(tmp_path) == tmp_path / SNAPFLOW_CHECKPOINT_NAME

    def test_falls_back_to_the_ordinary_checkpoint(self, tmp_path: Path) -> None:
        (tmp_path / CHECKPOINT_NAME).write_text("flow-matching")

        assert resolve_checkpoint(tmp_path) == tmp_path / CHECKPOINT_NAME


class TestLoadPolicyFromCheckpoint:
    def test_pi05_normalizes_max_autotune_to_default(self, tmp_path: Path) -> None:
        forward_fn = MagicMock()
        policy = MagicMock()
        policy.config = SimpleNamespace(compile_mode="max-autotune")
        policy.forward = forward_fn

        spec = TrainingJobSpec(policy="pi05", compile_model=True)
        with (
            patch("physicalai.policies.get_physicalai_policy_class") as get_cls,
            patch("torch.compile") as mock_compile,
        ):
            get_cls.return_value.load_from_checkpoint.return_value = policy
            _load_policy_from_checkpoint(spec, tmp_path / "checkpoint.ckpt")

        mock_compile.assert_called_once_with(forward_fn, mode="default")
