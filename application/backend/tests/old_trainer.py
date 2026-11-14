# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A stand-in for a trainer that predates early stopping.

A deployed trainer (a Modal app, an AWS Batch image, a remote trainer service)
runs whatever version of :class:`training.TrainingJobSpec` it was built with,
and that model forbids unknown fields. These helpers rebuild the spec as it was
before the early-stopping fields existed, so tests can check what such a
trainer accepts and how it fails.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from training import TrainingJobSpec

EARLY_STOPPING_FIELDS = ("early_stopping_patience", "early_stopping_min_delta")


def pre_early_stopping_spec_model() -> type[BaseModel]:
    """``TrainingJobSpec`` without its early-stopping fields, forbidding unknown fields like the real one."""
    fields = {
        name: (field.annotation, field)
        for name, field in TrainingJobSpec.model_fields.items()
        if name not in EARLY_STOPPING_FIELDS
    }
    return create_model(  # type: ignore[call-overload]
        "TrainingJobSpec", __config__=ConfigDict(extra="forbid"), **fields
    )


def old_trainer_rejection(spec_json: str) -> str | None:
    """Validate a serialized spec the way an old trainer does.

    Returns:
        ``str(ValidationError)`` -- what the batch worker writes into its
        failed ``status.json`` -- or None when the old trainer accepts the spec.
    """
    try:
        pre_early_stopping_spec_model().model_validate_json(spec_json)
    except ValidationError as exc:
        return str(exc)
    return None
