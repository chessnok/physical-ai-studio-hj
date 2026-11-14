# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Deploy with `modal deploy application/modal/train.py` from the repository root.

The registry image provides the training environment (Python, PyTorch, the
``physicalai`` library) together with an upstream copy of the ``trainer`` and
``training`` packages. The deployment overlays this checkout's own copies of
those two packages, which :func:`train` puts first on ``sys.path``, so a Modal
run executes the same training code as the Studio that deployed it (for
example, the spec fields Studio sends). Redeploy after changing them.
"""

import os
import sys
from pathlib import Path
from uuid import UUID

import modal

STUDIO_SOURCE_DIR = "/opt/physicalai-studio/src"
"""Where this checkout's ``trainer`` and ``training`` packages are mounted in the container."""

_OVERLAID_PACKAGES = ("trainer", "training")
# Parent of this file's package; only meaningful where the app is deployed from.
_LOCAL_SOURCE_DIR = Path(__file__).resolve().parents[1]

app = modal.App(os.environ.get("PHYSICALAI_MODAL_APP", "physicalai-trainer"))
volume = modal.Volume.from_name(
    os.environ.get("PHYSICALAI_MODAL_VOLUME", "physicalai-training"), create_if_missing=True
)
image = modal.Image.from_registry(
    os.environ.get("PHYSICALAI_MODAL_IMAGE", "ghcr.io/open-edge-platform/physicalai-trainer-cuda:main"),
    # The trainer image strips pip; Modal needs it to install its container runtime.
    setup_dockerfile_commands=["RUN python -m ensurepip --default-pip"],
)
for _package in _OVERLAID_PACKAGES:
    image = image.add_local_dir(
        _LOCAL_SOURCE_DIR / _package,
        f"{STUDIO_SOURCE_DIR}/{_package}",
        ignore=lambda path: "__pycache__" in Path(path).parts,
    )
secret_name = os.environ.get("PHYSICALAI_MODAL_SECRET")


@app.function(
    image=image,
    gpu=os.environ.get("PHYSICALAI_MODAL_GPU", "A10G"),
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    retries=0,
    volumes={"/training-jobs": volume},
    secrets=[modal.Secret.from_name(secret_name)] if secret_name else [],
)
def train(job_id: str) -> int:
    """Run the existing batch worker; expose only UUID-named job directories."""
    import shutil

    # Before the first import of either package: the overlaid source wins over the image's copy.
    if Path(STUDIO_SOURCE_DIR).is_dir() and STUDIO_SOURCE_DIR not in sys.path:
        sys.path.insert(0, STUDIO_SOURCE_DIR)

    from trainer.batch import BatchSettings, LocalDirJobStorage, run
    from trainer.batch.status import BatchJobStatus
    from trainer.settings import get_settings

    job_id = str(UUID(job_id))
    volume.reload()
    root = Path("/training-jobs") / "jobs" / job_id
    settings = BatchSettings(PHYSICALAI_BATCH_BUCKET="modal", PHYSICALAI_BATCH_JOB_PREFIX=f"jobs/{job_id}")

    class VolumeJobStorage(LocalDirJobStorage):
        def put_status(self, status: BatchJobStatus) -> None:
            super().put_status(status)
            # Commit metrics and artifacts before publishing their corresponding status.
            volume.commit()

    try:
        return run(VolumeJobStorage(root, settings), settings)
    finally:
        volume.commit()
        local = get_settings()
        for path in (local.models_dir / job_id, local.storage_dir / "cache" / job_id):
            shutil.rmtree(path, ignore_errors=True)
        (local.archives_dir / f"{job_id}.zip").unlink(missing_ok=True)
