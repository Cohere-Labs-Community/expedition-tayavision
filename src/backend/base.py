"""Backend seam: the contract every accelerator path implements.

Why this exists
---------------
`pipeline/train_*.py` used to hardcode CUDA and DDP: `torch.autocast("cuda")`,
`torch.device(f"cuda:{rank}")`, `torch.cuda.manual_seed_all`,
`DistributedDataParallel`, `DistributedSampler`. None of that runs under
torch_xla, and inlining `if tpu: ... else: ...` at every call site would make the
Modal/GPU path -- which produced every published number -- something you have to
re-verify on every TPU change.

So the accelerator-specific behaviour lives behind this ABC, and exactly one
module is allowed to `import torch_xla` at module scope:
`src/backend/tpu_backend.py`. `scripts/ci/check_backend_seam.sh` enforces that.
A module-level `import torch_xla` anywhere in the shared trees drags `libtpu`
into the import graph and breaks every CPU/GPU run and the whole test suite.

Selection is by environment, not by import:
    TAYAVISION_TPU=1  ->  TPUBackend      (set by scripts/tpu/train_launcher.sh)
    otherwise         ->  GPUBackend      (DDP + CUDA, unchanged)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any

import torch


class Backend(ABC):
    """One accelerator strategy: device placement, distribution, stepping."""

    name: str = "base"

    # --- topology -----------------------------------------------------------

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The device tensors and the model should live on."""

    @property
    @abstractmethod
    def rank(self) -> int:
        """Global process rank. 0 when not distributed."""

    @property
    @abstractmethod
    def world_size(self) -> int:
        """Number of participating processes.

        NOTE this is *processes*, not chips. Under SPMD it is 1 even on a
        64-chip slice -- one process drives the whole mesh. Anything that
        divides a global batch by this must account for that.
        """

    @property
    def local_rank(self) -> int:
        return self.rank

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    @abstractmethod
    def data_parallel_size(self) -> int:
        """How many ways the *data* is split.

        Equals `world_size` under DDP, and the chip count under SPMD. This --
        not `world_size` -- is what a per-device batch size should be derived
        from.
        """

    # --- lifecycle ----------------------------------------------------------

    @abstractmethod
    def setup(self) -> None:
        """Initialise process groups / runtime. Idempotent."""

    @abstractmethod
    def cleanup(self) -> None:
        """Tear down. Safe to call when setup never ran."""

    @abstractmethod
    def barrier(self) -> None:
        """Synchronise all processes. No-op when not distributed."""

    # --- compute ------------------------------------------------------------

    @abstractmethod
    def autocast(self, dtype: torch.dtype) -> AbstractContextManager:
        """Mixed-precision context for this backend's device type."""

    @abstractmethod
    def manual_seed(self, seed: int) -> None:
        """Seed the accelerator RNG in addition to torch's CPU RNG."""

    @abstractmethod
    def wrap_model(self, model: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
        """Apply the distribution strategy. May return the model unchanged."""

    @abstractmethod
    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step the optimizer, including any backend-specific graph flush."""

    @abstractmethod
    def sync(self) -> None:
        """Flush pending work. No-op on eager backends."""

    # --- data ---------------------------------------------------------------

    @abstractmethod
    def make_sampler(self, dataset: Any, *, shuffle: bool, seed: int) -> Any | None:
        """A distributed sampler, or None when the backend must not shard.

        Returning None matters: under SPMD there is ONE process, so a
        `DistributedSampler` would hand it a 1/N shard and it would train on
        1/N of the data while reporting nothing wrong.
        """

    @abstractmethod
    def wrap_loader(self, loader: Any) -> Any:
        """Wrap a DataLoader for device feeding. Identity on GPU."""

    # --- checkpointing ------------------------------------------------------

    @abstractmethod
    def save(self, obj: Any, path: Any) -> None:
        """Persist `obj`, which may contain device tensors.

        CALL THIS ON EVERY RANK. Moving tensors off the accelerator is device
        work, and on SPMD device work is collective -- if only rank 0 calls it,
        the mesh desynchronises and the run deadlocks with no traceback. The
        implementation decides which rank writes the bytes; the caller must not.
        """

    # --- diagnostics --------------------------------------------------------

    @abstractmethod
    def memory_info(self) -> dict[str, float]:
        """Best-effort {used_gib, total_gib}. Empty dict when unavailable."""

    @property
    @abstractmethod
    def addressable_device_count(self) -> int:
        """Devices owned by THIS process."""

    @abstractmethod
    def check_topology(self, *, global_batch: int) -> list[str]:
        """Startup guards. Returns problems; empty list means healthy."""

    def describe(self) -> str:
        return (f"{self.name}: device={self.device} rank={self.rank}/"
                f"{self.world_size} dp={self.data_parallel_size}")


def create_backend(force: str | None = None) -> Backend:
    """Select a backend from the environment.

    Imports are deferred into the branches on purpose: importing
    `tpu_backend` pulls in `torch_xla`, and that must not happen on a GPU box.
    """
    choice = (force or os.environ.get("TAYAVISION_BACKEND") or "").lower()
    if not choice:
        choice = "tpu" if os.environ.get("TAYAVISION_TPU") == "1" else "gpu"

    if choice == "tpu":
        from src.backend.tpu_backend import TPUBackend

        return TPUBackend()
    if choice == "gpu":
        from src.backend.gpu_backend import GPUBackend

        return GPUBackend()
    raise ValueError(f"Unknown backend {choice!r}; expected 'gpu' or 'tpu'")
