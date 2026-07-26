"""CUDA + DDP backend -- the authoritative path.

Every published number, checkpoint, and eval came from this code. It is an
extraction, not a rewrite: each method below does exactly what the corresponding
lines of `pipeline/train_alignment.py` and `pipeline/utils.py` did before the
seam existed, including the details that look incidental.

If you are tempted to "improve" something here, don't. Behaviour changes belong
in a separate commit with a re-run behind them.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from src.backend.base import Backend


class GPUBackend(Backend):
    name = "gpu"

    def __init__(self) -> None:
        # `is_torchrun()` in pipeline/utils.py -- kept as the same LOCAL_RANK
        # probe rather than importing it, so src/ does not depend on pipeline/.
        self._use_ddp = "LOCAL_RANK" in os.environ
        self._local_rank = 0
        self._rank = 0
        self._world_size = 1
        self._device: torch.device | None = None

    # --- topology -----------------------------------------------------------

    @property
    def device(self) -> torch.device:
        if self._device is None:
            # Matches the pre-seam branch exactly: an explicit cuda:<local_rank>
            # under torchrun, otherwise cuda-if-available with a CPU fallback.
            # The CPU fallback is load-bearing -- the test suite runs on it.
            self._device = (
                torch.device(f"cuda:{self._local_rank}")
                if self._use_ddp
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        return self._device

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def data_parallel_size(self) -> int:
        # Under DDP one process drives one GPU, so these coincide.
        return self._world_size

    # --- lifecycle ----------------------------------------------------------

    def setup(self) -> None:
        if not self._use_ddp or dist.is_initialized():
            return
        self._local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self._local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{self._local_rank}"),
        )
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._device = None  # recompute now that local_rank is known

    def cleanup(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()

    # --- compute ------------------------------------------------------------

    def autocast(self, dtype: torch.dtype) -> AbstractContextManager:
        return torch.autocast("cuda", dtype=dtype)

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def wrap_model(self, model: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
        if not self._use_ddp:
            return model
        return DDP(model, device_ids=[self._local_rank], **kwargs)

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def sync(self) -> None:
        return None  # CUDA is eager from the caller's point of view

    # --- data ---------------------------------------------------------------

    def make_sampler(self, dataset: Any, *, shuffle: bool, seed: int) -> Any | None:
        if not self._use_ddp:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=shuffle,
            seed=seed,
        )

    def wrap_loader(self, loader: Any) -> Any:
        return loader

    # --- checkpointing ------------------------------------------------------

    def save(self, obj: Any, path: Any) -> None:
        """Rank 0 writes; the others return immediately.

        Unchanged from the pre-seam behaviour, where the caller wrapped the
        save in `if is_main`. Under DDP that is safe -- nothing here is
        collective, so the non-writers can simply skip it.
        """
        if self.is_main:
            torch.save(obj, path)

    @property
    def addressable_device_count(self) -> int:
        return 1  # one process drives one GPU under DDP

    def check_topology(self, *, global_batch: int) -> list[str]:
        # Same contract as the TPU path so callers need no branch. Under DDP
        # processes and devices coincide, so there is only one divisor to check.
        if global_batch % self._world_size:
            return [
                f"global batch {global_batch} not divisible by world_size "
                f"{self._world_size}"
            ]
        return []

    # --- diagnostics --------------------------------------------------------

    def memory_info(self) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        free_b, total_b = torch.cuda.mem_get_info()
        return {
            "used_gib": (total_b - free_b) / 1024**3,
            "total_gib": total_b / 1024**3,
        }
