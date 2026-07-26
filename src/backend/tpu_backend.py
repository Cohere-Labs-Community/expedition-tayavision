"""torch_xla SPMD backend -- experimental.

**This is the only module in the repo allowed a module-level `import torch_xla`.**
`scripts/ci/check_backend_seam.sh` enforces that. A module-level import anywhere
in `src/`, `models/`, `pipeline/`, or `config/` drags `libtpu` into the import
graph, which breaks every CPU/GPU run and all 129 tests.

Measured on a live v6e-16 (2026-07-25) -- see
`.claude/orchestration/playbook/baseline-v6e8-siglip-cohere2.md`:

  * the 4.33B model materializes at **8.058 GiB per chip** of 31.246 GiB
  * so `replicated` fits with ~21 GiB of headroom under the 29 GiB ceiling

That is why `replicated` is the default and FSDPv2 is opt-in: wrapping each of
the 36 `Cohere2DecoderLayer`s produces 36 bf16 reduce-scatters and hits
pytorch/xla #8591 / #8778 (NaN at step ~24-130) on this exact backbone. At this
parameter budget that risk is avoidable rather than survivable.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from typing import Any

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr

from src.backend.base import Backend


class _XLADeviceModule:
    """The bare minimum `torch.utils.checkpoint` needs to see for device "xla".

    Gradient checkpointing is mandatory here -- it is what keeps activations to
    per-layer boundaries (~8.7 GiB/chip instead of something that will not fit).
    But `torch.utils.checkpoint` cannot run on XLA out of the box:

        checkpoint.py:92   def _get_device_module(device): return getattr(torch, device)
        checkpoint.py:132  _infer_device_type() -> "xla"   (tensors are on xla, no cuda)
        checkpoint.py:1476 device_module = _get_device_module(device_type)   # UNCONDITIONAL

    and `torch.xla` does not exist -- the module is `torch_xla` -- so it raises
    `AttributeError: module 'torch' has no attribute 'xla'` before step 1.
    Line 1476 runs BEFORE the `if preserve_rng_state:` guard, which is why
    passing `preserve_rng_state=False` does not avoid it, and
    `DefaultDeviceType.set_device_type()` does not help either because that only
    applies when there are NO non-CPU tensors.

    The one thing torch then asks of the module is:

        checkpoint.py:1496  if getattr(device_module, "_initialized", False):
                                fwd_devices, fwd_device_states = get_device_states(*args)

    So a module that deliberately does **not** define `_initialized` makes torch
    skip RNG stashing entirely, and `get_device_states` -- which would call
    `.device()` / `.get_rng_state()` that torch_xla has no module-level
    equivalent for -- is never reached.

    That skip is semantically right, not a dodge: XLA RNG is managed through
    `xm.set_rng_state`, not torch's per-device generator, and this backend seeds
    it in `manual_seed`. Deliberately NOT registering `torch_xla` itself, which
    carries a large surface that other torch code could then discover.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return "<tayavision xla device-module shim>"


def _register_xla_device_module() -> None:
    """Make `getattr(torch, "xla")` resolve. Idempotent."""
    if hasattr(torch, "xla"):
        return
    shim = _XLADeviceModule()
    register = getattr(torch, "_register_device_module", None)
    if register is not None:
        try:
            register("xla", shim)
            return
        except Exception:  # noqa: BLE001 - already registered, or refused
            if hasattr(torch, "xla"):
                return
    # Fall back to a plain attribute: the only contract torch.utils.checkpoint
    # has with this object is `getattr(module, "_initialized", False)`.
    torch.xla = shim  # type: ignore[attr-defined]


def _to_cpu(obj: Any) -> Any:
    """Recursively pull tensors to CPU, leaving everything else alone.

    Written out rather than reusing `xm._maybe_convert_to_cpu` for two reasons:
    that helper is private, and its `convert=` flag is the very footgun this
    exists to avoid (see `TPUBackend.save`). Here the transfer is
    unconditional, so it is identical on every rank by construction.
    """
    if torch.is_tensor(obj):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        # Preserve namedtuples, which take positional args rather than an iterable.
        vals = [_to_cpu(v) for v in obj]
        return type(obj)(*vals) if hasattr(obj, "_fields") else tuple(vals)
    return obj


def _assert_checkpoint_contract() -> None:
    """Run one real non-reentrant checkpoint on device, at setup, and fail loud.

    WHY THIS EXISTS
    ---------------
    Everything the shim above does rests on torch internals that are private and
    that provably move between releases. `torch/nn/parallel/_functions.py` used
    `getattr(torch, device.type, None)` in 2.6.0 and was rewritten to
    `torch.accelerator` in 2.9.1 -- one risk site vanished between the venv this
    was authored in and the VM that runs it. `scripts/tpu/startup_script.sh`
    installs `torch==${TORCH_VERSION:-2.9.*}`, a WILDCARD resolved at
    provisioning time, so the torch under this shim can change without a commit.

    Two upstream changes would break the shim, and both are plausible:

      * `getattr(device_module, "_initialized", False)` (checkpoint.py:1496)
        becomes `device_module.is_initialized()`. torch is already moving that
        way -- `torch/distributed/device_mesh.py:506` and
        `torch/_subclasses/fake_tensor.py:719` both call `is_initialized()` with
        NO default. The shim would then raise AttributeError.
      * `_get_device_module` stops being a bare `getattr(torch, device)`.

    Neither is caught by anything else. `tests/` cannot import this module (it
    needs libtpu), and `scripts/ci/check_backend_seam.sh` only checks import
    placement. So the shim's contract is asserted NOWHERE -- which is exactly
    how a safety argument decays into a stale comment.

    This runs the real thing: a non-reentrant checkpoint over an XLA tensor with
    `preserve_rng_state=True`, which is the precise configuration
    `pipeline/train_alignment.py` uses. It touches `_get_device_module`,
    `_infer_device_type`, the `_initialized` gate, `fork_rng(device_type="xla")`,
    and the autocast capture/replay -- every internal the shim depends on. Costs
    one tiny HLO compile (<1s, 4x4 matmul) against a v6e-16 run measured in
    hours.

    Set TAYAVISION_SKIP_CKPT_CONTRACT=1 to skip. Do that only to get a diagnostic
    run past a KNOWN break -- gradient checkpointing is not optional at this
    memory budget, so skipping this does not make the run work, it only moves the
    failure somewhere less legible.
    """
    if os.environ.get("TAYAVISION_SKIP_CKPT_CONTRACT") == "1":
        return

    from torch.utils.checkpoint import checkpoint

    dev = torch_xla.device()
    lin = torch.nn.Linear(4, 4).to(dev)
    x = torch.randn(2, 4, device=dev, requires_grad=True)

    try:
        # Mirror the trainer exactly: non-reentrant, preserve_rng_state default
        # (True), under this backend's autocast.
        with torch.autocast("xla", dtype=torch.bfloat16):
            out = checkpoint(lambda t: lin(t).relu(), x, use_reentrant=False)
        out.sum().backward()
        torch_xla.sync()
    except Exception as exc:  # noqa: BLE001 - re-raised with the diagnosis
        raise RuntimeError(
            "XLA gradient-checkpointing contract FAILED. The `torch.xla` shim in "
            "src/backend/tpu_backend.py no longer satisfies torch.utils.checkpoint.\n"
            f"  torch={torch.__version__} "
            f"torch_xla={getattr(torch_xla, '__version__', 'unknown')}\n"
            f"  underlying: {type(exc).__name__}: {exc}\n"
            "Re-diff torch/utils/checkpoint.py (_get_device_module, "
            "_infer_device_type, the `_initialized` gate, the fork_rng call) and "
            "torch/random.py against the version this shim was written for "
            "(2.6.0 / 2.9.1). Do NOT paper over it by adding attributes to "
            "_XLADeviceModule until you know which gate flipped -- a guessed "
            "attribute turns a loud AttributeError into a silently wrong device."
        ) from exc

    if x.grad is None:
        raise RuntimeError(
            "XLA gradient-checkpointing contract FAILED: backward through "
            "checkpoint produced no gradient. Activations would be recomputed "
            "for nothing and the projector would never train."
        )


class TPUBackend(Backend):
    name = "tpu"

    def __init__(self) -> None:
        self._strategy = os.environ.get("TPU_STRATEGY", "replicated").lower()
        self._device: torch.device | None = None
        self._setup_done = False
        self._mesh = None

    # --- topology -----------------------------------------------------------

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = torch_xla.device()
        return self._device

    @property
    def rank(self) -> int:
        """Index of THIS process among the hosts. 0..process_count-1.

        `xr.process_index()`, NOT `xr.global_ordinal()`. Measured on the v6e-16
        (2026-07-25): `global_ordinal()` returns **0 on every host** and
        `world_size()` returns **1 on every host** -- those are the non-SPMD
        values and are useless here. Using them would make host sharding a
        silent no-op, with all four hosts training on shard 0.

        Also note process_index is NOT the gcloud worker number: measured
        w-0->2, w-1->0, w-2->1, w-3->3. It is a valid permutation (each index
        appears exactly once), which is all sharding needs -- but it means the
        `is_main` process, and therefore the W&B writer, was on **worker 1**.
        Do not go looking for the wandb log on worker 0.
        """
        return xr.process_index()

    @property
    def world_size(self) -> int:
        """Number of PROCESSES (= hosts). 4 on the v6e-16, 1 on a v6e-8.

        This is what the DataLoader batch divides by. Not the chip count.
        """
        return xr.process_count()

    @property
    def data_parallel_size(self) -> int:
        """Total CHIPS across all hosts -- 16 on the v6e-16.

        Distinct from `world_size` (4 processes). The mesh shards across chips;
        the DataLoader splits across processes. Conflating the two is what made
        `batch_size % world_size` fail at 8 % 16.
        """
        return xr.global_runtime_device_count()

    @property
    def addressable_device_count(self) -> int:
        """Chips owned by THIS process. 4 on the v6e-16."""
        return xr.addressable_runtime_device_count()

    # --- lifecycle ----------------------------------------------------------

    def setup(self) -> None:
        if self._setup_done:
            return
        xr.use_spmd()
        _register_xla_device_module()
        # Assert the shim's contract on real silicon before anything expensive
        # happens. A break here costs one compile; the same break discovered at
        # the first backward costs the model load, the dataset scan, and the
        # slice time to get there.
        _assert_checkpoint_contract()
        self._setup_done = True

    def cleanup(self) -> None:
        # Flush anything still queued so a checkpoint write is not left pending.
        try:
            torch_xla.sync()
        except Exception:  # noqa: BLE001 - teardown must never raise
            pass

    def barrier(self) -> None:
        # One process: nothing to synchronise with. Still flush, because callers
        # use barrier() to mean "everything before this is done".
        torch_xla.sync()

    # --- compute ------------------------------------------------------------

    def autocast(self, dtype: torch.dtype) -> AbstractContextManager:
        return torch.autocast("xla", dtype=dtype)

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        xm.set_rng_state(seed)

    def wrap_model(self, model: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
        if self._strategy == "replicated":
            # Nothing to wrap: SPMD replicates by default and the model fits.
            return model

        if self._strategy in ("fsdpv2", "fsdpv2_lora"):
            from torch_xla.distributed.spmd.xla_sharding import Mesh
            from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
                SpmdFullyShardedDataParallel as FSDPv2,
            )

            num_devices = xr.global_runtime_device_count()
            mesh = Mesh(list(range(num_devices)), (num_devices, 1), ("fsdp", "model"))

            # DO NOT add Cohere2DecoderLayer to this policy. 36 per-layer wraps
            # => 36 bf16 reduce-scatters => NaN at step ~24-130. FSDPv2 has no
            # fp32_reduce_scatter (FSDPv1 only, #3588/#8056). One outer wrap.
            #
            # The policy matches on type(module).__name__: a stale class name
            # wraps NOTHING, silently falls back to replicated, and looks like a
            # successful FSDPv2 run. Verify the wrap count before trusting it.
            return FSDPv2(model, mesh=mesh, **kwargs)

        raise ValueError(
            f"Unknown TPU_STRATEGY {self._strategy!r}; "
            "expected 'replicated', 'fsdpv2', or 'fsdpv2_lora'"
        )

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # Under SPMD there is one process, so there is no gradient all-reduce to
        # perform here -- sharding handles it. `xm.optimizer_step` would add a
        # second, redundant reduction. Step, then flush the graph.
        optimizer.step()
        torch_xla.sync()

    def sync(self) -> None:
        torch_xla.sync()

    # --- data ---------------------------------------------------------------

    def make_sampler(self, dataset: Any, *, shuffle: bool, seed: int) -> Any | None:
        """Shard the dataset across HOSTS, so the epoch is covered exactly once.

        Multi-host (process_count > 1): each host takes 1/N of the data and
        decodes only its share -- for 558K JPEGs that is the difference between
        4x redundant decode and none.

        Single-host (process_count == 1): None. One process reads everything and
        the mesh shards it across the local chips.

        `drop_last=True` is NOT a preference. A short final batch is a different
        shape, and on XLA a different shape is a new HLO compile -- so without it
        every epoch ends in a recompile. It costs at most (batch-1) samples per
        host per epoch; measured at this config, 48 of 558,128 = 0.0086%.
        """
        if self.world_size <= 1:
            return None
        from torch.utils.data.distributed import DistributedSampler

        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,   # hosts, not chips
            rank=self.rank,                 # process_index, not worker number
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )

    def check_topology(self, *, global_batch: int) -> list[str]:
        """Startup guards. Returns a list of problems; empty means healthy.

        Every one of these failures is otherwise SILENT -- a mis-sharded run
        hangs or trains on a malformed global batch without raising. That
        asymmetry is the whole reason host-sharding needs guarding and
        replication does not.
        """
        problems: list[str] = []
        procs, chips = self.world_size, self.data_parallel_size

        if global_batch % chips:
            problems.append(
                f"global batch {global_batch} not divisible by {chips} chips -- "
                "the mesh cannot shard it evenly"
            )
        if global_batch % procs:
            problems.append(
                f"global batch {global_batch} not divisible by {procs} hosts -- "
                "per-host DataLoader batch would be fractional"
            )
        addressable = self.addressable_device_count
        if procs * addressable != chips:
            problems.append(
                f"topology inconsistent: {procs} processes x {addressable} "
                f"addressable chips != {chips} global chips"
            )
        # The minibatch contract, asserted at startup rather than discovered on
        # the first batch. torch_xla checks this itself in
        # `xla_model.send_cpu_data_to_device`, but only once data flows -- by
        # which point we have already paid model load and compile.
        if procs > 1 and not (global_batch % procs) and addressable:
            per_host = global_batch // procs
            if per_host % addressable:
                problems.append(
                    f"per-host batch {per_host} (= {global_batch}/{procs} hosts) "
                    f"not divisible by {addressable} local chips -- "
                    "ShardingSpec(minibatch=True) requires it"
                )
        return problems

    def wrap_loader(self, loader: Any) -> Any:
        """Feed the device AND shard the batch across chips.

        `input_sharding` is not an optimisation here, it is what makes this data
        parallel at all. Without it, `replicated` replicates the model to every
        chip and then hands every chip the WHOLE host batch -- 4 chips each doing
        identical work on 64 samples, instead of 16 samples each.

        That is how a v6e-16 OOM'd at global batch 256: the HLO showed
        `bf16[64,640,11008]` = 860 MB per MLP intermediate, i.e. the per-HOST
        batch of 64, not the per-chip 16. (11008 is Cohere2's FFN intermediate,
        5.38x the 2048 hidden size -- the dominant activation term.)

        The spec shards dim 0 (batch) over the mesh's `data` axis and replicates
        the rest, which is textbook SPMD data parallelism.

        `minibatch=True` IS MANDATORY HERE and is not a tuning knob. It is the
        flag that tells XLA the tensor this host just handed it is the host's
        SHARD of the global batch, not the global batch itself.

        We are host-sharded: `make_sampler` gives each of the N hosts a disjoint
        1/N of the dataset, so each host's loader yields `global/N` rows (64 of
        256 here). The mesh, however, spans all 16 GLOBAL chips -- it has to,
        because `ShardingSpec.__post_init__` derives its sharding type from
        `xr.global_runtime_device_count()`.

        With `minibatch=False` (the default) those two facts contradict each
        other: every host asserts that ITS 64 rows are the global tensor to be
        split across all 16 chips, and the four hosts disagree about what the
        global tensor is. Nothing raises -- 64 is divisible by 16, so every
        shape check passes -- and the program simply never makes progress.

        Measured 2026-07-25, v6e-16, tag deploy-20260725-092930: hung 2h39m with
        HBM allocated at 17.14 GiB/chip, duty cycle 0%, host CPU 1 jiffy per 5s,
        main thread in `futex_wait_queue`, zero traceback on all four hosts.
        That silence is the entire cost of this one keyword.

        With `minibatch=True`, torch_xla instead requires the per-host batch to
        divide by the LOCAL device count (`xla_model.py:1279`), i.e. 64 % 4 == 0
        -> 16 rows per chip, 256 global. `check_topology` asserts that up front.
        """
        import torch_xla.distributed.spmd as xs

        mesh = self._data_mesh()

        # Each entry of the batch dict is sharded on its batch dimension.
        # pixel_values is (B, C, H, W); the token tensors are (B, S).
        minibatch = self.world_size > 1
        spec = xs.ShardingSpec(mesh, ("data", None), minibatch=minibatch)
        img_spec = xs.ShardingSpec(
            mesh, ("data", None, None, None), minibatch=minibatch
        )
        return pl.MpDeviceLoader(
            loader,
            self.device,
            input_sharding={
                "input_ids": spec,
                "attention_mask": spec,
                "labels": spec,
                "pixel_values": img_spec,
            },
        )

    # --- checkpointing ------------------------------------------------------

    def save(self, obj: Any, path: Any) -> None:
        """Every rank transfers; only the global master writes.

        Deliberately NOT `xm.save`. That helper does

            should_write = master_only is False or is_master_ordinal(...)
            cpu_data = _maybe_convert_to_cpu(data, convert=should_write)
            if should_write: torch.save(cpu_data, path)

        i.e. it performs the device->host transfer ONLY on the writer
        (`xla_model.py:1246`). That is precisely the asymmetry that deadlocked
        this trainer once already: on SPMD the transfer is collective, so a rank
        that skips it leaves the others waiting forever. `xm.save`'s own
        docstring papers over this by telling the caller to follow it with
        `xm.rendezvous(...)`, which is a barrier -- and a barrier cannot repair
        a mismatch in the collective work itself.

        So: flush, convert on ALL ranks, write on one, meet again.
        """
        torch_xla.sync()
        xm.wait_device_ops()
        cpu_obj = _to_cpu(obj)
        if self.is_main:
            torch.save(cpu_obj, path)
        self.barrier()

    def _data_mesh(self):
        """1-D mesh over every chip, named `data`. Cached."""
        import numpy as np
        import torch_xla.distributed.spmd as xs

        if self._mesh is None:
            n = xr.global_runtime_device_count()
            self._mesh = xs.Mesh(np.arange(n), (n, 1), ("data", "model"))
        return self._mesh

    # --- diagnostics --------------------------------------------------------

    def memory_info(self) -> dict[str, float]:
        # xm.get_memory_info() is unusable under SPMD: it defaults to
        # xla_device(), which is the *virtual* device SPMD:0, and
        # _xla_memory_info rejects it. libtpu's metrics server works instead.
        try:
            from tpu_info import device as tpu_device
            from tpu_info import metrics as tpu_metrics

            chip_type, _count = tpu_device.get_local_chips()
            if chip_type is None:
                return {}
            usages = tpu_metrics.get_chip_usage(chip_type)
            return {
                "used_gib": max(u.memory_usage for u in usages) / 1024**3,
                "total_gib": max(u.total_memory for u in usages) / 1024**3,
            }
        except Exception:  # noqa: BLE001 - diagnostics must never break a run
            return {}
