"""Control-plane smoke entrypoint — the default TAYAVISION_ENTRYPOINT.

This is NOT training. `pipeline/train_*.py` is DDP + CUDA and does not run under
torch_xla; see `.claude/orchestration/SPEC.md` section 9.

What it does prove, on real silicon:
  * torch_xla imports (i.e. the libpython / _XLAC.so link resolved)
  * chips are visible and a mesh can be built
  * bf16 tensors of realistic size fit, and what the HBM headroom is
  * the XLA compile-cause counters behave (flat after warmup)
  * the log markers every watcher greps for are emitted in the right format

With --load-backbone it additionally downloads and materializes the real model,
which exercises the gated-HF-repo path (`HF_TOKEN` reaching the VM inside the
tarball's `.env`) and reports the true memory envelope. Those are the two things
most likely to be wrong on a first boot.

Deliberately module-level `import torch_xla`: this file lives under `scripts/`,
outside the trees `scripts/ci/check_backend_seam.sh` guards.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

STEPS = 20


def hbm_report() -> list[str]:
    """Per-chip HBM and duty cycle, or an honest explanation of why not.

    `xm.get_memory_info()` defaults to `xla_device()`, which under SPMD is the
    *virtual* device `SPMD:0`, and `_xla_memory_info` rejects it outright:
    "MemoryInfo not supported for SPMD virtual device". Passing
    `torch_xla.devices()` does not help -- under SPMD that returns the same
    virtual device. Both were tried; both fail.

    Two things that do work, in preference order:

    1. `tpu_info` (installed as a torch_xla dependency) reads libtpu's gRPC
       metrics server directly. SPMD-agnostic, per-chip, and it also yields
       duty cycle -- which is what tells a compile apart from a hang
       (tpu-diagnoser row 15).
    2. `xm.get_memory_info()` against the *real* runtime device strings from
       `_xla_get_all_runtime_devices()`, which are not the SPMD virtual device.

    This is the logic `src/backend/base.py::memory_info()` will need when the
    seam lands (PLAN P6); keep it portable.
    """
    def gib(b: int) -> str:
        return f"{b / 1024**3:.3f} GiB" if b >= 1024**3 else f"{b / 1024**2:.1f} MiB"

    out: list[str] = []
    total_b = used_b = 0
    src_total = src_used = "none"
    detail = ""

    # Source 1: libtpu's gRPC metrics server, via tpu_info.
    try:
        from tpu_info import device as tpu_device
        from tpu_info import metrics as tpu_metrics

        chip_type, count = tpu_device.get_local_chips()
        if chip_type is not None:
            usages = tpu_metrics.get_chip_usage(chip_type)
            total_b = max(u.total_memory for u in usages)
            src_total = "tpu_info"
            # 1 MiB floor separates "nothing resident" from a real measurement.
            # The gauge DOES track program allocations -- it read 8.058 GiB with
            # the 4.33B model resident (2026-07-25), matching 4.33e9 x 2 bytes
            # exactly. An earlier reading of ~0 with `--alloc-gib 8` was a flawed
            # test, not a broken gauge: a `torch.zeros` that no computation
            # consumes gets dead-code-eliminated by XLA before it is ever
            # allocated. Loaded weights cannot be elided, which is why they show.
            u_b = max(u.memory_usage for u in usages)
            if u_b >= 1024**2:
                used_b, src_used = u_b, "tpu_info"
            duty = max(u.duty_cycle_pct for u in usages)
            detail = (f"{len(usages)} core(s), {count} local chip(s) [{chip_type}]")
            out.append(f"duty_cycle: max {duty:.1f}% across chips")
    except Exception as exc:  # noqa: BLE001 - diagnostic only, never fatal
        out.append(f"note: tpu_info unavailable ({type(exc).__name__}: {exc})")

    # Source 2: XLA's own allocator view, per REAL runtime device. Not a plain
    # fallback -- it is consulted whenever libtpu reported usage 0, because on
    # torch_xla 2.9 / libtpu 0.0.21 the MEMORY_USAGE gauge reads 0 even with
    # 8 GiB pinned on device (verified 2026-07-25), while TOTAL_MEMORY and
    # duty cycle are correct. Two half-working sources, merged.
    try:
        import torch_xla as _txla
        import torch_xla.core.xla_model as _xm

        real = _txla._XLAC._xla_get_all_runtime_devices()
        infos = [_xm.get_memory_info(d) for d in real]
        x_used = max(i["bytes_used"] for i in infos)
        x_total = max(i["bytes_limit"] for i in infos)
        if used_b == 0 and x_used > 0:
            used_b, src_used = x_used, "xla"
        if total_b == 0 and x_total > 0:
            total_b, src_total = x_total, "xla"
        if not detail:
            detail = f"{len(real)} runtime device(s)"
    except Exception as exc:  # noqa: BLE001 - diagnostic only, never fatal
        out.append(f"note: xla memory_info unavailable "
                   f"({type(exc).__name__}: {str(exc)[:110]})")

    if total_b and used_b:
        out.insert(0, f"hbm: peak {gib(used_b)} of {gib(total_b)} per chip — {detail} "
                      f"(total via {src_total}, used via {src_used}; "
                      f"working ceiling 29 GiB on v6e)")
    elif total_b:
        out.insert(0, f"hbm: {gib(total_b)} per chip — {detail} (via {src_total}), "
                      f"under 1 MiB resident. Expected when nothing large is live: "
                      f"XLA elides allocations no computation consumes, so a "
                      f"synthetic buffer may never exist. Run --load-backbone for a "
                      f"real figure. Working ceiling 29 GiB.")
    else:
        out.insert(0, "hbm: no source reported chip memory")
    return out


def check_backend() -> int:
    """Acceptance test for src/backend/ on real TPU silicon.

    Exercises every abstract method. The two that matter most are asserted
    rather than merely printed:

      * data_parallel_size must be the CHIP count, not the process count. Under
        SPMD world_size is 1; deriving a per-device batch from it would hand
        every chip the full global batch.
      * make_sampler MUST return None. SPMD is one process, so a
        DistributedSampler would silently train on 1/N of the dataset.
    """
    import torch

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
    from src.backend import create_backend

    b = create_backend()
    b.setup()
    print(f"backend: {b.describe()}", flush=True)

    failures: list[str] = []

    if b.name != "tpu":
        failures.append(f"expected the tpu backend, got {b.name!r} "
                        "(is TAYAVISION_TPU=1 set?)")

    dp = b.data_parallel_size
    print(f"  data_parallel_size = {dp} (world_size = {b.world_size})", flush=True)
    if dp <= 1:
        failures.append(f"data_parallel_size is {dp}; expected the chip count")
    if b.world_size != 1:
        failures.append(f"SPMD world_size should be 1, got {b.world_size}")

    sampler = b.make_sampler(object(), shuffle=True, seed=0)
    print(f"  make_sampler -> {sampler!r}", flush=True)
    if sampler is not None:
        failures.append("make_sampler must return None under SPMD; a "
                        "DistributedSampler would train on 1/N of the data")

    # A real parameter, a real backward, a real optimizer step through the seam.
    lin = torch.nn.Linear(2048, 2048, dtype=torch.bfloat16).to(b.device)
    lin = b.wrap_model(lin)
    print(f"  wrap_model -> {type(lin).__name__}", flush=True)
    opt = torch.optim.SGD(lin.parameters(), lr=1e-4)
    with b.autocast(torch.bfloat16):
        out = lin(torch.randn(4, 196, 2048, device=b.device, dtype=torch.bfloat16))
        loss = out.float().mean()
    loss.backward()
    b.optimizer_step(opt)
    b.sync()
    print(f"  autocast + backward + optimizer_step ok (loss={loss.item():.6f})", flush=True)

    b.manual_seed(1234)
    b.barrier()
    mem = b.memory_info()
    print(f"  memory_info -> {mem}", flush=True)
    if not mem.get("total_gib"):
        failures.append("memory_info reported no total_gib")

    b.cleanup()

    if failures:
        for f in failures:
            print(f"FAIL: {f}", flush=True)
        return 1
    print("backend-check-ok: every seam method exercised on TPU", flush=True)
    print("Reached maximum training steps", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument(
        "--load-backbone",
        action="store_true",
        help="also materialize the real model (needs an authorised HF_TOKEN)",
    )
    ap.add_argument(
        "--alloc-gib",
        type=float,
        default=0.0,
        help=(
            "hold N GiB of bf16 on device for the whole run. Use it to prove the "
            "HBM probe registers real memory: the default matmul is a few MB "
            "sharded 16 ways, which correctly reads as 0.0 MiB per chip and so "
            "cannot distinguish 'probe works' from 'probe broken'."
        ),
    )
    ap.add_argument(
        "--check-backend",
        action="store_true",
        help=(
            "exercise every method of src/backend/ on real silicon. This is the "
            "seam's acceptance test: a full pipeline/train_alignment.py run needs "
            "the 13 GB LLaVA-Pretrain corpus staged, but the backend contract can "
            "and should be validated without it."
        ),
    )
    args, _unknown = ap.parse_known_args()

    print("tayavision tpu_smoke: control-plane check, NOT training", flush=True)

    if args.check_backend:
        return check_backend()

    try:
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met
        import torch_xla.runtime as xr
    except ImportError as exc:
        # The single most likely first-boot failure. Diagnoser row 1.
        print(f"FATAL: torch_xla import failed: {exc}", flush=True)
        print("       If this is libpython3.12.so.1.0, LD_LIBRARY_PATH was not set —", flush=True)
        print("       see scripts/tpu/startup_script.sh step 5.", flush=True)
        return 1

    xr.use_spmd()
    devices = xm.get_xla_supported_devices()
    n = xr.global_runtime_device_count()
    print(f"torch_xla {torch_xla.__version__} | devices={len(devices)} global={n}", flush=True)
    print(f"mesh: 1-D over {n} chips (SPMD, single process)", flush=True)

    dev = torch_xla.device()  # xm.xla_device() is deprecated in 2.9

    resident_model = None
    if args.load_backbone:
        print("loading the real model (gated HF repos — needs HF_TOKEN)", flush=True)
        # Running as `python scripts/tpu/tpu_smoke.py` puts scripts/tpu/ on
        # sys.path, not the repo root, and this project is not pip-installed into
        # the venv. `pipeline/train_*.py` all carry the same insert for the same
        # reason. Without it: ModuleNotFoundError: No module named 'config'.
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
        try:
            from config.model_config import TinyAyaVisionConfig
            from models.tiny_aya_vision import TinyAyaVisionForConditionalGeneration

            cfg = TinyAyaVisionConfig.for_base()
            model = TinyAyaVisionForConditionalGeneration(cfg).to(dev, dtype=torch.bfloat16)
            # torch_xla is LAZY: `.to(dev)` only records the transfer. Without an
            # explicit sync the weights never reach HBM, and a naive `del model`
            # afterwards means the fit was never actually tested -- the first
            # version of this reported 18.4 MiB peak for a 4.33B model.
            torch_xla.sync()
            n_params = sum(p.numel() for p in model.parameters())
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"params total={n_params/1e9:.2f}B "
                  f"trainable={n_train/1e9:.2f}B ({100*n_train/n_params:.0f}%)", flush=True)
            print("NOTE: the bare constructor freezes only the vision encoder. Phase-1's "
                  "~11.5M-trainable figure comes from the training pipeline freezing the "
                  "LLM, not from the model class.", flush=True)
            # Deliberately kept resident: `model` must stay alive through the
            # HBM samples below, or this measures nothing.
            resident_model = model
        except OSError as exc:
            print(f"FATAL: gated repo — {exc}", flush=True)
            print("       .env must carry an authorised HF_TOKEN and reach the VM", flush=True)
            print("       inside the code tarball. Diagnoser row 2.", flush=True)
            return 1

    ballast = None
    if args.alloc_gib > 0:
        n = int(args.alloc_gib * 1024**3 / 2)  # bf16 = 2 bytes
        ballast = torch.zeros(n, device=dev, dtype=torch.bfloat16)
        torch_xla.sync()
        print(f"ballast: holding {args.alloc_gib:.2f} GiB bf16 on device", flush=True)

    # Shaped like the real projector output: (batch, 196 image tokens, d=2048).
    x = torch.randn(8, 196, 2048, device=dev, dtype=torch.bfloat16)
    w = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)

    step_times = []
    for step in range(1, args.steps + 1):
        # Perturb the input each step. With a constant input XLA caches the
        # result and every step after the first reports the enqueue time, not
        # the compute time -- the first version of this script printed
        # step_time=0.0001 and an identical loss for 18 straight steps.
        # The SHAPE never changes, so this costs no recompile.
        x.add_(0.001)

        t0 = time.monotonic()
        y = (x @ w).relu()
        loss = y.float().mean()
        torch_xla.sync()
        # sync() flushes the graph but does not block on completion. Reading the
        # value does. Without this the timer measures lazy enqueue.
        loss_val = loss.item()
        dt = time.monotonic() - t0

        step_times.append(dt)
        # Format matched by tpu-watchdog and the WATCH loop in SPEC.md section 4.
        print(f"step={step} loss={loss_val:.6f} step_time={dt:.4f}", flush=True)

        # Sample mid-run, while tensors are resident and the device is busy.
        # Sampling only at the end reports duty_cycle 0% every time, which reads
        # like an idle chip -- the exact signature tpu-diagnoser row 15 keys on.
        if step == args.steps // 2:
            for line in hbm_report():
                print(f"[mid-run] {line}", flush=True)

    if ballast is not None:
        del ballast  # released only after the final HBM sample
    if resident_model is not None:
        del resident_model  # ditto: the model had to survive the HBM samples

    warm = sorted(step_times[3:]) or step_times
    p50 = warm[len(warm) // 2]
    print(f"step_time: first={step_times[0]:.4f} p50_warm={p50:.4f} "
          f"(compile is in the first 2-3 steps)", flush=True)

    for line in hbm_report():
        print(line, flush=True)

    report = met.metrics_report()
    print(f"compile-cause markers in report: {report.count('CompileTime')}", flush=True)
    # NOTE: this script calls .item() every step by design (see above), so
    # aten::_local_scalar_dense is EXPECTED here and is not a finding. Only
    # aten::nonzero indicates the dynamic-shape problem the model actually has.
    if "aten::nonzero" in report:
        print("WARNING: aten::nonzero present — dynamic-shape fallback. Diagnoser row 11.",
              flush=True)

    print("Reached maximum training steps", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
