# Tier definitions

The escalation ladder. Lower tiers are less disruptive and preferred when the failure
mode is well understood.

**This table splits the two source policies at T3.** `tinyaya-stage2-scale` locked T3 to
"always escalate — never auto-recreate the QR". `llm-architectures` inverted it and
auto-recycles. Here T3 is automatic **only when checkpoints are durable** — see the
`gs://` gate below and `../SPEC.md` §5.

| Tier | Action | Trigger | Auto? | Disruption |
|------|--------|---------|-------|------------|
| **T0** | Continue | All signals nominal; step lines advancing, compile-cause count flat | yes | none |
| **T1** | Wait it out | Known-benign stall: XLA compile (**budget 20 min cold, not 10**), checkpoint write, first-batch dataset materialization | yes | none |
| **T2** | Redeploy with patch | Classified, fixable error from the diagnosis table (config/env/code) | yes | ~2 min tarball + relaunch, **plus a full recompile** |
| **T3** | Recycle the queued resource | Node preempted, `SUSPENDED`/`FAILED`, or hosts unreachable | **conditional** — `qr_watch.sh` recycles iff `SAVE_CKPT_DIR` is a `gs://` URI; otherwise notify-only | ~10-30 min: new spot acquisition, then boot self-heal |
| **T4** | Stop and notify | Budget spent, quota-class abort, repeat classification, non-durable checkpoints, or unknown signature | n/a | indefinite — needs a human |

**T2 costs more here than on the JAX sibling.** A redeploy on torch_xla throws away the
XLA executable cache, and that cache has nondeterministic keys on v6e SPMD so it never
warm-hits. Budget ~14 min of recompile per T2, not the ~1 min a JAX redeploy costs. Batch
your fixes: two T2s in a row for two separate one-line patches is half an hour of silicon
spent on compilation.

## Auto-escalation to T4

Stop and push a notification regardless of tier when:

1. The same classification fires twice consecutively (circuit breaker — the patch is not
   working, and here each retry burns ~14 min of compile with it).
2. `MAX_RESUBMITS` (20) is exhausted.
3. The QR failure is quota-class. On this grant the commonest such wall is
   `IN_USE_ADDRESSES` (regional cap 8, and every 8-host slice wants exactly 8), not TPU
   chips.
4. **`SAVE_CKPT_DIR` is not a `gs://` URI.** Recycling without durable checkpoints
   restarts from step 0 and re-pays the compile, silently, up to twenty times.
5. The failure signature is not in the diagnosis table.
6. A second resubmitter is detected for the same QR.
7. **Loss goes non-finite.** NaN on this backbone has a known structural cause
   (`../SPEC.md` §7, FSDPv2 per-layer reduce-scatter); it is never a "retry and see"
   condition.

## Locked invariants

- **One resubmitter per QR, ever.** `qr_watch.sh` exits if the QR is absent rather than
  racing a human mid-recreate. Two resubmitters produce duplicate nodes and split-brain
  rendezvous.
- **T3 auto-recycle is bounded**, never unbounded. Budget plus cooldown is what separates
  self-healing from a flapping loop that burns a day of capacity.
- **T3 auto-recycle requires durable checkpoints.** This is the tayavision-specific brake.
  Self-healing that discards all progress is not self-healing.
- **T2 redeploys are only safe mid-run when checkpointing and resume are configured.** A
  redeploy kills `train` on every worker; without `SAVE_CKPT_DIR` +
  `TAYAVISION_RESUME=auto` that discards progress.
- **T1 is a real tier, not a stalling tactic.** On torch_xla most apparent hangs are a
  compile. Capture the compile-cause counter from `met.metrics_report()` as evidence
  before escalating anything that looks like a stall — elapsed time alone cannot tell a
  14-minute normal boot from a deadlock.

## Classification source

The signature → classification table is canonical in `.claude/agents/tpu-diagnoser.md`.
This file owns *policy* (what to do at each tier); that file owns *detection* (which tier
a log belongs to).
