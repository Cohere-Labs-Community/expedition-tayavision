# TPU capacity log -- observed queue times + autonomous fallback policy

> **Provenance.** Entries dated before 2026-07-25 were recorded by sibling projects
> (`llm-architectures`, `tinyaya-stage2-scale`) running on the **same TRC grant and the
> same GCP project `ml-pipelines-315702`**.
>
> What transfers exactly: quota behaviour, queue times, zone availability, the
> `IN_USE_ADDRESSES` cap, QR-husk accounting, and spot-churn frequency. These are
> properties of the grant, not of a model.
>
> What does **not** transfer: throughput, step time, and HBM figures. Those belong to
> other models. Compile-time observations sit in between — the ~14 min cold-boot
> recompile in §8 is a property of torch_xla on v6e SPMD and is expected to hold here,
> but it has not been measured for Tiny Aya Vision.
>
> The first Tiny Aya Vision TPU run happened 2026-07-25 on a v6e-16; its numbers live in
> `.claude/orchestration/playbook/baseline-v6e8-siglip-cohere2.md`, not this file.

## 2026-07-26 — large-slice capacity in `europe-west4-a` is a hard wall, not a queue

Five attempts across ~10 hours, zone clean before each (zero QRs, zero nodes, no husks),
`IN_USE_ADDRESSES` measured at 16/64 so **IP quota was never the binding constraint**:

| Time (UTC) | Profile | Outcome |
|---|---|---|
| 2026-07-25 10:46 | `v6e-64-ew4a` | FAILED, code 13, ~5 min |
| 2026-07-25 ~11:0x | `v6e-32-ew4a` | FAILED, code 13 |
| 2026-07-25 ~11:2x | `v6e-16-ew4a` | **ACTIVE** — ran a full 2,180-step epoch |
| 2026-07-26 02:36 | `v6e-64-ew4a` | FAILED, code 13, ~4.5 min |
| 2026-07-26 02:42 | `v6e-64-ew4a` | FAILED, code 13, ~3.5 min |
| 2026-07-26 02:46 | `v6e-32-ew4a` | FAILED, code 13, ~4.7 min |

Every failure is byte-identical in shape: `WAITING_FOR_RESOURCES → PROVISIONING →
SUSPENDING → FAILED`, `{"code": 13, "message": "an internal error has occurred"}`.

**Read `code 13` as "no capacity for this slice size", not as a bug to debug.** The
service reaches `PROVISIONING` — it accepts the request and starts allocating — then
withdraws. There is no error text to act on and no queue position to wait in: a failed QR
does **not** stay queued, it terminates, so "wait longer" is not a strategy. Retrying is
cheap (~5 min) and occasionally worth one shot, but three v6e-64 attempts and two v6e-32
attempts across ten hours all failed while v6e-16 succeeded twice.

**Practical rule for this grant, this zone: 16 chips is the ceiling that actually
provisions.** Anything larger needs either a different zone (cross-region, pays egress
against the EU bucket) or the v5e family, whose podslice quota does exist here
(`TPU_LITE_PODSLICE_V5: usage=0 limit=16` in `europe-west4`) but which has 16 GiB HBM per
chip against v6e's 31.25 — a change the batch config would have to absorb, since the
measured run already sits at 27.8 GiB/chip.

## 2026-07-25 — Tiny Aya Vision first QR (v6e-64-ew4a)

First provisioning attempt for this project. Recorded here because it corrects a
claim this file has been carrying.

| Fact | Value |
|---|---|
| Profile | `v6e-64-ew4a` (v6e-64, 8 hosts, `europe-west4-a`) |
| QR | `tayavision-v6e64-qr`, spot |
| Zone state before | **clean**: zero QRs, zero nodes, no husks |
| 10:46 | submitted, accepted → `WAITING_FOR_RESOURCES` |
| 10:48 | `PROVISIONING`, node `CREATING` — capacity was allocated |
| 10:50 | `SUSPENDING`, `stateInitiator: SERVICE` |
| 10:51 | `FAILED` — `code 13, "an internal error has occurred"` |
| 10:52 | husk deleted, zone clean again |
| **Total** | **~5 min, PROVISIONING → FAILED** |

### This falsifies section 7.1's hypothesis

The 2026-05-05 rows attribute the identical failure — `PROVISIONING → SUSPENDING →
FAILED in <5 min`, code 13 — to the **`IN_USE_ADDRESSES` cap of 8**: "8-host slice + 8
IP cap = identical IP quota gate."

That explanation is wrong, or at least no longer holds. Measured at submit time,
**`europe-west4` `IN_USE_ADDRESSES` was usage 16 / limit 64** — 48 free, against the 8
an 8-host slice needs. The quota was raised at some point since May. The failure
reproduced anyway, unchanged, with ample IP headroom.

So `code 13 / "an internal error has occurred"` on a 64-chip v6e spot QR is **not an IP
quota gate**. It is GCP surfacing something else — most plausibly genuine capacity
unavailability for a slice that large, reported through an opaque error rather than
`WAITING_FOR_RESOURCES`. Note the QR reached `PROVISIONING` and the node reached
`CREATING` first, so it is not a submission-time refusal either: capacity was briefly
allocated and then taken back by the service.

**Operational consequence:** do not spend time raising IP quota to unblock v6e-64. It is
already unblocked and still fails. Treat 64-chip v6e spot as un-gettable on this grant
until proven otherwise, which matches the 2026-07-06 note ("v6e-64 spot proved
un-gettable on both zones") — that observation was right even though the reason given
for it was not.

### Other notes from this attempt

- **No `TPU_V6E_*` metric exists in the regional compute quotas at all.** v6e capacity is
  governed by the TPU API and the TRC grant, not by a compute quota row. "Check the
  quota" cannot answer "will I get a v6e-64"; only submitting does.
- The control plane behaved correctly end to end: tarball upload, QR create with
  self-heal metadata, cross-region check (silent, correctly — this is an EU profile),
  durability gate satisfied via `SAVE_CKPT_DIR=gs://…`, and forensics captured on
  failure.
- `gs://tayavision-eu` was created by `setup_gcp.sh` in this session; it did not exist
  before.

## 2026-07-08 update

**v6e-8 spot in ew4a is readily gettable** (three same-day QRs went
WAITING→PROVISIONING→ACTIVE in ~2-5 min each: smoke-scan2, smoke-scan3 after
quota freed) BUT the 64-chip preemptible quota counts **SUSPENDED/FAILED QR
husks too**: with 3 arms + 2 dead smokes + 1 failed arm QR still registered,
new creates 429'd (`TPUV6EPreemptiblePerProjectPerZoneForTPUAPI exhausted`)
even though only 32 chips were live. Deleting the husks un-jammed launches
immediately. Same-day preemption observed on a v6e-8 (~2h lifetime,
smoke-scan) -- plan for spot churn on multi-hour arms (`--resume auto` +
GCS-staged data tarball keep recovery cheap).

## 2026-07-06 update

Current topology: **v6e-16 spot in `europe-west4-a`** (production) + **v6e-8** (smoke /
overfit / eval), both from the v6e spot quota. v6e-64 spot proved un-gettable on both
zones; v4-32 (`us-central2-b`) and v5e are legacy. The fallback policy in section 1 below
is preserved as the autonomous decision tree for new capacity attempts. See
[`tpu-trc-allocation.md`](tpu-trc-allocation.md) for the grant table and
[`tpu-runbook.md`](tpu-runbook.md) for launch.

**Purpose:** Record real queue-wait durations so future sessions can
make smart autonomous decisions about which TRC slice to try. Updated
after every QR submission attempt.

The authoritative TRC quota table lives in
[`docs/tpu-trc-allocation.md`](./tpu-trc-allocation.md); this file
augments it with *observed* capacity behaviour.

---

## 1. Autonomous fallback policy

This section is **canonical** for zone selection.
[`tpu-trc-allocation.md`](tpu-trc-allocation.md) owns the grant table and points here
rather than carrying a second copy of this tree — two copies drift, and a stale fallback
policy sends you to a zone that costs money.

**Reordered 2026-07-25 for Tiny Aya Vision.** The previous order tried the US zones
early. That was correct when this policy served a project with `-us` and `-usc1` sibling
buckets. This project has **one bucket, `gs://tayavision-eu` in `europe-west4`**, so a US
zone now pays egress on every checkpoint write and every code-tarball pull.

```text
Start: TRC_PROFILE=v6e-8-eu  (single host, co-located, no external-IP pressure)
       Wait up to 15 min.
+-- ACTIVE -> proceed.
+-- Still WAITING_FOR_RESOURCES:

1. Run `ops.sh status` and DELETE SUSPENDED/FAILED QR husks first.
   Husks book quota while dead. Observed 2026-07-08: 3 arms + 2 dead
   smokes + 1 failed arm 429'd new creates at only 32/64 chips live.
   Deleting the husks un-jammed launches immediately.
   Then retry v6e-8-eu once, after a 10 min gap.
+-- Still nothing:

2. Need more chips?
   -> v6e-64-ew4a  or  v5e-64-ew4b   (both EU, both co-located)
      CHECK IN_USE_ADDRESSES FIRST -- see 7.1 for the failure it causes.
      MEASURED europe-west4 2026-07-25: usage=16 limit=64 -> 48 free.
      An 8-host slice needs 8. NOT a blocker in this region today,
      contrary to what 7.1's original entry implies.
      Multi-host still adds PJRT rendezvous to the failure surface.
+-- Not willing, or still nothing:

3. Cross-region. A DELIBERATE CHOICE, not a default.
   -> v6e-64-ue1d / v5e-64-uc1a / v4-32-uc2b
      Either accept the egress for a short run, or stand up a
      region-paired bucket and point SAVE_CKPT_DIR at it.
      launch_spot.sh warns; it does not block.
+-- Nothing anywhere:

4. Stop and wait. TRC spot has no escalation path, and churning QRs
   loses your place in the queue. Do NOT auto-delete a QR the user
   may want left queued.
```

**Important rules:**
- Never submit more than one QR at a time. A queued QR still
  consumes quota; two parallel QRs for the same type will both
  block.
- Delete the failed QR before submitting the next one.
- Record every attempt in section 2 below (timestamp, profile, wait
  duration, outcome).
- After 3 consecutive failures in the same day, stop and ask
  the user rather than burning the retry budget.

## 2. Observed queue durations

| Date | Time (UTC) | Profile | Tier | Wait | Outcome | Notes |
|---|---|---|---|---|---|---|
| 2026-05-03 | ~12:00 | v4-64 on-demand (us-central2-b) | on-demand | <5 min | ACTIVE | Original probe run; QR provisioned quickly. |
| 2026-05-03 | ~14:00 | v4-64 on-demand (us-central2-b) | on-demand | ~0 | ACTIVE | Real composite fsdpv2_lora compile; QR already up from earlier. |
| 2026-05-05 | 09:23 | v4-32 spot (us-central2-b) | spot | 17+ min | WAITING | Cancelled after 17 min; no progress. Spot v4 in this zone is contended. |
| 2026-05-05 | 09:46 | v4-64 on-demand (us-central2-b) | on-demand | 10+ min | WAITING/CANCELLED | Cancelled after 10 min per fallback policy timeout; on-demand v4 is also genuinely contended in this zone today. |
| 2026-05-05 | 12:45 | v5e-64 spot (europe-west4-b) | spot | 2 min PROVISIONING | FAILED | `IN_USE_ADDRESSES limit (8)` -- regional IP quota too small for 8-host v5e-64 with external IPs. Not a TPU capacity issue. Mitigations: (a) request IP quota bump, (b) use `--internal-ips` + Private Google Access + GCS-mirrored dataset. |
| 2026-05-05 | 13:02 | v6e-64 spot (us-east1-d) | spot | 5 min PROVISIONING | FAILED | Code 13 "an internal error has occurred" -- same failure pattern as v5e-64 ew4b (PROVISIONING -> SUSPENDING -> FAILED in <5 min). 8-host slice + 8 IP cap = identical IP quota gate. |
| 2026-05-05 | 13:42 | v4-32 spot (us-central2-b) | spot | 3.5 min PROVISIONING | ACTIVE | Retry of the 09:23 attempt with same config + same QR submission. Spot pool cleared in the intervening hours; ACTIVE reached at 13:51 UTC. First successful canary launch of the day. |
| 2026-05-08 | -- | v4-32 spot (us-central2-b) | spot | hours | SUSPENDED | v4 spot pool reclaimed by other TRC users; QR has been SUSPENDED for several hours with no recovery. Pivoted away. |
| 2026-05-08 | -- | v6e-8 spot (europe-west4-a) | spot | ~4:44 | ACTIVE | Single-host (8 chips), 32 GiB HBM/chip. ACTIVE within 5 min from QR submission. Iter 13b (run `zd42n7di`) reached 20 steps + canonical save in 23.3 min wall. |
| 2026-05-09 | 15:25 | v6e-8 spot (europe-west4-a) | spot | ~3 min to ACTIVE | ACTIVE / COMPLETED | Iter 24h production retry. QR reached ACTIVE at 15:28 UTC; run `7rrjupc7` completed 5000/5000 steps at 2026-05-10T01:47:24Z and uploaded `step_005000_final` (8 objects, 2.37 GiB). |
| 2026-05-13 | ~11:30 | v6e-8 spot (europe-west4-a) | spot | immediate preemptions / capacity retry | FAILED | Phase 4 `opt-4-depth32` first attempts hit spot preemption and then QR capacity error code 8. User chose to keep retrying same zone. |
| 2026-05-13 | ~12:00 | v6e-8 spot (europe-west4-a) | spot | ~few min to ACTIVE | ACTIVE / COMPLETED | Relaunched Phase 4 with `REPO_TARBALL_GS_URI=gs://tinyaya-stage2-eu/code/phase4-depth32-20260513T120005Z.tar.gz`; startup avoided private GitHub clone and W&B `i15igq8d` completed 300/300 steps with exit 0. |

*(Update this table after every attempt.)*

## 3. Per-profile heuristics

Based on the limited data so far:

| Profile | Expected wait | Confidence | Notes |
|---|---|---|---|
| v4-64 on-demand (uc2b) | 0-15 min | LOW (1 success, 1 pending) | Succeeds when no other TRC user in this zone has the on-demand quota occupied. |
| v6e-8 spot (ew4a) | 0-10 min when capacity is available; retry after preemption/capacity code 8 | HIGH | Current validated production/optimization profile. One host = one external IP, fits quota, no multi-host rendezvous. |
| v4-32 spot (uc2b) | 0-20 min, hour-by-hour | MEDIUM (1 fail at 09:23, 1 success at 13:42) | Legacy fallback. Spot pool clears on a sub-hour timescale. Cancel-and-retry within a few hours is a valid strategy. 4 hosts = 4 IPs, fits the 8-IP regional cap easily. |
| v5e-64 spot (ew4b) | 2 min to PROVISIONING then FAILED | LOW (1 IP-quota fail) | Hits `IN_USE_ADDRESSES` regional quota (8) because 8-host slice needs 8 external IPs. Use `--internal-ips` (requires Private Google Access on subnet + GCS-mirrored dataset) or request IP quota bump. |
| v5e-64 spot (uc1a) | unknown | NONE | Never tried. |
| v6e-64 spot (ue1d) | 5 min PROVISIONING then FAILED | LOW (1 IP-quota fail) | Same root cause as v5e-64: 8-host slice + regional 8-IP cap. Code 13 ("internal error") instead of explicit IP-quota message, but identical failure pattern and timing. |

**Actionable insight:** v5e-64 in europe-west4-b is the strongest
fallback candidate because (a) 64 chips is the biggest spot grant we
have, (b) europe region may have lower TRC competition than us-central2,
(c) the canary config is already retuned for v5e HBM/chip constraints.
Try it THIRD (after on-demand v4 and spot v4).

## 4. Session checklist for autonomous QR launch

Before submitting any QR, a droid session must:

- [ ] `bash scripts/tpu/ops.sh preflight` -- gcloud auth, `.env` keys, entrypoint,
  bucket reachability. **`HF_TOKEN` is not optional**: `CohereLabs/tiny-aya-*` is gated
  and the run cannot start without it.
- [ ] Read this file for the latest observed wait times.
- [ ] Read [`tpu-trc-allocation.md`](tpu-trc-allocation.md) for the quota table.
- [ ] `bash scripts/tpu/ops.sh status` -- confirm no existing QR, and **delete
  SUSPENDED/FAILED husks**; they book quota while dead.
- [ ] Pick the first profile from the decision tree in section 1 (starts at
  `v6e-8-eu`).
- [ ] Submit the QR, start a 15-min poller.
- [ ] If ACTIVE: proceed with `TAYAVISION_ENTRYPOINT=scripts/tpu/tpu_smoke.py`.
  **Not a training entrypoint** -- `pipeline/train_*.py` is DDP + CUDA and does not run
  on TPU yet. See `.claude/orchestration/SPEC.md` §9.
- [ ] If still WAITING: delete the QR, advance to the next branch, update section 2
  with the observed wait.
- [ ] After 3 failures: stop and ask the user.
- [ ] When done: `bash scripts/tpu/ops.sh delete`. A live slice bills whether or not
  anything is running on it.

## 5. Cross-references

- [`tpu-trc-allocation.md`](tpu-trc-allocation.md) -- authoritative TRC quota table.
- [`tpu-runbook.md`](tpu-runbook.md) -- the operator command sequence.
- `scripts/tpu/launch_spot.sh` -- the `TRC_PROFILE`-aware launch wrapper; warns on
  cross-region profiles against the single `gs://tayavision-eu` bucket.
- `scripts/tpu/ops.sh` -- `preflight`, `status`, `tail-logs`, `attach`, `delete`.
- `.claude/orchestration/SPEC.md` -- the run loop, the T3 recycle policy, and §9 on
  what does not work yet.
- `.claude/agents/tpu-diagnoser.md` -- signature to tier classification.
- .factory/memories.md -- decision entry for the fallback policy.

## 6. Maintenance

Prune entries older than 90 days from section 2 during the monthly
archive-progress cycle. The heuristics in section 3 should be updated
whenever a new profile is tried (success or failure).

## 7. Known issues + workarounds

### 7.1 `IN_USE_ADDRESSES limit` on v5e/v6e slices

**Symptom:** QR transitions `WAITING_FOR_RESOURCES` -> `PROVISIONING`
-> `SUSPENDING` -> `FAILED` within 2 minutes with error
`You have reached IN_USE_ADDRESSES limit. [EID: ...]`.

**Cause:** Regional Compute Engine `IN_USE_ADDRESSES` quota is **8**
in every region we have TPU quota in (europe-west4, us-central1,
us-central2). v5litepod-64 (8 hosts) and v6e-64 (8 hosts) want one
external IP per host = 8 IPs total, hitting the limit exactly.
v4-64 (4 hosts? or 8 hosts depending on topology) is right at the
edge. Even momentary other usage (a Cloud Build worker, a Compute
VM) pushes us over.

**Workaround A (recommended):** Use `INTERNAL_IPS=1` with
`launch_qr.sh` (requires the `--internal-ips` flag plumbing added
2026-05-05). Prerequisites:
  1. Private Google Access enabled on `default` subnet in the target
     region (`gcloud compute networks subnets update default
     --region=<R> --enable-private-ip-google-access`).
  2. Dataset mirrored to `gs://tinyaya-stage2-eu/encoded/` so the
     startup script can pull it via GCS instead of HF Hub.
  3. Code tarball already lives in GCS (already done: `code/tinyaya-repo-hot.tar.gz`).

**Workaround B:** Request a regional IP quota bump from GCP support
(typical turnaround 1-3 business days).

**Workaround C:** Pick a smaller slice (v4-32 = 4 hosts = 4 IPs;
stays under quota) at the cost of less data parallelism.

### 7.2 GCP quota increase request (drafted 2026-05-05)

A quota-increase email was drafted for the user to send to GCP
(both via the IAM & Admin console and to
`cloud-tpu-trc-support@google.com`). Request: raise
`IN_USE_ADDRESSES` from 8 to 32 in europe-west4, us-central1,
us-east1, and us-central2. Draft is at
`_artifacts/gcp-quota-increase-request.md`. Until the quota is
bumped, v5e-64 and v6e-64 slices remain blocked; v4-32 and v4-64
stay viable (4 hosts = 4 IPs = fits cap).

## 8. Compile time observations

Recording how long XLA compile took on each successful run, so the
fallback policy can budget realistic wall time before declaring a
canary failed.

| Date | Slice | Strategy | scan_layers state | Forward step 1 | Backward step 1 | Notes |
|---|---|---|---|---|---|---|
| 2026-05-03 | v4-64 on-demand uc2b | fsdpv2_lora | (probe only, no training) | n/a | n/a | Per `.factory/memories.md` 2026-05-03 entries; 25+ min compile feared, never completed mid-session. |
| 2026-05-05 | v4-32 spot uc2b | fsdpv2_lora | TypeError -> manual loop | ~20 min | **~4h 25min** | scan_layers fallback was triggered on every layer call (40+ TypeError lines per step). Backward unrolled 36 CohereDecoderLayer + 6 MoshiDecoderLayer instances into HLO; XLA optimisation passes spent multi-hour wall time before producing executable. Process died at 8h 24min wall (likely supervisor timeout or OOM-kill); supervisor loop restarted from scratch with no XLA cache reuse. |

**Lesson:** the documented "25+ min compile" budget assumes
`scan_layers` is taking the fast path. With the manual-loop fallback
the compile is 5-10x longer. Until the `_KwargBoundLayer` patch in
`.factory/memories.md` 2026-05-05 "scan_layers TypeError" is applied,
do not budget less than 5h wall time for first-canary compile.


## v6e-16 multi-host envelope (hardening probes, 2026-07-14)

- Steady step (global batch 32 = b2/chip × 16, grad-ckpt ON, chunk 100):
  **1.79–1.85 s/step**; HBM **25.7–27.8 / 31.25 GiB** with train+val graphs.
- Capacity ladder: **b4/chip OOMs** — program arena wants 18.99 G with 18.42 G
  free (resident weights+optimizer ≈ 12.8 G/chip). The locked b2 recipe sits
  just under half the activation envelope.
- Step-time levers all neutral (probes): grad-ckpt OFF, depth_chunk 300,
  flash-attention ON. Persistent XLA compile cache: nondeterministic keys on
  torch_xla 2.9 v6e SPMD — never hits; ~14 min recompile per cold boot.
- Val×4 (`val_per_chip_batch: 8`, same 3200-sample gate): cycle ~210 s → ~41–60 s
  measured warm on the rehearsals.
