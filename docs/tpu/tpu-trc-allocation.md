# TRC TPU allocation -- authoritative record

## 2026-07-25 update (Tiny Aya Vision)

**The grant is shared across sibling projects.** The same TRC allocation and the same GCP
project back `llm-architectures` (nanoGPT-JAX) and `tinyaya-stage2-scale` (speech), which
is why the quota rows below have history attached to models that are not this one. The
grant table is authoritative; the per-project usage notes are not.

For Tiny Aya Vision specifically:

- **Default profile `v6e-8-eu`** — single host, 8 chips, `europe-west4-a`. Single-host
  means no PJRT rendezvous and no external-IP pressure (see the `IN_USE_ADDRESSES`
  caveat in [`tpu-capacity-log.md`](tpu-capacity-log.md), which blocks every 8-host
  slice in this grant).
- **Checkpoints and code tarballs → `gs://tayavision-eu` (europe-west4).** One bucket,
  deliberately. Every other zone in this grant is therefore cross-region for this
  project — see "Bucket co-location" below before picking a non-EU profile.
- Prior observations recorded here against v6e-16 / v6e-64 came from sibling projects.
  The **quota and capacity behaviour transfers exactly** (same grant, same project); the
  throughput and memory numbers do not (different models).

**Status:** Active
**Captured:** 2026-05-05
**Recipient:** `<TRC grant recipient>`
**GCP project:** `ml-pipelines-315702`
**Grant duration:** 90 days (free Cloud TPU usage; non-TPU GCP services
still billed; Google may reclaim capacity at any time)
**Source:** TRC welcome email from `trc-support@google.com` (subject
"You have access to free Cloud TPUs"), pasted by the user on
2026-05-05 and archived verbatim in this document.

This file is the **single source of truth** for which TPU types,
zones, and tiers we are entitled to.

---

## 1. Allocation table

| Quantity | TPU type | Zone | Tier |
|---:|---|---|---|
| 32 chips | Cloud TPU v4 | `us-central2-b` | **spot** |
| 32 chips | Cloud TPU v4 | `us-central2-b` | **on-demand** |
| 64 chips | Cloud TPU v5e | `europe-west4-b` | spot |
| 64 chips | Cloud TPU v5e | `us-central1-a` | spot |
| 64 chips | Cloud TPU v6e | `europe-west4-a` | spot |
| 64 chips | Cloud TPU v6e | `us-east1-d` | spot |

The dual v4 lines in `us-central2-b` are independent quotas: 32 chips
of on-demand AND 32 chips of spot in the same zone.
`v6e-8-eu` is a smaller single-host slice requested against the v6e
spot capacity in `europe-west4-a`; it is not a separate TRC grant row.

### Profile shorthands

`scripts/tpu/launch_spot.sh` accepts the following `TRC_PROFILE`
values, each mapping to a single row above:

| `TRC_PROFILE` | TPU type passed to `gcloud` | Zone | Hosts | Bucket | Notes |
|---|---|---|---|---|---|
| **`v6e-8-eu`** (default) | `v6e-8` | `europe-west4-a` | 1 | co-located | Single-host 8-chip slice. No rendezvous, no external-IP pressure. The only profile that is unblocked today. |
| `v6e-64-ew4a` | `v6e-64` | `europe-west4-a` | 8 | co-located | Newest gen; needs `v2-alpha-tpuv6e`. Blocked on `IN_USE_ADDRESSES`. |
| `v5e-64-ew4b` | `v5litepod-64` | `europe-west4-b` | 16 | co-located | Largest v5e slice. Same region as the bucket. Blocked on `IN_USE_ADDRESSES`. |
| `v6e-64-ue1d` | `v6e-64` | `us-east1-d` | 8 | **cross-region** | Same as `-ew4a`, US zone. Egress on every checkpoint write. |
| `v5e-64-uc1a` | `v5litepod-64` | `us-central1-a` | 16 | **cross-region** | Same chip family as `-ew4b`, US zone. Egress. |
| `v4-32-uc2b` | `v4-32` | `us-central2-b` | 4 | **cross-region** | Legacy v4 fallback; same zone as the on-demand v4 quota. Egress. |

The on-demand v4 quota does not need a profile here -- it is already
the default of `scripts/tpu/launch_qr.sh` (no `--spot`).

### Bucket co-location

Tiny Aya Vision uses **one bucket, `gs://tayavision-eu` in `europe-west4`.** Only the
three `europe-west4-*` profiles are co-located with it.

A cross-region profile still works, but every checkpoint write and every code-tarball
pull crosses regions. On a 3.35B backbone with a projector checkpoint per save interval
that is not a rounding error, and TRC covers the TPU, not the egress.

So: exhaust EU capacity before reaching for a US zone. If a US zone is genuinely the only
capacity available, that is a deliberate call with two honest options — accept the egress
for a short run, or stand up a second region-paired bucket and point `SAVE_CKPT_DIR` at
it. `scripts/tpu/launch_spot.sh` prints a warning rather than blocking; it will not
decide this for you.

## 2. Important caveats (verbatim from the email)

> This free 90-day trial is only available for new Cloud TPUs you
> create in the zones listed above. To avoid charges, please be sure
> to create your Cloud TPUs in the appropriate zone.

> While your Cloud TPUs are free, you'll still be charged for the
> rest of the GCP services you use. If you have a new account, Google
> Cloud's $300 USD introductory credit may completely offset these
> costs, and you can minimize costs even more by utilizing the new
> Cloud TPU VM architecture.

> Please note that demand for Cloud TPUs is high, so we can't
> guarantee you'll get to use all of your TPU quota. Google reserves
> the right to reclaim TRC quota and TRC Cloud TPU capacity at any
> time.

> If you have access to both on-demand and preemptible quotas, we
> recommend preferring on-demand and falling back to preemptible
> if/when on-demand is not available.

> If you have access to v2-8 and/or v3-8 quotas, please be aware
> that these individual devices cannot be used in pod configurations.
> *(Not applicable to this grant -- no v2/v3 quotas.)*

> If you encounter an error message indicating that your quota is
> exhausted, confirm that you have deleted any unused Cloud TPUs
> and/or Queued Resources that may still be consuming quota.

## 3. How to pick a zone (decision tree)

**The decision tree is canonical in
[`tpu-capacity-log.md`](tpu-capacity-log.md) §1**, not here. That file owns observed
capacity behaviour and the fallback policy derived from it; this file owns the grant.
Keeping one copy is deliberate — a fallback tree that drifts from the observed capacity
log sends you to a zone that costs egress or is blocked outright.

**Operational default for Tiny Aya Vision:** `v6e-8-eu`. It is the only profile not
blocked by the external-IP cap, it is co-located with `gs://tayavision-eu`, and
single-host removes rendezvous from the failure surface entirely — which matters while
the TPU path is still being brought up. Set `TRC_PROFILE` explicitly for anything else.

## 4. Program requirements (we accepted these)

Per the welcome email, TRC participants are expected to:

- Share TRC-supported research with the world (peer-reviewed
  publications, open-source code, blog posts, or other means).
- Share detailed feedback with Google to help improve the TRC
  program and the underlying Cloud TPU platform.
- Conduct research in accordance with the Google AI Principles.
- Accept Google's Terms and Conditions.
- Acknowledge that the participant's information will be used
  in accordance with Google's Privacy Policy.

Acknowledged. Outputs from this repo (code, blog post, eval results)
are intended to satisfy the publication requirement.

## 5. Support

- Email: `trc-support@google.com`.
- Discord: `#tpu-research-cloud` channel on the Google Developer
  Community Discord server.
- Recommended reading: PyTorch/XLA performance debugging blog series (Parts I-III).

## 6. Footer (verbatim from the email)

> Google LLC 1600 Amphitheatre Parkway, Mountain View, CA 94043
>
> This email was sent to `<TRC grant recipient>` to update you
> about important information regarding your Google Cloud Platform
> account.

## 7. Cross-references

- [`tpu-runbook.md`](tpu-runbook.md) -- current launch / resume / staging.
- [`tpu-capacity-log.md`](tpu-capacity-log.md) -- observed queue times, the
  `IN_USE_ADDRESSES` blocker, and the autonomous fallback policy.
- `scripts/tpu/launch_spot.sh` -- materialises the `TRC_PROFILE` shorthands into
  `gcloud` flags and warns on cross-region profiles.
- `.claude/orchestration/CONTROL_PLANE.md` -- names this file as the owner of "TRC quota
  + zone grants". Live capacity is owned by the capacity log, not here.
- `.claude/memories.md` -- durable decisions, including the single-bucket choice.
