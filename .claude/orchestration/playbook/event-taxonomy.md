# Event taxonomy

Push events, not blocking prompts. `tinyaya-stage2-scale` required a blocking `AskUser`
snapshot at T+15/30/45/60/90 min; that assumes a human watching a single foreground run,
and it would stall an overnight run at the first milestone.

Events are emitted by `notify()` in `scripts/tpu/_lib.sh`, which is a **no-op when
`NTFY_TOPIC` is unset**. Call sites are therefore unconditional — nothing has to check
whether notifications are configured, and a first run with no `.env` works, just deaf.

## Events

| Source | Event | When | Why it matters |
|---|---|---|---|
| `qr_watch` | `QR state changed` | QR leaves `ACTIVE` | First warning that capacity was lost |
| `qr_watch` | `resubmitted (n/20)` | after a recycle | Confirms self-heal fired; the counter is the flap signal |
| `qr_watch` | `quota abort` | quota-class failure | Resubmitting cannot help — needs a human. Usually `IN_USE_ADDRESSES`, not chips. |
| `qr_watch` | `recycle BLOCKED — SAVE_CKPT_DIR not gs://` | preemption with local checkpoints | Self-heal was declined **on purpose**; recycling would restart from step 0 and re-pay ~14 min of compile |
| `qr_watch` | `budget exhausted` | 20 resubmits spent | Run is stopped, capacity is gone |
| `qr_watch` | heartbeat | every 2h | Proves the watcher itself is alive |
| `deploy_tarball` | `deployed tag=<RUN_TAG> to N worker(s)` | fan-out complete | The tag is what scopes every later log query |
| `train_launcher` | `entrypoint missing: <path>` | launch, pre-exec | Catches a typo'd `TAYAVISION_ENTRYPOINT` in seconds instead of 20 min into a spot acquisition |
| `train_launcher` | `run OK — <last metric line>` | clean exit | The result, without opening W&B |
| `train_launcher` | `run FAILED rc=<n> — <tail>` | dirty exit | Slice is idle and needs a look |

## Design rules

1. **Every event carries its outcome, not just its occurrence.** `run OK — step=1000
   loss=2.41` beats `run finished`. An event you have to go look something up after is
   only half an event.
2. **A silent watcher is indistinguishable from a dead one**, hence the 2-hour heartbeat.
   The absence of bad news is not good news.
3. **Never push per-step or per-interval progress.** Metrics belong on W&B. A muted topic
   is worse than no topic, and the fastest way to get a topic muted is to make it noisy.
4. **Terminal events say what is now idle.** The expensive mistake is not a failed run;
   it is a slice sitting unused for six hours because nobody knew the run ended.

## Where to look when an event arrives

| Event | First thing to open |
|---|---|
| Any `qr_watch` failure | `/tmp/qr_watch_tayavision.log`, then `ops.sh status` |
| `recycle BLOCKED` | Your `SAVE_CKPT_DIR` — point it at `gs://tayavision-eu/...` and redeploy |
| `run FAILED` | worker 0 `/tmp/train.log` **tail-first**, then `tpu-diagnoser` |
| `entrypoint missing` | `TAYAVISION_ENTRYPOINT` in your launch env; `../SPEC.md` §9 for what is legal |
| `run OK` | `https://wandb.ai/cataluna84/tayavision-tpu`, then `ops.sh delete` if you are done |
| heartbeat gap > 2h | `tmux ls` on the workstation — has `qrwatch-tayavision` died? |
