---
description: Run the slow checks the Stop hook deliberately skips - full pytest and the currently-red drift checks.
---

`.claude/VERIFY.md` only contains checks fast enough to run after every
response (~0.5s total). This command runs the rest.

Steps:

1. `uv run --no-sync ruff check .` — report the count and the top rules. This
   one is also a Stop-hook block, so it should already be 0; report any drift.
2. `uv run --no-sync pytest tests/ -q` — 129 tests, budget ~180s.
   - Expect **17 errors** in `tests/test_processing.py` and
     `tests/test_vlm_assembly.py` unless an `HF_TOKEN` with access to the gated
     `CohereLabs/tiny-aya-base` / `-global` repos is present. Report those as
     `known-red (gated repo)`, not as regressions.
   - The other 112 must pass. Any failure there is a real regression.
   - GPU tests self-skip via `requires_gpu` when CUDA is absent.
   - For a fast signal, the non-gated subset is 110 tests in ~22s:
     `uv run --no-sync pytest tests/ -q --ignore=tests/test_processing.py --ignore=tests/test_vlm_assembly.py`
3. Run every ` ```sh ` block in the "Not run by the Stop hook" section of
   VERIFY.md and report each as pass / fail / known-red. The dependency-manifest
   and checkpoint-path checks are known-red pending PLAN items P3 and P4.
4. Append one `done | verify` or `fail | verify` entry to PROGRESS.md
   summarising all three, via the `update-progress` skill.

Report as a markdown table: `check | status | detail`.

Do not auto-fix. In particular do not run `ruff check --fix` — `pipeline/` and
`scripts/` are actively edited upstream, and a mechanical reformat there will
conflict.
