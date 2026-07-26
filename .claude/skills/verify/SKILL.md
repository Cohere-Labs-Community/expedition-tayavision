---
name: verify
description: Read .claude/VERIFY.md, execute every fenced bash block in order, and report pass/fail in a structured table. Use when the user asks "is it done", "/verify", or before declaring any milestone complete.
---

# Verify

Runs the curated verification blocks and produces a clean pass/fail report. The
Stop hook invokes this automatically after every response; humans invoke it via
`/verify`.

## Steps

1. Parse `.claude/VERIFY.md`. Extract every fenced ` ```bash ` block, preserving
   order. **Blocks fenced ` ```sh ` are documentation and `#verify` proposals —
   never execute them.** The slow and known-red checks live there and in
   `/verify-full`.
2. Each block is identified by its `# verify: <id>` first line. Use that id in
   all reporting; it is also what ties the block to a `<!-- verify:<id> -->`
   tag in PLAN.md.
3. For each block:
   - Run with `bash -eo pipefail -c <block>` from the repo root.
   - Capture stdout + stderr (last 400 chars).
   - Record exit code and elapsed time.
   - Honor a 15-second per-block timeout, and a cap of 5 blocks.
4. Render a markdown table with columns `#`, `id`, `status`, `time (s)`,
   `exit`, `tail`. Do not emit `<json-render>` or other widget markup — Claude
   Code renders it as literal text.
5. If all pass:
   - Tick matching `<!-- verify:<id> -->` items in `.claude/PLAN.md` via the
     `update-plan` skill.
   - Append a `done | verify` entry to PROGRESS.md.
6. If any fail:
   - Append a `fail | verify` entry with the failing block id and its tail.
   - List the failed ids at the bottom of the output.

## Skips

A block that guards optional tooling and prints `skip: <reason>` before
exiting 0 counts as **skipped**, not passed. Report it as such — a run where
everything skipped is not a green run.

## Budget

15s per block x a cap of 5 blocks = 75s, inside the 90s `Stop` timeout in
`settings.json`. Never add a check slower than about 2 seconds: this runs after
every single assistant response. pytest (171s here, and red without an HF token
for the gated `CohereLabs/tiny-aya-*` repos) belongs in `/verify-full` or CI.

## Success criteria

- The table exhaustively reflects every ` ```bash ` block in VERIFY.md.
- A run of any outcome leaves the working tree unchanged — confirm with
  `git status --short`.

## Failure modes

- VERIFY.md missing -> point the user at `.claude/MEMORY-SYSTEM.md`.
- A block exceeds 15s -> kill it and report `timeout`.
- A block prints to stderr but returns 0 -> still a pass; the stderr lands in
  the `tail` column.

## Composes with

- `update-plan` — called on full pass to tick PLAN items.
- `update-progress` — called regardless, to log the verification.
