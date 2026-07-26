---
name: archive-progress
description: Move PROGRESS.md entries older than 90 days (or beyond 500 total lines) into .claude/archive/PROGRESS-YYYY-Qn.md. Use when PROGRESS.md grows unwieldy or as part of monthly maintenance.
---

# Archive Progress

Keeps `.claude/PROGRESS.md` lean by moving old entries to a quarterly archive.
Used by `/curate` and the `memory-curator` subagent.

## When to run

- Monthly, as routine maintenance.
- Whenever `wc -l .claude/PROGRESS.md` exceeds 500. (The upstream log reached
  2,784 lines before anyone ran this — check occasionally.)
- Before a major branch cut where you want a clean log.

## Steps

1. Read `.claude/PROGRESS.md`.
2. Parse the file into entry blocks: each starts with
   `## YYYY-MM-DDTHH:MM:SSZ ...`. The H1, the `<!-- progress:marker -->` line,
   and any hand-written narrative section are not entries.
3. Compute the cutoff: `now - 90 days`, configurable via `MEMORY_ARCHIVE_DAYS`.
4. Group entries older than the cutoff by `YYYY-Qn` from their timestamp.
5. Append each group, in chronological order, to
   `.claude/archive/PROGRESS-YYYY-Qn.md`. Create the file if absent with a
   1-line header `# PROGRESS archive YYYY Qn`.
6. Remove the archived entries from PROGRESS.md. **Leave the H1 and the
   `<!-- progress:marker -->` line intact** — the marker is the ordering
   contract for every subsequent write.
7. Add a single `info | session` entry noting the count and destination files.

## Idempotency

- Archive files are append-only; rerunning is a no-op for entries already moved.
- Never deletes from archive files.

## Success criteria

- `wc -l .claude/PROGRESS.md` decreases.
- `<!-- progress:marker -->` still present (the `memory-system` VERIFY block
  checks this).
- Total entry count across PROGRESS + all archives is unchanged.

## Failure modes

- Permission denied on the archive dir -> create it and retry; report and stop
  on a second failure.
- Malformed entry header -> skip it and log a warning to PROGRESS.md.

## Composes with

- `update-progress` — used to record the archival event.
- `memory-curator` (subagent) — runs this skill plus a dedupe pass.
