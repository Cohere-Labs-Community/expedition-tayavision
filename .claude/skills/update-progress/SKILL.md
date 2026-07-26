---
name: update-progress
description: Append a structured entry to .claude/PROGRESS.md with timestamp, branch, sha, status, kind, and detail. Use when the user asks to log a change, capture a failure, or record a decision after the fact.
---

# Update Progress

Programmatic interface to the running log. Used by hooks and subagents; humans
should prefer `/progress` or the `#progress` quick-capture trigger.

`PROGRESS.md` owns chronological events: edits, commands, verification results,
session boundaries. Durable decisions graduate to `memories.md`; procedures
belong in `.claude/MEMORY-SYSTEM.md` or the repo's `docs/`.

## Inputs

- `status`: one of `info | done | fail | block` (required)
- `kind`: one of `edit | exec | decide | plan | verify | session` (required)
- `summary`: a one-line summary, <= 280 chars (required)
- `detail`: optional multi-line block

## Steps

1. Read `.claude/PROGRESS.md` and locate the `<!-- progress:marker -->` line.
2. Compose the entry:
   ```
   ## <iso8601 utc> | <branch>@<short-sha> | <status> | <kind>
   <summary>

   <detail>
   ```
3. Insert it **immediately below the marker**, so the newest entry is first.
   If the marker is absent, re-seed it under the H1 — do not append at the end
   of the file. (Upstream searched for a `---` separator the file never
   contained, so 572 entries silently appended at EOF in reverse of the
   documented order.)
4. Run the same secret-scrub regexes as `_lib.py` before writing.
5. Confirm to the caller in one short line.

## Success criteria

- The new entry is the topmost dated entry in PROGRESS.md.
- Branch and short-sha match `git rev-parse --abbrev-ref HEAD` and
  `git rev-parse --short HEAD`.
- No secret patterns survived the scrub.

## Failure modes

- File missing -> recreate it per the "Rebuilding from scratch" section of
  `.claude/MEMORY-SYSTEM.md` (H1, blank line, marker) and retry.
- Read-only -> surface the error; do not silently swallow it.

## Composes with

- `update-plan` — typically runs together when a plan item completes.
- `archive-progress` — runs after this skill to keep the file lean.
