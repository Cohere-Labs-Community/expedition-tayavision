---
name: update-plan
description: Maintain .claude/PLAN.md - tick completed items, add new sub-tasks, edit Definition of Done. Idempotent. Use when work has progressed and the plan needs to reflect reality.
---

# Update Plan

Edits PLAN.md in place. The Stop hook ticks items automatically via their
`<!-- verify:<id> -->` tags; humans can call `/plan` (which delegates to the
`plan-driver` subagent for full regeneration) or add `#plan <task>`
quick-capture lines.

`PLAN.md` owns only the active goal, phase checklist, and Definition of Done.
Chronological events go in `PROGRESS.md`, durable decisions in `memories.md`.

## Inputs

One of:

- `tick: <substring>` — find the first matching `- [ ]` line and promote it to
  `- [x]`.
- `add: <text>` — add a new `- [ ] <text>` under `## Tasks`.
- `dod_add: <text>` — add a new bullet under `## Definition of Done`.
- `regenerate: true` — delegate to the `plan-driver` subagent.

## Steps

1. Read `.claude/PLAN.md`.
2. Apply the requested edit:
   - `tick`: case-insensitive substring search; replace `- [ ]` with `- [x]` on
     the first match. If no match, fail loudly.
   - `add`: append under `## Tasks`, before any `## Out of scope`.
   - `dod_add`: append under `## Definition of Done`.
3. Write the file back atomically.
4. If all Definition of Done items are checked, append a milestone entry to
   `memories.md` and a `done | plan` entry to `PROGRESS.md`.

## Auto-ticking from VERIFY

A PLAN item tagged `<!-- verify:<id> -->` is ticked by the Stop hook whenever
the VERIFY.md block with that `# verify: <id>` first line passes. Prefer that
tag over a manual `tick:` for anything a check already covers — it cannot drift.

## Idempotency

- Ticking an already-checked item is a no-op.
- Adding an item that already exists (substring match on the body) is a no-op.

## Success criteria

- The applied diff is minimal — a single-line change for `tick`, a single-line
  insert for `add`.
- The file still parses as markdown, and the literal `## Tasks` heading
  survives (the `#plan` router depends on it).

## Failure modes

- `## Tasks` heading missing -> refuse to add; ask the user to seed PLAN.md per
  `.claude/MEMORY-SYSTEM.md`.
- More than one match for `tick` -> tick the first and warn.
