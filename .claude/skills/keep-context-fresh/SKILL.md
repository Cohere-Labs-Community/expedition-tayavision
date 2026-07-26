---
name: keep-context-fresh
description: Master entry-point for the external memory system. Invoke at the start of any non-trivial task to load PROGRESS, PLAN, VERIFY, and memories.md, then propose the next action consistent with what's recorded. Use when the user asks "where were we", "what's the state", "load context", or starts a new goal.
---

# Keep Context Fresh

The master skill for the external memory system. It guarantees that every
non-trivial session begins from the same snapshot of state.

## When to use

- The user starts a new task and you have no recent memory of the goal.
- The user asks "where were we?", "load context", "recall the state".
- A new session begins after a `clear`, compact, or restart.
- Before producing a plan that touches multiple files or subsystems.

## Steps

1. Read all four canonical memory files:
   - `.claude/PROGRESS.md` (last 30 entries — newest are at the top, directly
     under `<!-- progress:marker -->`)
   - `.claude/PLAN.md` (in full)
   - `.claude/VERIFY.md` (in full)
   - `.claude/memories.md` (in full)

2. Root `CLAUDE.md` and `AGENTS.md` are already in the system prompt — Claude
   Code loads them natively. Do **not** re-read them; that just burns context.

3. If the task touches TPU work, a training run, or anything under `scripts/tpu/`,
   also read:
   - `.claude/orchestration/CONTROL_PLANE.md` — which surface owns which fact, and
     which of the two compute paths is authoritative for it
   - `.claude/orchestration/SPEC.md` — the run loop. **§9 says what does not work
     yet**; read it before proposing to launch anything.

   Skip this step for pure Modal/GPU work. The SessionStart hook already injected
   the first 100 lines of CONTROL_PLANE.md, so re-read it only for a row the
   truncation cut.

4. Render a concise state summary in chat as plain markdown — headings, tables,
   and lists. Do not emit `<json-render>` blocks or any other widget markup;
   Claude Code renders those as literal text. Include:
   - Goal (from PLAN's `## Goal`)
   - Top 5 unchecked PLAN items
   - Top 5 most recent PROGRESS entries with status
   - The `# verify: <id>` ids currently defined in VERIFY.md
   - If step 3 ran: the active `TRC_PROFILE` and whether a QR is live
   - One line: "What I'd do next", derived from PLAN and PROGRESS

5. Ask the user to confirm the goal before proceeding to actual edits.

## Inputs

- Optional `goal: <string>` override if the user names a different goal.

## Success criteria

- All four memory files were read in the current session (verifiable in the
  tool log).
- The user has either confirmed the goal or asked for plan regeneration — in
  which case invoke the `plan-driver` subagent.

## Failure modes

- If any memory file is missing, do **not** silently continue. Surface the gap
  and offer to recreate it following `.claude/MEMORY-SYSTEM.md`.

## Composes with

- `recall-context` — programmatic recall, no chat output.
- `plan-driver` (subagent) — for goal -> PLAN.md generation.
- `verify` — for running the VERIFY.md blocks.
