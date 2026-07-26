---
name: recall-context
description: Background knowledge skill. Auto-loads when the agent plans a multi-step task; reads memory files into context without rendering them in chat. Use when starting a substantive change.
---

# Recall Context

A silent, model-only counterpart to `keep-context-fresh`. Loaded automatically
when a task is judged non-trivial; it does not appear in the slash-command
palette.

## When the agent invokes this

- Before generating a multi-file plan or refactor.
- Before answering a question that requires knowing prior decisions (e.g. "why
  does the model force DynamicCache instead of HybridCache?").
- Before producing a commit message that should align with the PROGRESS log.

## Steps

1. Read `.claude/PROGRESS.md`, last 50 lines.
2. Read `.claude/memories.md` in full.
3. Read `.claude/PLAN.md` in full.
4. Root `CLAUDE.md` and `AGENTS.md` are already in the system prompt — do not
   re-read them.
5. If the task touches TPU work, also read `.claude/orchestration/CONTROL_PLANE.md`
   §0 and §2 (the two compute paths and the source-of-truth table), plus
   `.claude/agents/tpu-diagnoser.md` when a log excerpt is in play. Skip for
   Modal/GPU work.
6. Produce no chat output. Internal reasoning only. Make subsequent decisions
   consistent with what was just read.

## Success criteria

- The next action references at least one memory entry by date or decision
  title (e.g. "per the 2026-07-25 decision on the PROGRESS marker").

## When NOT to invoke

- Trivial single-line edits.
- Questions answered entirely by code in the working tree.
- When the user explicitly asks for a fresh take ("ignore prior decisions").
