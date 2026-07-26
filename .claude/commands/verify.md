---
description: Run every ```bash block in .claude/VERIFY.md and produce a structured pass/fail report.
---

Delegate to the `verifier` subagent (see `.claude/agents/verifier.md`).

Steps:

1. Confirm `.claude/VERIFY.md` exists and is non-empty.
2. Invoke the `verifier` subagent via the Agent tool. It will:
   - Run every fenced ` ```bash ` block in VERIFY.md, in order, with a 15s
     per-block ceiling.
   - **Skip every ` ```sh ` block** — those are documentation and `#verify`
     proposals, not live checks. The slow ones live in `/verify-full`.
   - Render a markdown table of results (`#`, `id`, `status`, `time`, `exit`,
     `tail`).
   - Append a `done | verify` or `fail | verify` entry to PROGRESS.md.
3. Render the subagent's output verbatim.
4. If anything failed, lead with one line: `verify: FAIL on <block-id>`.
5. If all pass, tick matching `<!-- verify:<id> -->` items in `.claude/PLAN.md`
   via the `update-plan` skill.

This command does **not** block the session on failure by default. To make the
Stop hook blocking, set `MEMORY_STOP_BLOCK_ON_FAIL=1` in your shell before
launching `claude`. To silence Stop-hook verification entirely, set
`MEMORY_STOP_DISABLE_VERIFY=1`.

Out of scope: editing VERIFY.md, modifying source code, or auto-fixing
failures. The verifier reports; the user decides.
