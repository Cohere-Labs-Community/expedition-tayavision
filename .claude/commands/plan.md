---
description: Generate or regenerate .claude/PLAN.md from the current goal using the plan-driver subagent.
argument-hint: <optional explicit goal>
---

Delegate to the `plan-driver` subagent (see `.claude/agents/plan-driver.md`) to
produce or regenerate `.claude/PLAN.md`.

Steps:

1. If `$ARGUMENTS` is provided, treat it as the explicit goal. If not, ask the
   user to state the goal in one sentence.
2. Read the current `.claude/PLAN.md` (if any) so existing structure is
   preserved when refining.
3. Invoke the `plan-driver` subagent via the Agent tool with:
   - The user's goal
   - The current branch
   - A pointer to `.claude/PROGRESS.md` and `.claude/memories.md`
4. Wait for the subagent's report.
5. Verify the new PLAN.md:
   - Has `## Goal`, `## Definition of Done`, `## Tasks`, `## Out of scope`
     headings. The literal `## Tasks` heading is required — the `#plan`
     quick-capture router appends under it.
   - Has at least 5 Definition of Done items, each observable.
   - Has at least 3 phased Task subsections.
6. If the subagent produced non-conforming output, ask it to retry once with
   the constraints made explicit.
7. Confirm in one line: `plan: regenerated PLAN.md from goal "<...>"`.

Out of scope for this command: editing PROGRESS.md, memories.md, VERIFY.md,
CLAUDE.md, or AGENTS.md.
