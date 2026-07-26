---
description: Print the current memory state (PLAN, PROGRESS, VERIFY, memories) as structured markdown.
---

Run the `keep-context-fresh` skill in interactive mode: read the four memory
files and render a clean state summary. Write nothing.

Output structure:

1. **Heading** — `Memory state`, with the current branch and short sha.
2. **Goal** — one-line summary from `.claude/PLAN.md`'s `## Goal`.
3. **Definition of Done** — `n/m complete`, then the unchecked items as a list.
4. **Open PLAN tasks** — a markdown table of the top 10 unchecked items, with
   columns `phase`, `item`. Ordered by phase.
5. **Recent PROGRESS** — a markdown table of the 10 most-recent entries, with
   columns `when`, `kind`, `status`, `summary`.
6. **Verify status** — one line drawn from the most recent `verify`-kind entry
   in PROGRESS.md, e.g. `verify: 5/5 passed on Stop (2026-07-25T09:14:02Z)`.
   If there is none, say `verify: no run recorded yet`.
7. **Stop-hook checks** — the `# verify: <id>` ids currently defined in
   VERIFY.md, comma-separated.
8. **Decisions** — the count of `### YYYY-MM-DD:` blocks in `memories.md`, plus
   the three most recent titles.

Steps:

1. Read all four memory files.
2. Compose the summary as plain markdown — headings, tables, and lists. Do not
   emit `<json-render>` blocks or any other widget markup; Claude Code renders
   those as literal text.
3. Do not write any file.

Failure handling:

- If any of the four memory files is missing, lead with a bold warning line
  naming the missing file and pointing at `.claude/MEMORY-SYSTEM.md`, then
  render whatever is available.
- If PLAN.md has no `## Goal` heading, the Goal section reads
  `(none — run /plan to generate)`.
