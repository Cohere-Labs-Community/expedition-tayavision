---
description: Run the memory-curator subagent to dedupe, prune, and archive old PROGRESS entries and emit a "what changed" digest.
---

Delegate to the `memory-curator` subagent (see
`.claude/agents/memory-curator.md`).

Steps:

1. Invoke the curator via the Agent tool.
2. The curator will:
   - Run the `archive-progress` skill (move entries older than 90 days into
     `.claude/archive/PROGRESS-YYYY-Qn.md`).
   - Dedupe near-identical PROGRESS entries within 24h windows.
   - Identify stale `memories.md` decisions (no PROGRESS reference in 180 days,
     no `SUPERSEDED` tag).
   - Identify drifted PLAN items (no keyword match in PROGRESS or VERIFY).
3. Render the curator's digest verbatim.
4. The curator does NOT auto-fix `memories.md` or `PLAN.md`. It only flags
   items for human review.

Out of scope:

- Auto-deleting decisions: humans must mark them SUPERSEDED.
- Auto-removing PLAN items: humans must `- [x]` or remove them.
- Compressing archive files: archives are append-only forever.
