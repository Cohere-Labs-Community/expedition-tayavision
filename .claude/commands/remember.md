---
description: Save a long-term decision or piece of domain knowledge to .claude/memories.md.
argument-hint: <what to remember>
---

Append the user's note to `.claude/memories.md` under the `## Project decisions`
section, formatted as:

```markdown
### YYYY-MM-DD: <short title derived from the note>
**Decision:** $ARGUMENTS
**Captured-from:** /remember
```

Use today's UTC date. Choose a short title (<= 80 chars) from the first
sentence of `$ARGUMENTS`.

Steps:

1. Read `.claude/memories.md`.
2. Locate the `## Project decisions` heading. This exact heading is also what
   the `#decision` quick-capture router targets — do not rename it.
3. Insert the new block immediately before the next `## ` heading, or at the
   end of the section if it is the last one.
4. Run the secret-scrub regexes (`hf_[a-zA-Z0-9]{20,}`, `sk-…`, `gh[ps]_…`,
   `AKIA…`, Modal `a[ks]-…`, `WANDB_API_KEY=…`) before writing; replace any
   match with `[REDACTED]`.
5. Write the file back.
6. Confirm in one short line: `memory: saved decision to memories.md`.

If `$ARGUMENTS` is empty, ask the user what to remember and stop.
