---
description: Append a manual entry to .claude/PROGRESS.md (without going through a hook).
argument-hint: <what to log>
---

Append `$ARGUMENTS` to `.claude/PROGRESS.md` as the newest entry.

Use the `update-progress` skill with:

- `status: info`
- `kind: session`
- `summary: $ARGUMENTS` (truncated to 280 chars)

Format the entry exactly as PROGRESS.md's header block specifies:

```
## <iso8601 utc> | <branch>@<short-sha> | info | session
$ARGUMENTS
```

Insert it immediately below the `<!-- progress:marker -->` line so it becomes
the topmost dated entry. That marker is the file's ordering contract — see
`.claude/MEMORY-SYSTEM.md`. If it is missing, re-seed it under the H1 rather
than appending at the end of the file.

Run the secret-scrub before writing.

If `$ARGUMENTS` is empty, prompt the user for what to log and stop.

Confirm in one line: `memory: appended to PROGRESS.md`.
