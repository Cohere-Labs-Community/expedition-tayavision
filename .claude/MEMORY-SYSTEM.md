# MEMORY-SYSTEM.md — how this works and how to rebuild it

The external memory system gives a Claude Code session durable state across
restarts and compaction. Six lifecycle hooks read and write four markdown files;
skills, slash commands, and subagents are the manual interface to the same files.

## Who owns what

| File | Owner | Contains | Tracked in git? |
|---|---|---|---|
| `PLAN.md` | you + `plan-driver` | Current goal, Definition of Done, `## Tasks`, invariants | No |
| `PROGRESS.md` | hooks (append-only) | Timestamped log of edits, commands, verify runs, session boundaries | No |
| `VERIFY.md` | you | Fenced bash blocks the Stop hook runs as done-criteria | **Yes** |
| `memories.md` | you + `/remember` | Operational gotchas and dated project decisions | No |
| `settings.json` | you | Hook wiring | **Yes** |
| `hooks/`, `agents/`, `commands/`, `skills/` | you | The system itself | **Yes** |

The volatile files are gitignored via `.claude/.gitignore` — `PROGRESS.md` is a
machine log that would conflict on every merge, and `PLAN.md`/`memories.md` are
per-contributor working state.

> **`.claude/settings.json` needs a `.gitignore` escape.** The root `.gitignore`
> has a blanket `*.json` rule (line 11). Without `!.claude/settings.json`, a
> `git add .claude` produces a directory with no hook wiring. That negation is in
> place; do not remove it.

## The hooks

| Event | Script | Effect |
|---|---|---|
| SessionStart | `session_start.py` | Injects PLAN + recent PROGRESS + a VERIFY inventory line + memories as `additionalContext`. Deliberately does **not** inject `CLAUDE.md`/`AGENTS.md` — Claude Code already loads those into the cached system prompt |
| UserPromptSubmit | `user_prompt_submit.py` | Routes `#progress` / `#plan` / `#decision` / `#verify` quick-capture prefixes to the right file |
| PostToolUse | `post_tool_use.py` | Logs Write/Edit/MultiEdit and non-trivial Bash to PROGRESS, with repo-relative paths |
| Stop | `stop.py` | Runs every ` ```bash ` block in VERIFY.md, logs one entry, ticks matching PLAN items |
| PreCompact | `pre_compact.py` | Snapshots unchecked PLAN items so state survives compaction |
| SessionEnd | `session_end.py` | Writes a "next steps" block from unchecked PLAN items |

All seven files are stdlib-only, mode 644, and invoked as
`python3 "$CLAUDE_PROJECT_DIR"/.claude/hooks/<name>.py` — no exec bit needed.
**Every hook swallows its own exceptions**, so a broken install fails *silently*.
That is what the `memory-system` block in VERIFY.md exists to catch. For live
debugging, run `claude --debug`.

## The ordering contract

`PROGRESS.md` is newest-first. New entries are inserted directly below:

```
<!-- progress:marker -->
```

`_lib.PROGRESS_MARKER` is the single definition, imported by both
`_lib.append_progress` and `user_prompt_submit`. If the marker is missing,
`insert_newest_first` **re-seeds it** under the H1 rather than appending at EOF.

`---` was deliberately rejected as the anchor: every skill/agent/command file in
this system uses YAML front matter, so a `---` search eventually matches a
closing front-matter delimiter; `---` also renders as a setext H2 or thematic
break, gets rewritten by markdown formatters, and occurs naturally in captured
tool output. The upstream `llm-architectures` version searched for `\n---\n` in a
file containing zero of them, so 572 entries appended oldest-first while the
header promised newest-first.

## The Stop-hook budget

15s per block × a cap of 5 blocks = 75s, inside the 90s `Stop` timeout in
`settings.json`. All five current blocks together measure ~0.5s.

Nothing slower than ~2s belongs in VERIFY.md — it runs after *every* assistant
response. pytest (171s, and red without an HF token for the gated
`CohereLabs/tiny-aya-*` repos) lives in `/verify-full` and CI instead.

Environment overrides:

| Variable | Effect |
|---|---|
| `MEMORY_STOP_DISABLE_VERIFY=1` | Stop hook skips verification entirely |
| `MEMORY_STOP_BLOCK_ON_FAIL=1` | A failing block blocks the stop instead of just logging |
| `MEMORY_ARCHIVE_DAYS` | Cutoff for `archive-progress` (default 90) |

## Block IDs

Each VERIFY block's first line is `# verify: <id>`. A PLAN item tagged
`<!-- verify:<id> -->` is auto-ticked to `- [x]` when that block passes. Without
the id the Stop hook falls back to matching the block's first 60 characters,
which are identical across blocks — that was dead code upstream.

## Rebuilding from scratch

If `.claude/` is lost (a `git clean -xfd` removes the gitignored memory files):

1. The tracked half restores with `git checkout .claude` — `settings.json`,
   `hooks/`, `agents/`, `commands/`, `skills/`.
2. Recreate `PROGRESS.md` with exactly three lines: an H1, a blank line, and
   `<!-- progress:marker -->`. Nothing else is required; hooks append below it.
3. Recreate `PLAN.md` with a `## Tasks` heading (the `#plan` router targets it)
   or run `/plan <goal>` to generate one.
4. Recreate `memories.md` with a `## Project decisions` heading (the `#decision`
   router targets it).
5. Run `/verify` — the `memory-system` block confirms the install.

`VERIFY.md` is tracked, so it survives.

## Maintenance

- `/curate` runs the `memory-curator` subagent: dedupe, flag stale decisions,
  flag PLAN items that have drifted.
- `archive-progress` moves entries older than 90 days (or beyond 500 lines) into
  `archive/PROGRESS-YYYY-Qn.md`. The upstream log reached 2,784 lines before
  anyone ran it — check `wc -l .claude/PROGRESS.md` occasionally.
