#!/usr/bin/env python3
"""SessionStart hook: inject the memory files into Claude's context.

Line caps are deliberately tight: this payload is prepended to every session, so
each file gets only enough to orient, not its full contents.

Deliberately does NOT inject CLAUDE.md or AGENTS.md. Claude Code already loads
both natively into the cached system prompt; re-injecting them as
`additionalContext` duplicates the payload uncached on every session start,
including resume and post-compact. (If a subproject AGENTS.md is ever added --
e.g. `pipeline/AGENTS.md` -- reconsider, but today there is exactly one and it
lives at the root.)

It DOES inject two orchestration docs, because they are not in the system prompt
and nothing else records the Modal/TPU boundary. CONTROL_PLANE.md earns its place
in every session: it is what stops a Modal fact being written into a TPU surface
and vice versa. Both are guarded on .exists().
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _lib import (  # noqa: E402
    PROJECT_DIR,
    PROGRESS_FILE,
    PLAN_FILE,
    VERIFY_FILE,
    MEMORIES_FILE,
    ORCHESTRATION_CONTROL_PLANE,
    ORCHESTRATION_TPU_OPT_SPEC,
    emit,
    read_file_safe,
    read_input,
    git_branch,
    git_rev,
)


def verify_inventory() -> str:
    """List the Stop-hook check ids rather than dumping VERIFY.md.

    VERIFY.md is mostly python heredocs; truncating it at 80 lines would show
    two of five blocks and waste the budget. The ids are what matters here.
    """
    if not VERIFY_FILE.exists():
        return "<missing: .claude/VERIFY.md>"
    ids = re.findall(
        r"^#\s*verify:\s*(\S+)", VERIFY_FILE.read_text(encoding="utf-8"), re.M
    )
    if not ids:
        return "(no checks defined)"
    return "Stop-hook checks: " + ", ".join(f"`{i}`" for i in ids)


def main() -> None:
    data = read_input()
    cwd = data.get("cwd") or str(PROJECT_DIR)

    parts: list[str] = []
    parts.append(
        f"# Memory System Context\n"
        f"\nbranch: `{git_branch()}@{git_rev()}`  cwd: `{cwd}`\n"
    )

    parts.append("\n## PLAN.md (current goal)\n")
    parts.append(read_file_safe(PLAN_FILE, max_lines=150))

    parts.append("\n## PROGRESS.md (most recent)\n")
    parts.append(read_file_safe(PROGRESS_FILE, max_lines=60))

    parts.append("\n## VERIFY.md (done-criteria)\n")
    parts.append(verify_inventory())

    parts.append("\n## memories.md (decisions and gotchas)\n")
    parts.append(read_file_safe(MEMORIES_FILE, max_lines=150))

    # Guarded: absent orchestration must inject nothing rather than "<missing: ...>".
    if ORCHESTRATION_CONTROL_PLANE.exists():
        parts.append("\n## orchestration/CONTROL_PLANE.md (surface ownership)\n")
        parts.append(read_file_safe(ORCHESTRATION_CONTROL_PLANE, max_lines=100))

    if ORCHESTRATION_TPU_OPT_SPEC.exists():
        parts.append("\n## orchestration/TPU_OPTIMIZATION_SPEC.md (protected TPU config)\n")
        parts.append(read_file_safe(ORCHESTRATION_TPU_OPT_SPEC, max_lines=60))

    parts.append(
        "\n## Quick capture\n"
        "\nStart a prompt with `#progress <what>`, `#plan <task>`, "
        "`#decision <what and why>`, or `#verify <bash>` to write straight to "
        "the matching memory file. See `.claude/MEMORY-SYSTEM.md`.\n"
    )

    additional_context = "\n".join(parts).strip()

    emit({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context,
        },
        "suppressOutput": True,
    })


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
