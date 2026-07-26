#!/usr/bin/env python3
"""PostToolUse hook: append a structured PROGRESS entry whenever Claude runs
Write, Edit, MultiEdit, or a non-trivial Bash command.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _lib import append_progress, emit, read_input, rel  # noqa: E402


# Bash commands that are pure observation -- not worth a PROGRESS entry.
#
# Matched as a PREFIX, not a substring. The upstream version used `p in lower`,
# which dropped `uv run pytest && echo done` for containing "echo " while
# happily logging `cd /repo && rm -rf build`.
NOISY_PREFIXES = (
    "ls", "pwd", "cat", "head", "tail", "echo", "which", "command -v",
    "rg", "grep", "find", "wc", "stat", "du", "df", "tree", "file",
    "sed -n", "awk", "jq", "basename", "dirname", "realpath",
    "git status", "git log", "git diff", "git show", "git branch",
    "git rev-parse", "git check-ignore",
    "uv tree", "uv pip list", "nvidia-smi",
    "modal app list", "modal volume ls", "modal secret list",
)

INTERESTING_TOOLS = {"Write", "Edit", "MultiEdit", "Bash"}


def is_noise(first: str) -> bool:
    s = first.lstrip()
    return any(s == p or s.startswith(p + " ") for p in NOISY_PREFIXES)


def main() -> None:
    data = read_input()
    tool = data.get("tool_name", "")
    if tool not in INTERESTING_TOOLS:
        return

    inp = data.get("tool_input", {}) or {}
    resp = data.get("tool_response", {}) or {}

    if tool == "Bash":
        cmd = (inp.get("command") or "").strip()
        if not cmd:
            return
        first = cmd.splitlines()[0].strip()
        if is_noise(first):
            return
        # Claude's Bash tool_response has no standard `success` key; treat a
        # present non-zero/interrupted signal as failure, else done.
        if isinstance(resp, dict):
            failed = resp.get("interrupted") is True or resp.get("success") is False
        else:
            failed = False
        append_progress(
            status="fail" if failed else "done",
            kind="exec",
            summary=first[:280],
        )
        emit({"suppressOutput": True})
        return

    file_path = inp.get("file_path") or ""
    if not file_path:
        return

    if tool == "Write":
        action = "created"
    elif tool in {"Edit", "MultiEdit"}:
        action = "edited"
    else:
        return

    append_progress(
        status="done",
        kind="edit",
        summary=f"{action} `{rel(file_path)}`",
    )
    emit({"suppressOutput": True})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
