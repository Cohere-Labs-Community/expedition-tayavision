#!/usr/bin/env python3
"""Stop hook: when Claude finishes responding, run the VERIFY.md checks in
non-blocking mode.

Results land in PROGRESS.md; a failure does NOT block the stop by default --
override with MEMORY_STOP_BLOCK_ON_FAIL=1. Skip entirely with
MEMORY_STOP_DISABLE_VERIFY=1.

Budget: 12s per block x a cap of 6 blocks = 72s, inside the 90s Stop timeout in
settings.json. Was 5x15=75s; went to 6x12 when the backend-seam check landed, so
adding a block did not push the worst case past the ceiling. (Upstream allowed
20s x 12 = 240s against the same 90s ceiling, so the hook was killed mid-run
whenever more than four blocks were slow.)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _lib import (  # noqa: E402
    PROJECT_DIR,
    PROGRESS_FILE,
    VERIFY_FILE,
    PLAN_FILE,
    append_progress,
    emit,
    read_input,
)


# Only ```bash fences execute. ```sh fences in VERIFY.md are documentation and
# proposals -- that distinction is what keeps `#verify` quick-capture from
# injecting a command that then runs after every response.
CODE_FENCE = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)
VERIFY_ID = re.compile(r"^#\s*verify:\s*(\S+)", re.M)

PER_BLOCK_TIMEOUT = 12
MAX_BLOCKS = 6


def extract_commands() -> list[str]:
    if not VERIFY_FILE.exists():
        return []
    text = VERIFY_FILE.read_text(encoding="utf-8")
    return [m.group(1).strip() for m in CODE_FENCE.finditer(text) if m.group(1).strip()]


def block_id(cmd: str) -> str:
    """Stable handle for a block, from its `# verify: <id>` first line.

    Falls back to the first line, which is what upstream used exclusively --
    and which is identical across blocks once they all start with the same
    `cd "${CLAUDE_PROJECT_DIR:-$PWD}"`, making `tick_plan_for` dead code.
    """
    m = VERIFY_ID.search(cmd)
    return m.group(1) if m else cmd.splitlines()[0].strip()[:60]


def run_command(cmd: str, timeout: int = PER_BLOCK_TIMEOUT) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["bash", "-eo", "pipefail", "-c", cmd],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        out = (proc.stdout + proc.stderr).strip()
        return proc.returncode, out[-400:]
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    except Exception as exc:
        return 99, str(exc)[:400]


def tick_plan_for(cmd: str) -> None:
    """Tick any PLAN item tagged `<!-- verify:<id> -->` for this block.

    The tag may sit on a continuation line of a wrapped checklist item, so this
    walks back from the tag to the nearest preceding `- [ ]`. The original
    version required the tag and the checkbox on the SAME line and therefore
    silently ticked nothing for every multi-line item -- which is most of them.
    """
    if not PLAN_FILE.exists():
        return
    tag = f"<!-- verify:{block_id(cmd)} -->"
    lines = PLAN_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
    changed = False
    for i, line in enumerate(lines):
        if tag not in line:
            continue
        # Walk back to the item this tag belongs to, stopping at a blank line or
        # a heading so a stray tag cannot tick an unrelated item above it.
        for j in range(i, -1, -1):
            stripped = lines[j].strip()
            if not stripped or stripped.startswith("#"):
                break
            if lines[j].lstrip().startswith("- [ ]"):
                lines[j] = lines[j].replace("- [ ]", "- [x]", 1)
                changed = True
                break
            if lines[j].lstrip().startswith("- [x]"):
                break  # already ticked
    if changed:
        PLAN_FILE.write_text("".join(lines), encoding="utf-8")


def last_verify_summary() -> str:
    """Summary line of the most recent verify entry, for dedupe.

    Upstream logged one identical line 43 times out of 74 verify entries.
    """
    try:
        lines = PROGRESS_FILE.read_text(encoding="utf-8").splitlines()
    except Exception:
        return ""
    for i, line in enumerate(lines[:300]):
        if line.startswith("## ") and line.rstrip().endswith("| verify"):
            return lines[i + 1].strip() if i + 1 < len(lines) else ""
    return ""


def main() -> None:
    data = read_input()
    if data.get("stop_hook_active"):
        return

    if os.environ.get("MEMORY_STOP_DISABLE_VERIFY") == "1":
        return

    commands = extract_commands()
    if not commands:
        return

    dropped = max(0, len(commands) - MAX_BLOCKS)
    commands = commands[:MAX_BLOCKS]

    failures: list[tuple[str, int, str]] = []
    passes: list[str] = []
    for cmd in commands:
        rc, out = run_command(cmd)
        bid = block_id(cmd)
        if rc == 0:
            passes.append(bid)
            tick_plan_for(cmd)
        else:
            failures.append((bid, rc, out))

    summary = f"verify: {len(passes)}/{len(commands)} passed on Stop"
    if failures:
        summary += f" (first fail: {failures[0][0]})"
    if dropped:
        summary += f" [{dropped} block(s) over the {MAX_BLOCKS}-block cap not run]"

    detail_lines = []
    for bid, rc, out in failures:
        detail_lines.append(f"FAIL [{rc}] {bid}")
        if out:
            detail_lines.append(f"    {out.splitlines()[-1]}")
    detail = "\n".join(detail_lines)

    # Only log when something changed. A run that repeats the previous result
    # adds nothing and would dominate the log.
    if summary != last_verify_summary():
        append_progress(
            status="done" if not failures else "fail",
            kind="verify",
            summary=summary,
            detail=detail,
        )

    if failures and os.environ.get("MEMORY_STOP_BLOCK_ON_FAIL") == "1":
        emit({
            "decision": "block",
            "reason": (
                f"verify reported {len(failures)} failure(s): "
                f"{', '.join(f[0] for f in failures)}. "
                "Fix them before stopping, or unset MEMORY_STOP_BLOCK_ON_FAIL "
                "to disable this gate."
            ),
        })
        return

    emit({"suppressOutput": True})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
