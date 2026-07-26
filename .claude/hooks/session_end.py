#!/usr/bin/env python3
"""SessionEnd hook: write a 'Next steps' block to PROGRESS.md derived from any
unchecked items in PLAN.md, so the next session resumes from the right place.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _lib import (  # noqa: E402
    PLAN_FILE,
    append_progress,
    emit,
    read_input,
)


def unchecked() -> list[str]:
    if not PLAN_FILE.exists():
        return []
    items: list[str] = []
    for line in PLAN_FILE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("- [ ]"):
            items.append(s[5:].strip())
    return items


def main() -> None:
    data = read_input()
    reason = data.get("reason", "other")

    items = unchecked()
    if not items:
        summary = f"SessionEnd ({reason}): all PLAN items complete"
        detail = ""
    else:
        summary = f"SessionEnd ({reason}): {len(items)} item(s) carried forward"
        detail = "Next steps:\n" + "\n".join(f"- {i}" for i in items[:8])

    append_progress(
        status="info",
        kind="session",
        summary=summary,
        detail=detail,
    )

    emit({"suppressOutput": True})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
