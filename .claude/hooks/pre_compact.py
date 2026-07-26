#!/usr/bin/env python3
"""PreCompact hook: snapshot the current PLAN into PROGRESS.md so the upcoming
context compaction does not lose session state.
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


def unchecked_plan_items() -> list[str]:
    if not PLAN_FILE.exists():
        return []
    items: list[str] = []
    for line in PLAN_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- [ ]"):
            items.append(stripped[5:].strip())
    return items


def main() -> None:
    data = read_input()
    trigger = data.get("trigger", "auto")

    items = unchecked_plan_items()
    if items:
        summary = f"PreCompact ({trigger}): {len(items)} unchecked PLAN items"
        detail = "Top open items:\n" + "\n".join(f"- {i}" for i in items[:10])
    else:
        summary = f"PreCompact ({trigger}): no unchecked PLAN items"
        detail = ""

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
