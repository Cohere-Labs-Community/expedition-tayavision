#!/usr/bin/env python3
"""UserPromptSubmit hook: route '#progress', '#plan', '#decision', '#verify'
quick-capture prefixes to the right memory file.

Section headings here must match the headings that actually exist in the target
files -- upstream, `#decision` wrote to `## Architecture decisions` and `#plan`
to `## Tasks`, neither of which existed, so both silently appended at EOF under
a stray heading.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _lib import (  # noqa: E402
    PROGRESS_FILE,
    PLAN_FILE,
    VERIFY_FILE,
    MEMORIES_FILE,
    emit,
    git_branch,
    git_rev,
    insert_newest_first,
    now_iso,
    read_input,
    scrub,
)


TRIGGERS = {
    "#progress": ("progress", PROGRESS_FILE),
    "#plan": ("plan", PLAN_FILE),
    "#decision": ("decision", MEMORIES_FILE),
    "#verify": ("verify", VERIFY_FILE),
}

# Headings the routers target. These must exist in the seeded files.
PLAN_SECTION = "## Tasks"
DECISION_SECTION = "## Project decisions"
VERIFY_SECTION = "## Proposed checks"


def main() -> None:
    data = read_input()
    prompt = (data.get("prompt") or "").strip()
    if not prompt.startswith("#"):
        return

    first_token = prompt.split(None, 1)[0].lower()
    if first_token not in TRIGGERS:
        return

    kind, mem_file = TRIGGERS[first_token]
    body = prompt[len(first_token):].strip()
    if not body:
        return

    body = scrub(body)
    if kind == "progress":
        block = (
            f"## {now_iso()} | {git_branch()}@{git_rev()} | info | quick-capture\n"
            f"{body}"
        )
        insert_newest_first(mem_file, block)
    elif kind == "plan":
        line = f"- [ ] {body}  <!-- captured {now_iso()} -->\n"
        _append_under_section(mem_file, PLAN_SECTION, line)
    elif kind == "decision":
        ymd = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        block = (
            f"\n### {ymd}: {body[:80]}\n"
            f"**Decision:** {body}\n"
            f"**Captured-from:** quick-capture (`#decision`)\n"
        )
        _append_under_section(mem_file, DECISION_SECTION, block)
    elif kind == "verify":
        # A `sh` fence, not `bash`: the Stop hook only executes ```bash blocks,
        # so a captured command is a proposal until someone promotes it by hand.
        # Quick-capture must not be able to inject a command that then runs
        # after every response.
        block = f"\n```sh\n{body}\n```\n"
        _append_under_section(mem_file, VERIFY_SECTION, block)

    emit({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": (
                f"Memory: captured `{kind}` entry to .claude/{mem_file.name}."
                + (
                    " It is a ```sh proposal and will NOT run on Stop until"
                    " promoted to a ```bash fence."
                    if kind == "verify"
                    else ""
                )
            ),
        },
        "systemMessage": f"memory: saved `{kind}` to {mem_file.name}",
        "suppressOutput": True,
    })


def _append_under_section(path: Path, section_header: str, text: str) -> None:
    """Insert `text` at the end of the named section.

    The heading is matched as a whole line, not as a substring. A bare
    `content.find("## Tasks")` also matches the phrase "## Tasks" inside prose
    or an HTML comment -- PLAN.md's own quick-capture hint mentions the heading
    by name, and a substring match landed every captured task above the real
    section.
    """
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(section_header)}[ \t]*$", content, re.M)
    if m is None:
        path.write_text(
            content.rstrip("\n") + f"\n\n{section_header}\n{text}", encoding="utf-8"
        )
        return
    nxt = re.search(r"^## ", content[m.end():], re.M)
    if nxt is None:
        path.write_text(content.rstrip("\n") + "\n" + text, encoding="utf-8")
        return
    cut = m.end() + nxt.start()
    path.write_text(content[:cut] + text + content[cut:], encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
