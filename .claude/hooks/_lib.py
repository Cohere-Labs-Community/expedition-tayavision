"""Shared helpers for memory-system hooks. Imported by sibling hook scripts.

Stdlib only, so the hooks run under any `python3` without a virtualenv.

The ordering contract lives here: PROGRESS.md is newest-first, and new entries
are inserted directly below `PROGRESS_MARKER`. Both `append_progress` and
`user_prompt_submit` go through `insert_newest_first` so there is exactly one
implementation of that rule -- the upstream version had two copies searching for
a `---` separator that the file never contained, so every entry silently
appended at EOF and the log ran oldest-first for 572 entries.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()).resolve()
CLAUDE_DIR = PROJECT_DIR / ".claude"

PROGRESS_FILE = CLAUDE_DIR / "PROGRESS.md"
PLAN_FILE = CLAUDE_DIR / "PLAN.md"
VERIFY_FILE = CLAUDE_DIR / "VERIFY.md"
MEMORIES_FILE = CLAUDE_DIR / "memories.md"
ARCHIVE_DIR = CLAUDE_DIR / "archive"

# TPU run-control design docs. Separate from the four memory files because these are
# *tracked* and shared, where the memory files are per-contributor. Two of them are
# injected at SessionStart; the README and the run SPEC are read on demand by the
# tpu-orchestrate skill. Every consumer guards on .exists() -- a branch that predates
# the orchestration import must not get "<missing: ...>" prepended to every session.
ORCHESTRATION_DIR = CLAUDE_DIR / "orchestration"
ORCHESTRATION_README = ORCHESTRATION_DIR / "README.md"
ORCHESTRATION_CONTROL_PLANE = ORCHESTRATION_DIR / "CONTROL_PLANE.md"
ORCHESTRATION_TPU_OPT_SPEC = ORCHESTRATION_DIR / "TPU_OPTIMIZATION_SPEC.md"

# The anchor new PROGRESS entries are inserted below. An HTML comment rather
# than `---`: this system uses YAML front matter in every skill/agent/command
# file, `---` renders as a setext heading or thematic break, markdown formatters
# rewrite it, and it appears constantly inside captured tool output.
PROGRESS_MARKER = "<!-- progress:marker -->"

SECRET_PATTERNS = [
    re.compile(r"hf_[a-zA-Z0-9]{20,}"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"gh[ps]_[a-zA-Z0-9]{30,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"BEGIN.*PRIVATE KEY"),
    # Modal tokens (MODAL_TOKEN_ID / MODAL_TOKEN_SECRET) -- everything heavy in
    # this repo runs on Modal, so these are the likeliest thing to leak.
    re.compile(r"\ba[ks]-[A-Za-z0-9]{16,}\b"),
    # Named credentials in `KEY=value` form. Deliberately not a bare 40-hex
    # pattern, which would redact every git SHA in the log.
    re.compile(r"(?i)\b(wandb|hf|hugging\s?face)[_-]?(api[_-]?)?(key|token)\s*[=:]\s*\S+"),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_DIR),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        return out.strip() or "no-git"
    except Exception:
        return "no-git"


def git_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(PROJECT_DIR),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        return out.strip() or "detached"
    except Exception:
        return "no-git"


def scrub(text: str) -> str:
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def rel(path: str) -> str:
    """Render a path relative to the repo root when it lives inside it.

    Keeps machine-specific absolute paths out of the log.
    """
    try:
        return str(Path(path).resolve().relative_to(PROJECT_DIR))
    except Exception:
        return path


def read_input() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def emit(payload: dict | None = None) -> None:
    if payload is not None:
        print(json.dumps(payload))


def insert_newest_first(path: Path, block: str) -> None:
    """Insert `block` immediately below PROGRESS_MARKER (newest entry first).

    `block` is a markdown chunk with no leading or trailing blank lines.

    If the marker is absent we re-seed it under the H1 rather than appending at
    the end. The old append-at-end fallback silently inverted the file's
    documented ordering and nobody noticed for 572 entries.
    """
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    idx = content.find(PROGRESS_MARKER)
    if idx < 0:
        title, sep, rest = content.partition("\n")
        content = f"{title}{sep}\n{PROGRESS_MARKER}\n\n{block}\n\n{rest.lstrip(chr(10))}"
    else:
        end = idx + len(PROGRESS_MARKER)
        content = f"{content[:end]}\n\n{block}\n\n{content[end:].lstrip(chr(10))}"
    path.write_text(content, encoding="utf-8")


def append_progress(status: str, kind: str, summary: str, detail: str = "") -> None:
    if not PROGRESS_FILE.exists():
        return
    summary = scrub(summary).replace("\n", " ").strip()[:300]
    detail = scrub(detail).strip()
    if not summary:
        return
    block = f"## {now_iso()} | {git_branch()}@{git_rev()} | {status} | {kind}\n{summary}"
    if detail:
        block += f"\n\n{detail}"
    insert_newest_first(PROGRESS_FILE, block)


def read_file_safe(path: Path, max_lines: int | None = None) -> str:
    if not path.exists():
        return f"<missing: {path}>"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"<unreadable: {path} ({exc})>"
    if max_lines is not None:
        lines = text.splitlines()
        if len(lines) > max_lines:
            text = "\n".join(lines[:max_lines]) + f"\n... <{len(lines)-max_lines} more lines>"
    return text
