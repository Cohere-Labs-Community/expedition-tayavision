# VERIFY.md — done-criteria

Every fenced ` ```bash ` block below is executed by the **Stop hook after each
assistant response**, from the repo root, via `bash -eo pipefail -c`, with a
12-second per-block timeout and a cap of 6 blocks (72s against the 90s hook
timeout).

Rules for adding a block:

- First line must be `# verify: <stable-id>`. The id ties the block to a
  `<!-- verify:<id> -->` tag in PLAN.md, which the Stop hook auto-ticks.
- Side-effect free. Must leave the working tree byte-identical.
- Must finish in well under 2s on a warm cache. Anything slower belongs in
  `/verify-full` or CI.
- Must pass on a clean tree *today*. A permanently-red check trains everyone to
  ignore this file.
- Guard optional tooling with a `skip:` message and `exit 0`.

Blocks tagged ` ```sh ` are documentation only — the Stop hook does not run them.

---

## 1. Lint is clean

`per-file-ignores` in `pyproject.toml` ratifies E402 in `scripts/modal_*.py`
(Modal requires imports inside/after image definitions). Everything else must be
zero.

```bash
# verify: ruff-full
cd "${CLAUDE_PROJECT_DIR:-$PWD}" || exit 1
{ command -v uv >/dev/null 2>&1 && [ -d .venv ]; } || { echo "skip: no uv/.venv"; exit 0; }
uv run --no-sync ruff check . --quiet && echo "ruff-full-ok"
```

## 2. Root shell entrypoints parse

`eval_cvqa.sh`, `eval_cvqa_merged.sh`, `eval_cvqa_qwen3_vl_4b.sh`,
`merge_weights.sh` are hand-edited between eval runs, and a syntax error there is
only discovered minutes into a Modal job.

`scripts/tpu/*.sh` matters more than the root scripts: a syntax error there is only
discovered 10-30 minutes into a spot acquisition, on a slice that is already billing.
The `[ -e ]` guard keeps unmatched globs from failing before those directories exist.

```bash
# verify: shell-syntax
cd "${CLAUDE_PROJECT_DIR:-$PWD}" || exit 1
n=0
for f in ./*.sh scripts/tpu/*.sh scripts/ci/*.sh .claude/orchestration/diagrams/*.sh; do
  [ -e "$f" ] || continue          # unmatched globs stay literal; skip them
  bash -n "$f" || exit 1
  n=$((n+1))
done
echo "shell-syntax-ok ($n files)"
```

## 3. Config contracts hold

Three things at once: every YAML under `config/` parses; the SigLIP pixel-shuffle
token arithmetic is internally consistent (a wrong `num_tokens_after_shuffle`
produces a silent shape mismatch hours into alignment training); and no YAML
carries a key the Python config objects would silently drop.

```bash
# verify: config-contracts
cd "${CLAUDE_PROJECT_DIR:-$PWD}" || exit 1
PY=python3; { command -v uv >/dev/null 2>&1 && [ -d .venv ]; } && PY="uv run --no-sync python"
$PY - <<'EOF'
import ast, pathlib, sys, yaml

bad = []

class Loader(yaml.SafeLoader):
    pass
Loader.add_multi_constructor("!", lambda l, s, n: None)

docs = {}
for p in sorted(pathlib.Path("config").rglob("*.yaml")):
    try:
        docs[p] = yaml.load(p.read_text(encoding="utf-8"), Loader=Loader) or {}
    except Exception as exc:
        bad.append(f"{p}: unparseable: {type(exc).__name__}: {str(exc)[:100]}")

s = docs.get(pathlib.Path("config/vision/siglip.yaml"), {})
if s:
    g = s["image_size"] // s["patch_size"]
    d = s["downsample_factor"]
    p_ = s["padded_grid_size"]
    for k, got, want in [
        ("vision_grid_size", s["vision_grid_size"], g),
        ("num_vision_tokens", s["num_vision_tokens"], g * g),
        ("padded_grid_size", p_, g + (g % d)),
        ("num_tokens_after_shuffle", s["num_tokens_after_shuffle"], (p_ // d) ** 2),
        ("pixel_shuffle_embed_dim", s["pixel_shuffle_embed_dim"], s["vision_hidden_size"] * d**2),
    ]:
        if got != want:
            bad.append(f"config/vision/siglip.yaml: {k}={got}, derived={want}")

def names(path, kind):
    p = pathlib.Path(path)
    out = set()
    if not p.exists():
        return out
    for cls in (n for n in ast.parse(p.read_text(encoding="utf-8")).body
                if isinstance(n, ast.ClassDef)):
        for st in cls.body:
            if kind == "field" and isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                out.add(st.target.id)
            if kind == "init" and isinstance(st, ast.FunctionDef) and st.name == "__init__":
                out.update(a.arg for a in st.args.args + st.args.kwonlyargs)
    out.discard("self")
    return out

model_params = names("config/model_config.py", "init")
train_fields = set()
for f in ("config/training_config.py", "config/multilingual_config.py", "config/lora_config.py"):
    train_fields |= names(f, "field")
NESTED = {"lora"}  # structural sub-blocks, not dataclass fields

for y in sorted(pathlib.Path("config/vision").glob("*.yaml")):
    unknown = sorted(set(docs.get(y, {})) - model_params)
    if unknown:
        bad.append(f"{y}: keys unknown to TinyAyaVisionConfig.__init__: {unknown}")
for y in sorted(pathlib.Path("config/training").glob("*.yaml")):
    unknown = sorted(set(docs.get(y, {})) - train_fields - NESTED)
    if unknown:
        bad.append(f"{y}: keys unknown to the training dataclasses: {unknown}")

print("\n".join(bad) if bad else f"config-contracts-ok ({len(docs)} yaml)")
sys.exit(1 if bad else 0)
EOF
```

## 4. lm-eval task registry resolves

123 task files across 6 suites. Every `!function utils.X` must resolve to a name
the sibling `utils.py` actually exports, and every `include:` target must exist.
lm-eval only reports these at task-load time, minutes into a job.

Module-level *assignments* count as exports — `mtvqa/utils.py` builds
`process_docs_ar` .. `process_docs_vi` from a factory, so a def-only scan would
produce 18 false positives.

```bash
# verify: eval-task-registry
cd "${CLAUDE_PROJECT_DIR:-$PWD}" || exit 1
PY=python3; { command -v uv >/dev/null 2>&1 && [ -d .venv ]; } && PY="uv run --no-sync python"
$PY - <<'EOF'
import ast, pathlib, re, sys, yaml

class Loader(yaml.SafeLoader):
    pass
Loader.add_multi_constructor("!", lambda l, s, n: None)

bad, n = [], 0
for d in sorted(p for p in pathlib.Path("evaluation/tasks").iterdir() if p.is_dir()):
    util = d / "utils.py"
    names = set()
    if util.exists():
        for node in ast.parse(util.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(a.asname or a.name.split(".")[0] for a in node.names)
    for y in sorted(list(d.rglob("*.yaml")) + list(d.rglob("_default_yaml"))):
        n += 1
        text = y.read_text(encoding="utf-8")
        try:
            yaml.load(text, Loader=Loader)
        except Exception as exc:
            bad.append(f"{y}: unparseable: {type(exc).__name__}: {str(exc)[:100]}")
            continue
        for inc in re.findall(r"^\s*include:\s*[\"']?([^\"'\s]+)", text, re.M):
            if not (y.parent / inc).exists():
                bad.append(f"{y}: include target missing: {inc}")
        for ref in re.findall(r"!function\s+([A-Za-z_][\w.]*)", text):
            mod, _, fn = ref.rpartition(".")
            if mod == "utils" and fn not in names:
                bad.append(f"{y}: !function {ref} unresolved in {util}")
print("\n".join(bad) if bad else f"eval-task-registry-ok ({n} task files)")
sys.exit(1 if bad else 0)
EOF
```

## 5. The memory system itself is intact

Every hook named in `settings.json` exists and imports, all four memory files
exist, and the PROGRESS ordering anchor survives. Catches a half-installed or
half-`git clean`ed `.claude/`.

```bash
# verify: memory-system
cd "${CLAUDE_PROJECT_DIR:-$PWD}" || exit 1
python3 -B - <<'EOF'
import json, pathlib, re, sys

sys.path.insert(0, ".claude/hooks")
bad = []
c = pathlib.Path(".claude")

try:
    settings = json.loads((c / "settings.json").read_text(encoding="utf-8"))
except Exception as exc:
    print(f"settings.json unreadable: {exc}")
    sys.exit(1)

for event, groups in (settings.get("hooks") or {}).items():
    for g in groups:
        for h in g.get("hooks", []):
            m = re.search(r"\.claude/hooks/([\w.]+\.py)", h.get("command", ""))
            if not m:
                bad.append(f"{event}: command does not reference a .claude/hooks script")
            elif not (c / "hooks" / m.group(1)).exists():
                bad.append(f"{event}: missing .claude/hooks/{m.group(1)}")

for f in ("PLAN.md", "PROGRESS.md", "VERIFY.md", "memories.md"):
    if not (c / f).exists():
        bad.append(f"missing .claude/{f}")

prog = c / "PROGRESS.md"
if prog.exists() and "<!-- progress:marker -->" not in prog.read_text(encoding="utf-8"):
    bad.append("PROGRESS.md lost its ordering anchor; entries will append oldest-first")

# The orchestration integration spans three places: constants in _lib.py, injection in
# session_start.py, and content in a separate tree. Remove any one and the other two
# still look fine -- which is exactly the state this repo had to recover from once.
# Every assertion is gated on the tree existing, so this is green before and after.
orch = c / "orchestration"
if orch.is_dir():
    for f in ("README.md", "CONTROL_PLANE.md", "SPEC.md", "TPU_OPTIMIZATION_SPEC.md",
              "playbook/tier-definitions.md", "playbook/diagnosis-table.md"):
        if not (orch / f).exists():
            bad.append(f"missing .claude/orchestration/{f}")
    lib_src = (c / "hooks" / "_lib.py").read_text(encoding="utf-8")
    for const in ("ORCHESTRATION_DIR", "ORCHESTRATION_README",
                  "ORCHESTRATION_CONTROL_PLANE", "ORCHESTRATION_TPU_OPT_SPEC"):
        if const not in lib_src:
            bad.append(f"_lib.py lost {const}: orchestration wiring is half-removed")
    if "ORCHESTRATION_CONTROL_PLANE" not in (c / "hooks" / "session_start.py").read_text(encoding="utf-8"):
        bad.append("session_start.py no longer injects CONTROL_PLANE.md")
    if not (c / "agents" / "tpu-diagnoser.md").exists():
        bad.append("playbook/diagnosis-table.md points at a tpu-diagnoser.md that is gone")
    # The diagnosis table must exist in exactly one place. Five drifting copies is the
    # documented failure mode of the upstream this was ported from.
    for pb in (orch / "playbook").glob("*.md"):
        if "| Signature" in pb.read_text(encoding="utf-8"):
            bad.append(f"{pb} looks like a second diagnosis table; the agent is canonical")
    # Scripts named in the README's implementation table must exist. Self-timing:
    # only enforced once scripts/tpu/ lands.
    if pathlib.Path("scripts/tpu").is_dir():
        named = set(re.findall(r"scripts/tpu/([\w.-]+\.(?:sh|py))",
                               (orch / "README.md").read_text(encoding="utf-8")))
        for s in sorted(named):
            if not (pathlib.Path("scripts/tpu") / s).exists():
                bad.append(f"orchestration/README.md points at missing scripts/tpu/{s}")

try:
    import _lib, post_tool_use, pre_compact, session_end, session_start, stop, user_prompt_submit  # noqa: F401
except Exception as exc:
    bad.append(f"hook import failed: {type(exc).__name__}: {exc}")

print("\n".join(bad) if bad else "memory-system-ok")
sys.exit(1 if bad else 0)
EOF
```

## 6. The backend seam holds

`src/backend/tpu_backend.py` is the only module allowed a module-level
`import torch_xla`. A leak anywhere in `src/`, `models/`, `pipeline/`, `config/`, or
`evaluation/` drags `libtpu` into the import graph and breaks every CPU/GPU run and all
129 tests -- at import time, so the failure lands nowhere near its cause.

```bash
# verify: backend-seam
cd "${CLAUDE_PROJECT_DIR:-$PWD}" || exit 1
bash scripts/ci/check_backend_seam.sh
```

---

## Not run by the Stop hook

Too slow, or currently red. Run via `/verify-full`, or before a PR. These fences
are ` ```sh `, so the Stop hook skips them.

Full test suite — 129 tests, 171s, and 17 `OSError: gated repo` errors on
`CohereLabs/tiny-aya-{base,global}` without an HF token. Belongs in CI (PLAN P2):

```sh
uv run --no-sync pytest tests/ -q
```

The non-gated subset, 110 passed in 22s — still too slow for a per-response hook:

```sh
uv run --no-sync pytest tests/ -q \
  --ignore=tests/test_processing.py --ignore=tests/test_vlm_assembly.py
```

Dependency-manifest agreement — red today (PLAN P3): `requirements.txt` lists
`hydra-core`, `orjson`, and `unsloth` that `pyproject.toml` does not, and omits
`hydra-zen`, `matplotlib`, and `modal`. `unsloth` was deliberately dropped in
c5826fd to unbreak `uv sync`:

```sh
comm -3 \
  <(sed 's/[<>=!].*//' requirements.txt | tr -d ' ' | grep -v '^$' | sort -u) \
  <(grep -oP '^\s+"\K[A-Za-z0-9_.-]+' pyproject.toml | sort -u)
```

Checkpoint paths are not pinned to one contributor's Modal volume — red today
(PLAN P4). Promote to a ` ```bash ` block once P4 lands:

```sh
! grep -rnE '/[0-9a-f]{8}-[0-9a-f-]{27}/' config/training/*.yaml
```

## Proposed checks

`#verify <bash>` quick-capture appends here as a ` ```sh ` fence. Nothing in
this section runs until you promote it to a ` ```bash ` block above, give it a
`# verify: <id>` first line, and confirm it finishes in under 2s.
