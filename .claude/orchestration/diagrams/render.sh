#!/usr/bin/env bash
# Render Mermaid .mmd files to SVG (and optionally PNG).
#
# Usage:
#   bash render.sh                # render all *.mmd in this dir
#   bash render.sh 03-memory-lifecycle.mmd   # render one file
#
# Requires mermaid-cli (mmdc):
#   npm install -g @mermaid-js/mermaid-cli
# or run via npx: npx @mermaid-js/mermaid-cli -i in.mmd -o out.svg
#
# Output goes to ./svg/<name>.svg and ./png/<name>.png

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p svg png

# Pick mmdc path: prefer global, fall back to npx.
if command -v mmdc >/dev/null 2>&1; then
    MMDC="mmdc"
else
    MMDC="npx -y @mermaid-js/mermaid-cli"
fi

render_one() {
    local src="$1"
    local base
    base="$(basename "$src" .mmd)"
    echo "[render] $src -> svg/${base}.svg"
    $MMDC -i "$src" -o "svg/${base}.svg"
    echo "[render] $src -> png/${base}.png"
    $MMDC -i "$src" -o "png/${base}.png"
}

if [ $# -eq 0 ]; then
    for f in *.mmd; do
        [ -e "$f" ] || continue
        render_one "$f"
    done
else
    for f in "$@"; do
        render_one "$f"
    done
fi

echo "[render] done -> $(pwd)/svg/ and $(pwd)/png/"
