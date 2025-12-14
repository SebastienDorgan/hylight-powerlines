# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

# Top-level package/module name for coverage.
# Default is 'auto': we will try to guess it from 'src/<pkg>/'.
# You can override via CLI, e.g.: PKG=my_pkg just test-coverage
PKG := "auto"

# uv command (used by other tasks; installed by bootstrap)
UV := "uv"

# Keep uv's cache inside the repo by default (handy for sandboxed / CI environments).
export UV_CACHE_DIR := ".uv-cache"

# PlantUML native image version (GraalVM-based CLI)
# See: tags like v1.2025.0-native
PLANTUML_VERSION := "1.2025.0"

# Graphviz version for binary release
GRAPHVIZ_VERSION := "12.2.0"

# Notebooks config (MyST Markdown)
NB_DIR := "notebooks"
NB_FORMAT := "md:myst"
NB_HTML_DIR := "html"

# -------------------------------------------------------------------
# System tools bootstrap (non-uv tools)
# -------------------------------------------------------------------
# Installs/updates tools that are NOT managed via uv:
#   - uv (Python toolchain, to $HOME/.local/bin)
#   - Graphviz from binary release (dot, to $HOME/.local/bin)
#   - PlantUML native image (symlinked as 'plantuml' in $HOME/.local/bin)
#   - osv-scanner (Linux amd64 binary, to $HOME/.local/bin)
# Assumes: Linux amd64, curl, unzip, tar, and fontconfig already installed.
bootstrap:
    mkdir -p "$HOME/.local/bin"; if ! command -v uv >/dev/null 2>&1; then echo "Installing uv via official installer (to \$HOME/.local/bin)..."; curl -LsSf https://astral.sh/uv/install.sh | sh; else echo "Updating uv to latest version (if needed)..."; uv self update || true; fi; if ! command -v dot >/dev/null 2>&1; then echo "Installing Graphviz {{GRAPHVIZ_VERSION}} into \$HOME/.local/bin..."; ARCHIVE="graphviz-{{GRAPHVIZ_VERSION}}-linux-amd64.tar.gz"; URL="https://gitlab.com/graphviz/graphviz/-/releases/{{GRAPHVIZ_VERSION}}/downloads/${ARCHIVE}"; tmpdir="$(mktemp -d)"; curl -LsSf "$URL" -o "$tmpdir/$ARCHIVE"; mkdir -p "$HOME/.local/opt"; DEST="$HOME/.local/opt/graphviz-{{GRAPHVIZ_VERSION}}"; mkdir -p "$DEST"; tar -xzf "$tmpdir/$ARCHIVE" -C "$DEST" --strip-components=1 2>/dev/null || tar -xzf "$tmpdir/$ARCHIVE" -C "$DEST"; if [ -x "$DEST/bin/dot" ]; then ln -sf "$DEST/bin/dot" "$HOME/.local/bin/dot"; fi; if [ -x "$DEST/bin/neato" ]; then ln -sf "$DEST/bin/neato" "$HOME/.local/bin/neato"; fi; rm -rf "$tmpdir"; else echo "Graphviz already installed (dot in PATH)."; fi; if ! command -v plantuml >/dev/null 2>&1; then echo "Installing PlantUML native (headless, {{PLANTUML_VERSION}}) into \$HOME/.local/bin..."; TAG="v{{PLANTUML_VERSION}}-native"; ZIP="plantuml-headless-linux-amd64-{{PLANTUML_VERSION}}.zip"; URL="https://github.com/plantuml/plantuml/releases/download/${TAG}/${ZIP}"; tmpdir="$(mktemp -d)"; curl -LsSf "$URL" -o "$tmpdir/$ZIP"; mkdir -p "$HOME/.local/opt"; DEST="$HOME/.local/opt/plantuml-${TAG}"; mkdir -p "$DEST"; unzip -q "$tmpdir/$ZIP" -d "$DEST"; if [ -x "$DEST/plantuml-headless" ]; then ln -sf "$DEST/plantuml-headless" "$HOME/.local/bin/plantuml"; else echo "Warning: plantuml-headless not found in $DEST" >&2; fi; rm -rf "$tmpdir"; else echo "PlantUML already installed (plantuml in PATH)."; fi; if ! command -v osv-scanner >/dev/null 2>&1; then echo "Installing OSV-Scanner (linux_amd64 binary) to \$HOME/.local/bin..."; tmp="$(mktemp)"; curl -sSfL https://github.com/google/osv-scanner/releases/latest/download/osv-scanner_linux_amd64 -o "$tmp"; install -m 0755 "$tmp" "$HOME/.local/bin/osv-scanner"; rm -f "$tmp"; else echo "OSV-Scanner already installed (update manually if needed)."; fi; echo "Bootstrap finished."; echo 'Make sure $HOME/.local/bin is in your PATH.'

# -------------------------------------------------------------------
# Default task
# -------------------------------------------------------------------

default: test

# -------------------------------------------------------------------
# Environment / dependencies
# -------------------------------------------------------------------

# Install / sync all Python deps (including dev) via uv
sync:
    {{UV}} sync

# Update lockfile and re-sync (upgrade dependencies)
upgrade:
    {{UV}} lock --upgrade
    {{UV}} sync

# -------------------------------------------------------------------
# Linting & formatting (ruff)
# -------------------------------------------------------------------

# Lint the codebase (pass-through args to ruff)
# Usage examples:
#   just lint                   # basic lint
#   just lint --fix             # apply autofixes
#   just lint --select I F E    # choose rules
lint *args:
    {{UV}} run ruff check {{args}} .

# Auto-format the codebase
format:
    {{UV}} run ruff format .

# -------------------------------------------------------------------
# Type checking (pyrefly)
# -------------------------------------------------------------------

# Initialize pyrefly config (run once, or when layout changes)
pyrefly-init:
    {{UV}} run pyrefly init

# Type-check the project
typecheck:
    {{UV}} run pyrefly check --summarize-errors

# -------------------------------------------------------------------
# Tests & coverage (pytest)
# -------------------------------------------------------------------

# Run the test suite
test:
    {{UV}} run pytest

# Run the test suite with coverage.
# If PKG == "auto", we try to guess the package name from 'src/<pkg>/'.
test-coverage:
    PKG_NAME="{{PKG}}"; if [ "$PKG_NAME" = "auto" ]; then if [ -d src ]; then PKG_NAME="$(ls -1 src | head -n1 || true)"; fi; fi; if [ -z "$PKG_NAME" ]; then echo "Could not determine PKG for coverage." >&2; echo "Either set PKG explicitly, e.g. 'PKG=my_pkg just test-coverage',"; echo "or ensure you have a 'src/your_package/' directory."; exit 1; fi; echo "Running coverage for package: $PKG_NAME"; {{UV}} run pytest --cov="$PKG_NAME" --cov-report=term-missing

# -------------------------------------------------------------------
# Security / vulnerability scanning (osv-scanner)
# -------------------------------------------------------------------

security:
    # Run OSV-Scanner if available; degrade gracefully without network.
    #
    # Note: osv-scanner v2 uses subcommands (scan source ...). In sandboxed / offline
    # environments, avoid noisy network errors by running in --offline mode.
    if ! command -v osv-scanner >/dev/null 2>&1; then \
        echo "osv-scanner not found. Run 'just bootstrap' first." >&2; \
        exit 0; \
    fi; \
    if ! getent hosts api.osv.dev >/dev/null 2>&1; then \
        echo "No DNS/network for api.osv.dev; running OSV scan in --offline mode (no vulnerability lookup)." >&2; \
        osv-scanner scan source --offline -r . || true; \
        exit 0; \
    fi; \
    osv-scanner scan source -r .; code=$?; \
    if [ $code -eq 0 ]; then exit 0; fi; \
    if [ $code -eq 1 ]; then \
        echo "OSV scan found vulnerabilities (non-zero exit is expected)."; \
        echo "Suggested next step: 'uv lock --upgrade-package cryptography' then 'uv sync' (requires network)."; \
        if [ "${OSV_STRICT:-0}" = "1" ]; then exit 1; fi; \
        exit 0; \
    fi; \
    echo "OSV scan failed (exit=$code); rerun later with network access or use offline databases." >&2; \
    echo "Hint: osv-scanner scan source --download-offline-databases -r .  (requires network)" >&2; \
    exit 0

# -------------------------------------------------------------------
# Code Quality Aggregate
# -------------------------------------------------------------------

# "cqa" = Code Quality Analysis: lint + typecheck + security
cqa:
    just lint
    just typecheck
    just security

# CI-friendly aggregate
ci:
    just format
    just lint
    just typecheck
    just test
    just security

# -------------------------------------------------------------------
# Notebooks / Jupytext (MyST Markdown, HTML output only)
# -------------------------------------------------------------------

# Render a MyST notebook to HTML (no .ipynb written to disk)
# Usage: just nb-html notebooks/analysis.md
nb-html path:
    name=$(basename {{path}}); stem="${name%.*}"; mkdir -p {{NB_HTML_DIR}}; cat {{path}} | {{UV}} run jupytext --from {{NB_FORMAT}} --to ipynb --set-kernel - | {{UV}} run jupyter nbconvert --execute --to html --stdin --no-input --output "${stem}" --output-dir "{{NB_HTML_DIR}}"

# Render all notebooks in NB_DIR to HTML
nb-html-all:
    mkdir -p {{NB_HTML_DIR}}; for f in {{NB_DIR}}/*.md; do if [ -f "$f" ]; then just nb-html "$f"; fi; done
