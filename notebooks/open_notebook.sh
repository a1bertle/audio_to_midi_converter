#!/usr/bin/env bash
# open_notebook.sh — activate the project venv and launch Jupyter
# Usage: bash notebooks/open_notebook.sh [path/to/notebook.ipynb]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
NOTEBOOK="${1:-}"

# Resolve venv — prefer repo-root .venv, then notebook-local .venv
VENV=""
if [ -f "$REPO/.venv/bin/activate" ]; then
    VENV="$REPO/.venv"
elif [ -n "$NOTEBOOK" ] && [ -f "$(dirname "$NOTEBOOK")/.venv/bin/activate" ]; then
    VENV="$(dirname "$NOTEBOOK")/.venv"
else
    echo "WARNING: no .venv found — using system Python" >&2
fi

if [ -n "$VENV" ]; then
    source "$VENV/bin/activate"
fi

# Install requirements if present alongside the notebook
if [ -n "$NOTEBOOK" ] && [ -f "$(dirname "$NOTEBOOK")/requirements.txt" ]; then
    pip install -q -r "$(dirname "$NOTEBOOK")/requirements.txt"
fi

# Ensure Jupyter is available
if ! command -v jupyter &>/dev/null; then
    pip install -q jupyter
fi

# Ensure the repo venv is registered as a kernel
python -m ipykernel install --user --name audio2midi-venv \
    --display-name "audio2midi (.venv)" 2>/dev/null || true

if [ -n "$NOTEBOOK" ]; then
    jupyter notebook "$NOTEBOOK"
else
    jupyter notebook "$REPO/notebooks"
fi
