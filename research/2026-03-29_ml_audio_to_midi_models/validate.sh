#!/usr/bin/env bash
set -euo pipefail

PYTHON="python3"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
fi

echo "python: ${PYTHON}"
"${PYTHON}" --version

echo
echo "import checks (best-effort):"
"${PYTHON}" - <<'PY'
import importlib
import sys

modules = [
    "torch",
    "onnxruntime",
    "basic_pitch",
    "piano_transcription_inference",
    "transkun",
]

for name in modules:
    try:
        importlib.import_module(name)
        print(f"OK   {name}")
    except Exception as exc:
        print(f"FAIL {name}: {exc}", file=sys.stderr)
PY

echo
echo "pytest (optional):"
if "${PYTHON}" -c "import pytest" >/dev/null 2>&1; then
  "${PYTHON}" -m pytest -q
else
  echo "pytest not installed in this Python environment; skipping."
fi
