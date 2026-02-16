#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing ${VENV_DIR}. Run scripts/venv_create.sh first."
  exit 1
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Invalid virtual environment: ${VENV_DIR}"
  exit 1
fi

source "${VENV_DIR}/bin/activate"
exec python -m audio2midi.cli "$@"
