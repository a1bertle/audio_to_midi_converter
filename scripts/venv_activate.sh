#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing ${VENV_DIR}. Run scripts/venv_create.sh first."
  exit 1
fi

echo "Activation command: source ${VENV_DIR}/bin/activate"
SHELL_BIN="${SHELL:-/bin/bash}"
exec "${SHELL_BIN}" -i -c "source '${VENV_DIR}/bin/activate' && exec '${SHELL_BIN}' -i"
