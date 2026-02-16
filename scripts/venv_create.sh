#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ -d "${VENV_DIR}" ]]; then
  echo "Virtual environment already exists at ${VENV_DIR}"
else
  echo "Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

echo "Activate with: source ${VENV_DIR}/bin/activate"
