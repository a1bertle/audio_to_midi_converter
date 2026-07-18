#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

.venv/bin/python -m pytest tests/test_cli.py tests/test_rmvpe.py -q
.venv/bin/python scripts/eval_vocal_midi.py \
  --workdir outputs/2026-07-17_Adounravel_segfix \
  --json-out research/2026-07-17_adounravel-segfix-eval/final-measurements.json \
  --regions-out research/2026-07-17_adounravel-segfix-eval/final-hallucination-regions.json
.venv/bin/python scripts/eval_vocal_midi.py \
  --workdir outputs/2026-07-17_Foals_My_Number \
  --json-out research/2026-07-17_adounravel-segfix-eval/foals-measurements.json \
  --regions-out research/2026-07-17_adounravel-segfix-eval/foals-hallucination-regions.json
.venv/bin/python scripts/eval_vocal_midi.py \
  --workdir outputs/2026-07-17_Blue_Bird \
  --json-out research/2026-07-17_adounravel-segfix-eval/blue-bird-measurements.json \
  --regions-out research/2026-07-17_adounravel-segfix-eval/blue-bird-hallucination-regions.json

