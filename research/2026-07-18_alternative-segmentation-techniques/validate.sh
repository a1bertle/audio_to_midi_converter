#!/usr/bin/env bash
# Repeatable validation commands for this research session.
# Requires: outputs/2026-07-17_Adounravel_segfix/ workdir present (MBR vocal
# stem + mix), .venv with basic-pitch installed (pip install basic-pitch;
# also requires `pip install "setuptools<81"` for resampy's pkg_resources
# import, see notes.md). The continuation (Findings 5-8) additionally uses
# mir_eval (already present in .venv) but no other new dependency.
set -euo pipefail

cd "$(dirname "$0")/../.."

VOCALS="outputs/2026-07-17_Adounravel_segfix/stems/mbr/eZuDklxsaRg_(vocals)_MelBandRoformerSYHFTV3Epsilon.wav"
MIX="outputs/2026-07-17_Adounravel_segfix/extracted/eZuDklxsaRg.wav"
SESSION="research/2026-07-18_alternative-segmentation-techniques"

echo "=== Experiment A: Basic Pitch ==="
.venv/bin/python "$SESSION/run_basic_pitch.py" "$VOCALS" "$SESSION/adounravel_basic_pitch.mid"
.venv/bin/python scripts/eval_vocal_midi.py \
  "$SESSION/adounravel_basic_pitch.mid" "$VOCALS" "$MIX" \
  --json-out "$SESSION/basic-pitch-measurements.json" \
  --regions-out "$SESSION/basic-pitch-hallucination-regions.json"

echo "=== Experiment B: Viterbi voicing smoothing (raw 2-state) ==="
.venv/bin/python "$SESSION/run_viterbi_smoothing.py" "$VOCALS" "$SESSION/adounravel_viterbi.mid"
.venv/bin/python scripts/eval_vocal_midi.py \
  "$SESSION/adounravel_viterbi.mid" "$VOCALS" "$MIX" \
  --json-out "$SESSION/viterbi-measurements.json" \
  --regions-out "$SESSION/viterbi-hallucination-regions.json"

echo "=== Continuation: cache RMVPE+pYIN F0 and eval-grid pYIN ground truth ==="
mkdir -p "$SESSION/cache"
.venv/bin/python "$SESSION/cache_f0.py" "$VOCALS" "$SESSION/cache"
.venv/bin/python "$SESSION/cache_eval_pyin.py" "$VOCALS" "$SESSION/cache"

echo "=== Finding 5/6/7: persistence-debounced Viterbi grid search (fast, cached) ==="
.venv/bin/python "$SESSION/grid_search_v2.py"

echo "=== Finding 8: recommended candidate (persist_frames=2, f0_filter_frames=5) ==="
.venv/bin/python - <<'PYEOF'
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, "research/2026-07-18_alternative-segmentation-techniques")
from run_viterbi_v2 import viterbi_smooth_notes, write_midi

f0 = np.load("research/2026-07-18_alternative-segmentation-techniques/cache/f0_accepted.npy")
notes = viterbi_smooth_notes(f0, f0_filter_frames=5, persist_frames=2)
write_midi(notes, Path("research/2026-07-18_alternative-segmentation-techniques/adounravel_viterbi_v2.mid"), 135.00)
print(f"wrote {len(notes)} notes")
PYEOF
.venv/bin/python scripts/eval_vocal_midi.py \
  "$SESSION/adounravel_viterbi_v2.mid" "$VOCALS" "$MIX" \
  --json-out "$SESSION/viterbi-v2-measurements.json" \
  --regions-out "$SESSION/viterbi-v2-hallucination-regions.json"

echo "=== Finding 8: tolerance-window (COnPOff-style, mir_eval) cross-check ==="
.venv/bin/python "$SESSION/tolerance_window_eval.py"

echo "Done. See notes.md for interpretation."
