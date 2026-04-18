#!/usr/bin/env bash
# validate.sh — Run all vocal pitch tracker evaluation phases
# Session: research/2026-04-17_vocal-pitch-tracker-research/
# Usage:
#   ./research/2026-04-17_vocal-pitch-tracker-research/validate.sh
#
# Prerequisites:
#   source .venv/bin/activate
#   Phases 2–4 require additional pip installs (see implementation-plan.md)

set -euo pipefail

SESSION="research/2026-04-17_vocal-pitch-tracker-research"
VOCALS_STEM="output/2026-04-13_escene-htdemucs-4s-wav/E.scene \uff02\u610f\u8b58\uff02 Music Video/vocals.wav"
MIX="/tmp/yt_download/E.scene \uff02\u610f\u8b58\uff02 Music Video.mp3"

echo "=== Vocal Pitch Tracker Evaluation ==="
echo "Session: $SESSION"
echo "Date: $(date +%Y-%m-%d)"
echo ""

# Phase 0 — check shared post-processor
if [ ! -f "$SESSION/pitch_to_midi.py" ]; then
  echo "[ERROR] pitch_to_midi.py not found. Implement Phase 0 first." >&2
  exit 1
fi

# Phase 1 — pYIN (zero new installs)
echo "--- Phase 1: pYIN on htdemucs stem ---"
python3 "$SESSION/run_pyin.py" \
  --vocals-stem "$VOCALS_STEM" \
  --midi-out "$SESSION/pyin_output.mid" \
  --json-out "$SESSION/pyin_results.json"
echo "pYIN complete. RTF and notes:"
python3 -c "import json; d=json.load(open('$SESSION/pyin_results.json')); print(f'  RTF={d[\"rtf\"]}x  notes={d[\"midi_notes\"]}  voiced={d[\"voiced_fraction\"]*100:.1f}%')"

echo ""

# Phase 2 — SPICE on raw mix
echo "--- Phase 2: SPICE on raw mix ---"
if python3 -c "import tensorflow_hub" 2>/dev/null; then
  python3 "$SESSION/run_spice.py" \
    --mix "$MIX" \
    --midi-out "$SESSION/spice_output.mid" \
    --json-out "$SESSION/spice_results.json"
  echo "SPICE complete. RTF and notes:"
  python3 -c "import json; d=json.load(open('$SESSION/spice_results.json')); print(f'  RTF={d[\"rtf\"]}x  notes={d[\"midi_notes\"]}  voiced={d[\"voiced_fraction\"]*100:.1f}%')"
else
  echo "[SKIP] tensorflow_hub not installed. Run: pip install tensorflow tensorflow-hub"
fi

echo ""

# Phase 3 — PESTO on htdemucs stem
echo "--- Phase 3: PESTO on htdemucs stem ---"
if python3 -c "import pesto" 2>/dev/null; then
  python3 "$SESSION/run_pesto.py" \
    --vocals-stem "$VOCALS_STEM" \
    --midi-out "$SESSION/pesto_output.mid" \
    --json-out "$SESSION/pesto_results.json"
  echo "PESTO complete. RTF and notes:"
  python3 -c "import json; d=json.load(open('$SESSION/pesto_results.json')); print(f'  RTF={d[\"rtf\"]}x  notes={d[\"midi_notes\"]}  voiced={d[\"voiced_fraction\"]*100:.1f}%  conf={d[\"mean_confidence\"]}')"
else
  echo "[SKIP] pesto not installed. Run: pip install pesto-pitch"
fi

echo ""

# Phase 4 — FCPE on htdemucs stem
echo "--- Phase 4: FCPE on htdemucs stem ---"
if python3 -c "import torchfcpe" 2>/dev/null; then
  python3 "$SESSION/run_fcpe.py" \
    --vocals-stem "$VOCALS_STEM" \
    --midi-out "$SESSION/fcpe_output.mid" \
    --json-out "$SESSION/fcpe_results.json"
  echo "FCPE complete. RTF and notes:"
  python3 -c "import json; d=json.load(open('$SESSION/fcpe_results.json')); print(f'  RTF={d[\"rtf\"]}x  notes={d[\"midi_notes\"]}  voiced={d[\"voiced_fraction\"]*100:.1f}%')"
else
  echo "[SKIP] torchfcpe not installed. Run: pip install torchfcpe"
fi

echo ""
echo "=== Summary of completed phases ==="
for tool in pyin spice pesto fcpe; do
  jf="$SESSION/${tool}_results.json"
  if [ -f "$jf" ]; then
    python3 -c "
import json
d = json.load(open('$jf'))
print(f\"$tool: RTF={d['rtf']}x  notes={d['midi_notes']}  voiced={d['voiced_fraction']*100:.1f}%  duration={d['duration_s']}s\")
"
  fi
done
echo ""
echo "Document results in $SESSION/results.md"
