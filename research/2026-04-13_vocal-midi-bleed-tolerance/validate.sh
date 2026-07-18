#!/usr/bin/env bash
# validate.sh — vocal-to-MIDI bleed tolerance research session
# Runs evaluate_transcription.py against vocals stem (with bleed) and
# captures note count, pitch distribution, and confidence stats.
#
# Usage:
#   ./validate.sh
# Requirements:
#   - .venv activated (or run from repo root with source .venv/bin/activate)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION_DIR="$REPO_ROOT/research/2026-04-13_vocal-midi-bleed-tolerance"
VOCALS_WAV="$REPO_ROOT/output/2026-04-13_escene-htdemucs-4s-wav/E.scene ＂意識＂ Music Video/vocals.wav"
MIX_MP3="$REPO_ROOT/inputs/Foals - My Number (Official Audio).mp3"

source "$REPO_ROOT/.venv/bin/activate"

echo "=== Vocal-to-MIDI bleed tolerance validation ==="
echo "Session: $SESSION_DIR"
echo "Vocals stem: $VOCALS_WAV"
echo ""

python3 "$SESSION_DIR/evaluate_transcription.py" \
    --vocals-stem "$VOCALS_WAV" \
    --json-out "$SESSION_DIR/transcription_results.json" \
    2>&1 | tee "$SESSION_DIR/validate_output.log"

echo ""
echo "Done. Results in $SESSION_DIR/transcription_results.json"
