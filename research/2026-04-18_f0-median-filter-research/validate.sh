#!/usr/bin/env bash
# validate.sh — Validate F0 median filter implementation
# Usage: bash research/2026-04-18_f0-median-filter-research/validate.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$REPO/.venv/bin/python"
EVAL="$REPO/research/2026-04-18_blue-bird-vocals-midi-eval/evaluate.py"
URL="https://www.youtube.com/watch?v=2upuBiEiXDk"
OUT="$REPO/Naruto_Shippuden_Opening_3_Blue_Bird.mid"

pass() { echo "  PASS: $1"; }
fail() { echo "  FAIL: $1"; exit 1; }

echo "=== Validate: F0 median filter (default 7 frames / 70 ms) ==="
$VENV -m audio2midi.cli \
  --youtube-url "$URL" \
  --instrument vocals 2>&1 | grep -E "Notes produced|BPM="

UNISON=$($VENV "$EVAL" "$OUT" 2>/dev/null | grep '^unison_repeats:' | grep -o '[0-9]*' | head -1)
NOTES=$($VENV "$EVAL" "$OUT" 2>/dev/null | grep '^note_count:' | grep -o '[0-9]*' | head -1)
echo "  note_count=$NOTES  unison_repeats=$UNISON"
[ "$UNISON" -lt 30 ] && pass "unison repeats < 30" || fail "unison repeats $UNISON >= 30"
[ "$NOTES" -ge 150 ] && [ "$NOTES" -le 280 ] && pass "note count in range [150,280]" \
  || fail "note count $NOTES out of range"

echo ""
echo "=== Validate: filter disabled (--f0-filter-frames 1) ==="
$VENV -m audio2midi.cli \
  --youtube-url "$URL" \
  --instrument vocals \
  --f0-filter-frames 1 2>&1 | grep -E "Notes produced|BPM=" \
  && pass "no crash with --f0-filter-frames 1" || fail "crash with --f0-filter-frames 1"

echo ""
echo "=== Validate: large window (--f0-filter-frames 31) ==="
$VENV -m audio2midi.cli \
  --youtube-url "$URL" \
  --instrument vocals \
  --f0-filter-frames 31 2>&1 | grep -E "Notes produced|BPM=" \
  && pass "no crash with --f0-filter-frames 31" || fail "crash with --f0-filter-frames 31"

echo ""
echo "All validations passed."
