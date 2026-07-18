#!/usr/bin/env bash
# validate.sh — Vocal separation benchmark: MBR vs HTDemucs baseline
# Session: research/2026-04-26_vocal-stem-separation-alternatives/
# Usage: bash research/2026-04-26_vocal-stem-separation-alternatives/validate.sh
# Prerequisites: .venv active, audio-separator installed, benchmark input present

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
SESSION_DIR="$SCRIPT_DIR"

INPUT="$REPO_ROOT/inputs/Foals - My Number (Official Audio).mp3"
INPUT_DURATION_S=242.58   # [measured — prior session]
STEMS_DIR="$SESSION_DIR/stems"
EVAL_JSON_MBR="$SESSION_DIR/eval_mbr.json"
EVAL_JSON_BSR="$SESSION_DIR/eval_bsr.json"

# Baseline for comparison [measured — research/2026-04-12_alt-models-eval/notes.md]
BASELINE_SRR=12.79
BASELINE_XL=-4.65

cd "$REPO_ROOT"
source .venv/bin/activate

echo "=== audio-separator version ==="
audio-separator --version 2>/dev/null || python3 -c "import audio_separator; print(audio_separator.__version__)"

echo ""
echo "=== Step 1: Run Mel-Band RoFormer separation ==="
mkdir -p "$STEMS_DIR/mbr"

MBR_EXISTING=$(find "$STEMS_DIR/mbr" -iname "*vocals*.wav" ! -name "vocals.wav" | head -1)
if [[ -n "$MBR_EXISTING" ]]; then
    echo "  [skip] stem already exists: $MBR_EXISTING"
else
    MBR_START=$(python3 -c "import time; print(time.perf_counter())")
    audio-separator \
        "$INPUT" \
        --model_filename "MelBandRoformerSYHFTV3Epsilon.ckpt" \
        --output_dir "$STEMS_DIR/mbr" \
        --output_format wav \
        --single_stem "Vocals" \
        2>&1 | tee "$SESSION_DIR/mbr_separation.log"
    MBR_END=$(python3 -c "import time; print(time.perf_counter())")
    python3 - <<EOF
start = $MBR_START
end = $MBR_END
wall = end - start
rtf = wall / $INPUT_DURATION_S
print(f"MBR wall-clock: {wall:.1f}s  RTF: {rtf:.3f}x")
EOF
fi

echo ""
echo "=== Step 2: Evaluate MBR stems ==="
# evaluate_stems.py expects vocals.wav by exact name; symlink the MBR output
MBR_VOCALS_RAW=$(find "$STEMS_DIR/mbr" -iname "*vocals*.wav" ! -name "vocals.wav" | head -1)
if [[ -z "$MBR_VOCALS_RAW" ]]; then
    echo "ERROR: no vocals WAV found in $STEMS_DIR/mbr" && exit 1
fi
cp "$MBR_VOCALS_RAW" "$STEMS_DIR/mbr/vocals.wav"

python3 tests/evaluation/evaluate_stems.py \
    --mix "$INPUT" \
    --stems-dir "$STEMS_DIR/mbr" \
    --stems vocals \
    --json-out "$EVAL_JSON_MBR"

echo ""
echo "=== Step 3: Compare MBR vs baseline ==="
python3 - <<'EOF'
import json

with open("research/2026-04-26_vocal-stem-separation-alternatives/eval_mbr.json") as f:
    mbr = json.load(f)

vocals = mbr.get("stems", {}).get("vocals", {})
srr = vocals.get("stem_to_residual_ratio_db")
xl_other = vocals.get("cross_leakage_db", {}).get("other")

print(f"  Vocals SRR:          {srr:+.2f} dB  (baseline: +12.79 dB, target: ≥ +12.79 dB)")
if xl_other is not None:
    print(f"  Vocals/other XL:     {xl_other:.2f} dB  (baseline: -4.65 dB, target: ≤ -7.50 dB)")
else:
    print("  Vocals/other XL:     n/a (single-stem run; re-run with all stems for XL)")

print()
print("  NOTE: SRR is invalid for single-stem runs (residual ≈ full mix).")
print("  Bleed quality will be assessed via hallucination ratio in Step 4.")
EOF

echo ""
echo "=== Step 4: Separate Adounravel with MBR, then run eval_vocal_midi.py ==="
ADOUNRAVEL_WORKDIR="$REPO_ROOT/outputs/2026-04-26_12-34-26_Adounravel_歌いました"
ADOUNRAVEL_MIDI="$ADOUNRAVEL_WORKDIR/Adounravel_歌いました.mid"
ADOUNRAVEL_MIX="$ADOUNRAVEL_WORKDIR/extracted/eZuDklxsaRg.wav"
ADOUNRAVEL_MBR_DIR="$STEMS_DIR/mbr_adounravel"
mkdir -p "$ADOUNRAVEL_MBR_DIR"

# Separate Adounravel if not already done
ADOUNRAVEL_VOCALS_RAW=$(find "$ADOUNRAVEL_MBR_DIR" -iname "*vocals*.wav" ! -name "vocals.wav" | head -1)
if [[ -n "$ADOUNRAVEL_VOCALS_RAW" ]]; then
    echo "  [skip] Adounravel MBR stem already exists: $ADOUNRAVEL_VOCALS_RAW"
else
    echo "  Separating Adounravel mix with MBR..."
    audio-separator \
        "$ADOUNRAVEL_MIX" \
        --model_filename "MelBandRoformerSYHFTV3Epsilon.ckpt" \
        --output_dir "$ADOUNRAVEL_MBR_DIR" \
        --output_format wav \
        --single_stem "Vocals" \
        2>&1 | tee "$SESSION_DIR/mbr_adounravel_separation.log"
    ADOUNRAVEL_VOCALS_RAW=$(find "$ADOUNRAVEL_MBR_DIR" -iname "*vocals*.wav" ! -name "vocals.wav" | head -1)
fi

if [[ -z "$ADOUNRAVEL_VOCALS_RAW" ]]; then
    echo "  ⚠️  Adounravel MBR vocals stem not produced — skipping eval_vocal_midi."
elif [[ ! -f "$ADOUNRAVEL_MIDI" ]]; then
    echo "  ⚠️  Reference MIDI not found at $ADOUNRAVEL_MIDI — skipping eval_vocal_midi."
else
    echo "  Running eval_vocal_midi.py..."
    python3 scripts/eval_vocal_midi.py \
        "$ADOUNRAVEL_MIDI" \
        "$ADOUNRAVEL_VOCALS_RAW" \
        "$ADOUNRAVEL_MIX" \
        2>&1 | tee "$SESSION_DIR/eval_vocal_midi_mbr.log"
    echo ""
    echo "  Baseline hallucination ratio (htdemucs): 13.3% [measured]"
    echo "  MBR result logged to: $SESSION_DIR/eval_vocal_midi_mbr.log"
fi

echo ""
echo "=== Step 5 (optional): Run BS-RoFormer for comparison ==="
echo "(Uncomment below to run BS-RoFormer alongside MBR)"
# mkdir -p "$STEMS_DIR/bsr"
# audio-separator \
#     "$INPUT" \
#     --model_filename "bs_roformer_vocals_revive_v3e_unwa.ckpt" \
#     --output_dir "$STEMS_DIR/bsr" \
#     --output_format wav \
#     --single_stem "Vocals"
# python3 tests/evaluation/evaluate_stems.py \
#     --mix "$INPUT" \
#     --stems-dir "$STEMS_DIR/bsr" \
#     --json-out "$EVAL_JSON_BSR"

echo ""
echo "=== Done ==="
echo "Raw results: $EVAL_JSON_MBR"
echo "Separation log: $SESSION_DIR/mbr_separation.log"
