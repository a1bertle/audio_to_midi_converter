#!/usr/bin/env bash
# Benchmark script: alternate stem separation model evaluation
# Research session: research/2026-04-12_alt-models-eval/
# Date: 2026-04-12
# Evaluates: htdemucs_ft, htdemucs_4s+residual, Open-Unmix, htdemucs_6s+spectral-mask
# Benchmark input: inputs/Foals - My Number (Official Audio).mp3

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INPUT="$REPO_ROOT/inputs/Foals - My Number (Official Audio).mp3"
EVAL_SCRIPT="$REPO_ROOT/tests/evaluation/evaluate_stems.py"
OUT_ROOT="/tmp/demucs_alt_eval"
SESSION_DIR="$REPO_ROOT/research/2026-04-12_alt-models-eval"

# SSL fix (same as prior benchmark session)
export SSL_CERT_FILE="$( source "$REPO_ROOT/.venv/bin/activate" && python3 -c 'import certifi; print(certifi.where())' )"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

source "$REPO_ROOT/.venv/bin/activate"

run_and_eval() {
    local label="$1"
    local model="$2"
    local out_dir="$OUT_ROOT/$label"
    local stems_dir_pattern="$3"   # glob suffix under out_dir after demucs creates subdir
    local device="${4:-mps}"

    echo ""
    echo "========================================"
    echo "LABEL: $label  MODEL: $model  DEVICE: $device"
    echo "========================================"

    mkdir -p "$out_dir"

    /usr/bin/time -l \
        python3 -m demucs \
            --name "$model" \
            --device "$device" \
            --out "$out_dir" \
            "$INPUT" \
        2>&1 | tee "$SESSION_DIR/${label}_separation.log"

    # Find the stems directory (demucs creates model-named subdir)
    local stems_dir
    stems_dir=$(find "$out_dir" -mindepth 2 -maxdepth 2 -type d | head -1)
    echo "STEMS_DIR: $stems_dir"

    echo ""
    echo "--- Evaluation: $label ---"
    python3 "$EVAL_SCRIPT" \
        --mix "$INPUT" \
        --stems-dir "$stems_dir" \
        --json-out "$SESSION_DIR/${label}_eval.json" \
        2>&1 | tee "$SESSION_DIR/${label}_eval.log"
}

# --- Run 1: htdemucs_ft (fine-tuned 4-stem) ---
run_and_eval "htdemucs_ft" "htdemucs_ft" "" "mps"

# --- Run 2: htdemucs (4-stem, non-fine-tuned baseline for residual subtraction) ---
run_and_eval "htdemucs_4s" "htdemucs" "" "mps"

# --- Run 3: mdx_extra (MDX architecture, 4-stem, alternative to Demucs) ---
run_and_eval "mdx_extra" "mdx_extra" "" "cpu"

echo ""
echo "All runs complete. Results in $SESSION_DIR/"
