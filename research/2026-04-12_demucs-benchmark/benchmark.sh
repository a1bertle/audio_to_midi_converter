#!/usr/bin/env bash
# benchmark.sh — Demucs htdemucs_6s benchmark on M1 (MPS + CPU)
#
# Usage:
#   ./benchmark.sh --input <audio_file> [--device mps|cpu|both]
#
# Outputs:
#   - Separated stems under /tmp/demucs_bench/<device>/
#   - Timing and memory printed to stdout (captured by caller)
#
# Requirements:
#   - .venv activated (or demucs available on PATH)
#   - demucs>=4.0.0 installed

set -euo pipefail

INPUT=""
DEVICE="both"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input) INPUT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$INPUT" ]]; then
    echo "ERROR: --input <audio_file> is required" >&2
    exit 1
fi

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: Input file not found: $INPUT" >&2
    exit 1
fi

# ── Environment info ──────────────────────────────────────────────────────────
echo "=== Environment ==="
python3 -c "import sys; print('Python:', sys.version)"
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
python3 -c "import demucs; print('Demucs:', demucs.__version__)" 2>/dev/null \
    || python3 -c "import importlib.metadata; print('Demucs:', importlib.metadata.version('demucs'))"
echo ""

# ── Audio duration ────────────────────────────────────────────────────────────
DURATION_S=$(python3 - <<'PYEOF'
import sys, os
try:
    import librosa
    path = os.environ.get("BENCH_INPUT", "")
    y, sr = librosa.load(path, sr=None, mono=True)
    print(f"{len(y)/sr:.2f}")
except Exception as e:
    print(f"0.0  # ERROR: {e}")
PYEOF
)
BENCH_INPUT="$INPUT" python3 - <<PYEOF
import librosa, os
path = os.environ.get("BENCH_INPUT", "$INPUT")
y, sr = librosa.load(path, sr=None, mono=True)
duration = len(y) / sr
print(f"Input: $INPUT")
print(f"Audio duration: {duration:.2f} s  ({duration/60:.2f} min)")
PYEOF
echo ""

# ── Run one device ────────────────────────────────────────────────────────────
run_device() {
    local device="$1"
    local out_dir="/tmp/demucs_bench/${device}"
    rm -rf "$out_dir"
    mkdir -p "$out_dir"

    echo "=== Run: device=${device} ==="
    echo "Start time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # /usr/bin/time -l captures peak RSS on macOS
    /usr/bin/time -l python3 -m demucs \
        --name htdemucs_6s \
        --device "$device" \
        --out "$out_dir" \
        --mp3 \
        "$INPUT" \
        2>&1

    echo "End time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""
    echo "Output files:"
    find "$out_dir" -type f | sort | while read -r f; do
        size=$(du -sh "$f" | cut -f1)
        echo "  $size  $f"
    done
    echo ""
}

case "$DEVICE" in
    mps)  run_device mps ;;
    cpu)  run_device cpu ;;
    both) run_device mps; run_device cpu ;;
    *)
        echo "ERROR: --device must be mps, cpu, or both" >&2
        exit 1
        ;;
esac
