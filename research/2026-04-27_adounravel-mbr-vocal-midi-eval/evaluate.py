#!/usr/bin/env python3
"""
evaluate.py — MBR vocal MIDI transcription quality evaluator.

Delegates to scripts/eval_vocal_midi.py (--workdir mode) and augments with
stem audio properties (peak, RMS, spectral centroid, silence fraction).

Usage
-----
    python research/2026-04-27_adounravel-mbr-vocal-midi-eval/evaluate.py \\
        --workdir <audio2midi-output-dir>

Output
------
    JSON to stdout.  Exit non-zero on measurement failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np


def audio_properties(path: Path) -> dict:
    y, sr = librosa.load(str(path), sr=None, mono=True)
    rms = float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9))
    peak = float(20 * np.log10(np.max(np.abs(y)) + 1e-9))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    silence_frac = float(np.mean(
        librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0] < 10 ** (-60 / 20)
    ))
    return {
        "path": str(path),
        "duration_s": round(len(y) / sr, 3),
        "sample_rate": sr,
        "peak_dBFS": round(peak, 3),
        "rms_dBFS": round(rms, 3),
        "spectral_centroid_hz": round(centroid, 1),
        "silence_fraction": round(silence_frac, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    workdir = args.workdir.resolve()

    repo_root = Path(__file__).resolve().parents[2]

    # --- Vocal MIDI quality metrics ---
    print("Running eval_vocal_midi.py...", file=sys.stderr)
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "eval_vocal_midi.py"),
         "--workdir", str(workdir)],
        capture_output=True, text=True,
    )
    for line in result.stderr.splitlines():
        print(line, file=sys.stderr)
    if result.returncode != 0:
        sys.exit(f"eval_vocal_midi.py failed (exit {result.returncode})")

    midi_eval = json.loads(result.stdout)

    # Discover vocals stem path (same logic as eval_vocal_midi --workdir)
    vocals_candidates = list(workdir.glob("stems/mbr/*vocals*.wav"))
    if not vocals_candidates:
        vocals_candidates = list(workdir.glob("stems/htdemucs/*/vocals.wav"))
    vocals_path = vocals_candidates[0] if vocals_candidates else None

    mix_candidates = list(workdir.glob("extracted/*.wav"))
    mix_path = mix_candidates[0] if mix_candidates else None

    # --- Stem audio properties ---
    vocals_props = audio_properties(vocals_path) if vocals_path else None
    mix_props = audio_properties(mix_path) if mix_path else None

    print(json.dumps({
        "workdir": str(workdir),
        "vocals_stem_properties": vocals_props,
        "mix_properties": mix_props,
        "midi_eval": midi_eval,
    }, indent=2))


if __name__ == "__main__":
    main()
