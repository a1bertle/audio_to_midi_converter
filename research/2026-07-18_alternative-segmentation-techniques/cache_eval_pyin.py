#!/usr/bin/env python3
"""Cache pYIN ground truth on the exact grid scripts/eval_vocal_midi.py uses
(native sample rate, hop=512, fmax=C7), so local iteration can score
candidate MIDIs without re-running pYIN every time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def main() -> None:
    vocals_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(str(vocals_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    hop = 512
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop,
    )
    f0_midi = np.where(
        np.isfinite(f0) & voiced_flag & (f0 > 0),
        69 + 12 * np.log2(np.where(f0 > 0, f0, 1.0) / 440.0),
        np.nan,
    )
    np.save(out_dir / "eval_f0_midi.npy", f0_midi)
    np.save(out_dir / "eval_sr.npy", np.array([sr]))
    print(f"cached eval-grid pYIN: {len(f0_midi)} frames @ sr={sr}, hop={hop}")


if __name__ == "__main__":
    main()
