#!/usr/bin/env python3
"""Run Basic Pitch on an existing vocal stem and write a comparison MIDI.

Eval-only script for this research session. Does not modify the accepted
pipeline or any existing MIDI output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import scipy.signal

if not hasattr(scipy.signal, "gaussian"):
    # basic-pitch 0.3.0 calls the pre-1.13 scipy.signal.gaussian API, which
    # moved to scipy.signal.windows.gaussian. Shim it for this venv's scipy.
    from scipy.signal.windows import gaussian as _gaussian

    scipy.signal.gaussian = _gaussian

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: run_basic_pitch.py <vocals_wav> <out_midi>", file=sys.stderr)
        raise SystemExit(2)
    vocals_wav = Path(sys.argv[1])
    out_midi = Path(sys.argv[2])

    onnx_path = Path(str(ICASSP_2022_MODEL_PATH) + ".onnx")
    model_path = str(onnx_path) if onnx_path.exists() else ICASSP_2022_MODEL_PATH

    # Vocal pitch range bounds matched to rmvpe.py's existing C2-C6 bounds.
    _model_output, midi_data, note_events = predict(
        str(vocals_wav),
        model_path,
        minimum_frequency=65.0,   # ~C2
        maximum_frequency=1050.0,  # ~C6
    )
    midi_data.write(str(out_midi))
    print(f"wrote {out_midi} ({len(note_events)} notes)")


if __name__ == "__main__":
    main()
