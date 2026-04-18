"""Phase 1: pYIN baseline on htdemucs 4-stem vocals.wav."""
import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pitch_to_midi import pitch_track_to_midi

DEFAULT_STEM = (
    "output/2026-04-13_escene-htdemucs-4s-wav/"
    "E.scene \uff02\u610f\u8b58\uff02 Music Video/vocals.wav"
)


def run(vocals_path: str, midi_out: str, json_out: str) -> dict:
    y, sr = librosa.load(vocals_path, sr=None, mono=True)
    duration = len(y) / sr

    t0 = time.perf_counter()
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=2048,
        hop_length=512,
    )
    elapsed = time.perf_counter() - t0
    rtf = elapsed / duration

    times = librosa.times_like(f0, sr=sr, hop_length=512)
    f0_clean = np.where(voiced_flag, f0, 0.0)

    mid = pitch_track_to_midi(times, f0_clean, voiced_flag, voiced_probs)
    mid.save(midi_out)

    note_count = sum(
        1 for msg in mid.tracks[0]
        if msg.type == "note_on" and msg.velocity > 0
    )
    voiced_count = int(voiced_flag.sum())

    results = {
        "tool": "pyin",
        "input": vocals_path,
        "duration_s": round(duration, 2),
        "elapsed_s": round(elapsed, 2),
        "rtf": round(rtf, 4),
        "voiced_frames": voiced_count,
        "total_frames": len(f0),
        "voiced_fraction": round(voiced_count / len(f0), 4),
        "midi_notes": note_count,
        "notes_per_second": round(note_count / duration, 3),
    }
    with open(json_out, "w") as fh:
        json.dump(results, fh, indent=2)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocals-stem", default=DEFAULT_STEM)
    ap.add_argument("--midi-out",
                    default="research/2026-04-17_vocal-pitch-tracker-research/pyin_output.mid")
    ap.add_argument("--json-out",
                    default="research/2026-04-17_vocal-pitch-tracker-research/pyin_results.json")
    args = ap.parse_args()
    r = run(args.vocals_stem, args.midi_out, args.json_out)
    print(json.dumps(r, indent=2))
