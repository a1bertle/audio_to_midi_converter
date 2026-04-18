"""Phase 2: PESTO pitch tracker on htdemucs 4-stem vocals.wav."""
import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import pesto

sys.path.insert(0, str(Path(__file__).parent))
from pitch_to_midi import pitch_track_to_midi

DEFAULT_STEM = (
    "output/2026-04-13_escene-htdemucs-4s-wav/"
    "E.scene \uff02\u610f\u8b58\uff02 Music Video/vocals.wav"
)


def run(vocals_path: str, midi_out: str, json_out: str) -> dict:
    y, sr = librosa.load(vocals_path, sr=None, mono=True)
    duration = len(y) / sr

    # Split into 60s segments to avoid OOM on M1
    segment_s = 60
    segment_samples = segment_s * sr
    all_times, all_f0, all_conf = [], [], []
    t0 = time.perf_counter()
    offset = 0
    while offset < len(y):
        chunk = y[offset:offset + segment_samples]
        chunk_duration = len(chunk) / sr
        t_seg, f0_seg, conf_seg, _ = pesto.predict(
            torch.from_numpy(chunk).unsqueeze(0),
            sr,
            step_size=10,
            num_chunks=max(1, int(chunk_duration / 10)),
        )
        t_np = t_seg.numpy().flatten() / 1000.0 + (offset / sr)  # ms → seconds
        all_times.append(t_np)
        all_f0.append(f0_seg.numpy().flatten())
        all_conf.append(conf_seg.numpy().flatten())
        offset += segment_samples
    elapsed = time.perf_counter() - t0
    rtf = elapsed / duration

    times = np.concatenate(all_times)
    f0_hz = np.concatenate(all_f0)
    confidences = np.concatenate(all_conf)

    vocal_lo = librosa.note_to_hz("C2")
    vocal_hi = librosa.note_to_hz("C6")
    voiced = (confidences > 0.5) & (f0_hz >= vocal_lo) & (f0_hz <= vocal_hi)

    mid = pitch_track_to_midi(times, f0_hz, voiced, confidences,
                              min_note_duration_s=0.12)
    mid.save(midi_out)

    note_count = sum(
        1 for msg in mid.tracks[0]
        if msg.type == "note_on" and msg.velocity > 0
    )
    voiced_count = int(voiced.sum())
    mean_conf = float(confidences[voiced].mean()) if voiced_count > 0 else 0.0

    results = {
        "tool": "pesto",
        "input": vocals_path,
        "duration_s": round(duration, 2),
        "elapsed_s": round(elapsed, 2),
        "rtf": round(rtf, 4),
        "voiced_frames": voiced_count,
        "total_frames": len(f0_hz),
        "voiced_fraction": round(voiced_count / len(f0_hz), 4),
        "mean_confidence": round(mean_conf, 4),
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
                    default="research/2026-04-17_vocal-pitch-tracker-research/pesto_output.mid")
    ap.add_argument("--json-out",
                    default="research/2026-04-17_vocal-pitch-tracker-research/pesto_results.json")
    args = ap.parse_args()
    r = run(args.vocals_stem, args.midi_out, args.json_out)
    print(json.dumps(r, indent=2))
