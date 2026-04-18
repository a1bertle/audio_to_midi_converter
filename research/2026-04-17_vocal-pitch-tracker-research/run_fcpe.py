"""Phase 3: FCPE pitch tracker on htdemucs 4-stem vocals.wav."""
import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torchfcpe

sys.path.insert(0, str(Path(__file__).parent))
from pitch_to_midi import pitch_track_to_midi

DEFAULT_STEM = (
    "output/2026-04-13_escene-htdemucs-4s-wav/"
    "E.scene \uff02\u610f\u8b58\uff02 Music Video/vocals.wav"
)
TARGET_SR = 16000
HOP_S = 160.0 / TARGET_SR  # 10 ms default hop


def run(vocals_path: str, midi_out: str, json_out: str) -> dict:
    y, sr = librosa.load(vocals_path, sr=TARGET_SR, mono=True)
    duration = len(y) / TARGET_SR

    # CPU is faster than MPS+fallback for FCPE (STFT not supported natively on MPS)
    device = "cpu"
    model = torchfcpe.spawn_bundled_infer_model(device=device)

    t0 = time.perf_counter()
    wav = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    with torch.no_grad():
        f0_tensor = model.infer(
            wav,
            sr=TARGET_SR,
            decoder_mode="local_argmax",
            threshold=0.006,
        )
    elapsed = time.perf_counter() - t0
    rtf = elapsed / duration

    f0_hz = f0_tensor.squeeze().cpu().numpy()  # (frames,)
    times = np.arange(len(f0_hz)) * HOP_S

    vocal_lo = librosa.note_to_hz("C2")
    vocal_hi = librosa.note_to_hz("C6")
    voiced = (f0_hz > 0) & (f0_hz >= vocal_lo) & (f0_hz <= vocal_hi)
    # FCPE does not output per-frame confidence; use constant
    confidence = np.where(voiced, 0.8, 0.0)

    mid = pitch_track_to_midi(times, f0_hz, voiced, confidence,
                              min_note_duration_s=0.08)
    mid.save(midi_out)

    note_count = sum(
        1 for msg in mid.tracks[0]
        if msg.type == "note_on" and msg.velocity > 0
    )
    voiced_count = int(voiced.sum())

    results = {
        "tool": "fcpe",
        "input": vocals_path,
        "duration_s": round(duration, 2),
        "elapsed_s": round(elapsed, 2),
        "rtf": round(rtf, 4),
        "voiced_frames": voiced_count,
        "total_frames": len(f0_hz),
        "voiced_fraction": round(voiced_count / len(f0_hz), 4),
        "midi_notes": note_count,
        "notes_per_second": round(note_count / duration, 3),
        "device": device,
    }
    with open(json_out, "w") as fh:
        json.dump(results, fh, indent=2)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocals-stem", default=DEFAULT_STEM)
    ap.add_argument("--midi-out",
                    default="research/2026-04-17_vocal-pitch-tracker-research/fcpe_output.mid")
    ap.add_argument("--json-out",
                    default="research/2026-04-17_vocal-pitch-tracker-research/fcpe_results.json")
    args = ap.parse_args()
    r = run(args.vocals_stem, args.midi_out, args.json_out)
    print(json.dumps(r, indent=2))
