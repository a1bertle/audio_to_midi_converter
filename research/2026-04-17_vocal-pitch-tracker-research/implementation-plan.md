# Implementation Plan: Vocal Pitch Tracker Evaluation

**Date:** 2026-04-17
**Session:** `research/2026-04-17_vocal-pitch-tracker-research/`
**Status:** READY — pending execution

## Objective

Identify a viable replacement for the htdemucs + basic-pitch vocal MIDI pipeline. The current
pipeline produces subjectively unusable MIDI because the htdemucs vocals stem loses vocal
fundamentals and lower formants (centroid 4400–4800 Hz vs expected 1500–3000 Hz). Two paths:

1. **Direct-on-mix trackers** (SPICE, RMVPE) — no stem separation needed; designed for polyphony
2. **Higher-accuracy stem-based trackers** (PESTO, FCPE) — replace basic-pitch on existing stem

---

## Phase 0 — Shared Infrastructure: PitchToMIDI Post-Processor

All pitch-track tools (pYIN, SPICE, PESTO, FCPE, RMVPE, CREPE) output frame-level F0 arrays, not
MIDI. A shared `PitchToMIDI` post-processor must be implemented once before running evaluations.

### File: `research/2026-04-17_vocal-pitch-tracker-research/pitch_to_midi.py`

```python
"""Pitch track → MIDI note conversion utility.

Input:  f0 array (Hz), times array (seconds), optional confidence array
Output: mido.MidiFile with note events
"""
import numpy as np
import mido


def f0_to_midi_note(f0_hz: float) -> int:
    """Convert Hz to MIDI note number (round to nearest semitone)."""
    return int(round(12 * np.log2(f0_hz / 440.0) + 69))


def pitch_track_to_midi(
    times: np.ndarray,
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    confidence: np.ndarray | None = None,
    min_note_duration_s: float = 0.08,
    velocity_scale: float = 80.0,
    bpm: float = 120.0,
) -> mido.MidiFile:
    """
    Convert a pitch track (times, f0_hz, voiced) to a mido MidiFile.

    Args:
        times:              Frame timestamps in seconds (N,)
        f0_hz:              F0 estimates in Hz (N,); 0.0 or NaN = unvoiced
        voiced:             Boolean voiced flag (N,)
        confidence:         Optional confidence per frame (N,); used for velocity
        min_note_duration_s: Drop notes shorter than this (removes spurious events)
        velocity_scale:     MIDI velocity (0–127) for voiced frames with confidence=1.0
        bpm:                Tempo for the output MIDI file
    Returns:
        mido.MidiFile with one track, channel 0
    """
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    if confidence is None:
        confidence = np.ones_like(f0_hz)

    notes: list[tuple[float, float, int, int]] = []  # (start_s, end_s, midi, velocity)
    in_note = False
    note_start = 0.0
    note_midi = 0
    note_conf_sum = 0.0
    note_frame_count = 0

    for i, (t, f0, v, conf) in enumerate(zip(times, f0_hz, voiced, confidence)):
        if v and f0 > 0:
            midi = f0_to_midi_note(f0)
            if not in_note:
                in_note = True
                note_start = t
                note_midi = midi
                note_conf_sum = conf
                note_frame_count = 1
            else:
                if midi != note_midi:
                    # Pitch changed — close current note, start new one
                    duration = t - note_start
                    if duration >= min_note_duration_s:
                        avg_conf = note_conf_sum / note_frame_count
                        vel = max(1, min(127, int(avg_conf * velocity_scale)))
                        notes.append((note_start, t, note_midi, vel))
                    note_start = t
                    note_midi = midi
                    note_conf_sum = conf
                    note_frame_count = 1
                else:
                    note_conf_sum += conf
                    note_frame_count += 1
        else:
            if in_note:
                duration = t - note_start
                if duration >= min_note_duration_s:
                    avg_conf = note_conf_sum / note_frame_count
                    vel = max(1, min(127, int(avg_conf * velocity_scale)))
                    notes.append((note_start, t, note_midi, vel))
                in_note = False

    # Convert seconds to ticks and emit MIDI events
    ticks_per_second = 480 * (bpm / 60.0)
    current_tick = 0
    events: list[tuple[int, mido.Message]] = []
    for start_s, end_s, note, vel in notes:
        start_tick = int(start_s * ticks_per_second)
        end_tick = int(end_s * ticks_per_second)
        events.append((start_tick, mido.Message("note_on", note=note, velocity=vel, time=0)))
        events.append((end_tick, mido.Message("note_off", note=note, velocity=0, time=0)))

    events.sort(key=lambda x: x[0])
    current_tick = 0
    for abs_tick, msg in events:
        msg.time = abs_tick - current_tick
        track.append(msg)
        current_tick = abs_tick

    return mid
```

---

## Phase 1 — Baseline: pYIN on htdemucs stem

### Install
None — librosa 0.11.0 already in venv.

### Script: `research/2026-04-17_vocal-pitch-tracker-research/run_pyin.py`

```python
"""Baseline: pYIN on htdemucs 4-stem vocals.wav."""
import argparse
import json
import time
import numpy as np
import librosa
import soundfile as sf
from pitch_to_midi import pitch_track_to_midi

VOCALS_STEM = (
    "output/2026-04-13_escene-htdemucs-4s-wav/"
    "E.scene \uff02\u610f\u8b58\uff02 Music Video/vocals.wav"
)

def run(vocals_path: str, midi_out: str, json_out: str) -> dict:
    y, sr = librosa.load(vocals_path, sr=None, mono=True)
    t_start = time.perf_counter()
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=2048,
    )
    elapsed = time.perf_counter() - t_start
    duration = len(y) / sr
    rtf = elapsed / duration

    times = librosa.times_like(f0, sr=sr, hop_length=512)
    f0_clean = np.where(voiced_flag, f0, 0.0)
    mid = pitch_track_to_midi(times, f0_clean, voiced_flag, voiced_probs)
    mid.save(midi_out)

    voiced_count = voiced_flag.sum()
    results = {
        "tool": "pyin",
        "input": vocals_path,
        "duration_s": round(duration, 2),
        "elapsed_s": round(elapsed, 2),
        "rtf": round(rtf, 4),
        "voiced_frames": int(voiced_count),
        "total_frames": len(f0),
        "voiced_fraction": round(float(voiced_count) / len(f0), 4),
        "midi_notes": sum(
            1 for t in mid.tracks[0] if hasattr(t, "type") and t.type == "note_on" and t.velocity > 0
        ),
    }
    with open(json_out, "w") as fh:
        json.dump(results, fh, indent=2)
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocals-stem", default=VOCALS_STEM)
    ap.add_argument("--midi-out", default="research/2026-04-17_vocal-pitch-tracker-research/pyin_output.mid")
    ap.add_argument("--json-out", default="research/2026-04-17_vocal-pitch-tracker-research/pyin_results.json")
    args = ap.parse_args()
    r = run(args.vocals_stem, args.midi_out, args.json_out)
    print(json.dumps(r, indent=2))
```

---

## Phase 2 — SPICE on Raw Mix

### Install
```bash
pip install tensorflow tensorflow-hub
```

### Script: `research/2026-04-17_vocal-pitch-tracker-research/run_spice.py`

```python
"""SPICE vocal pitch tracker on raw mix (no stem separation)."""
import argparse
import json
import time
import numpy as np
import librosa
import tensorflow_hub as hub
from pitch_to_midi import pitch_track_to_midi

SPICE_MODEL_URL = "https://tfhub.dev/google/spice/2"
MIX_PATH = "/tmp/yt_download/E.scene \uff02\u610f\u8b58\uff02 Music Video.mp3"

def run(mix_path: str, midi_out: str, json_out: str) -> dict:
    model = hub.load(SPICE_MODEL_URL)
    y, sr = librosa.load(mix_path, sr=16000, mono=True)  # SPICE expects 16 kHz
    # Normalize to [-1, 1]
    y = y / (np.max(np.abs(y)) + 1e-8)

    t_start = time.perf_counter()
    output = model.signatures["serving_default"](tf.constant(y, dtype=tf.float32))
    elapsed = time.perf_counter() - t_start
    duration = len(y) / 16000
    rtf = elapsed / duration

    pitches = output["pitch"].numpy()        # [0, 1] normalized
    uncertainties = output["uncertainty"].numpy()
    confidences = 1.0 - uncertainties

    # Convert SPICE pitch [0,1] to Hz using linear mapping from paper
    # pitch_hz = 10 * 2^(pitch * 7.5)   (spans ~10 Hz to ~10 kHz)
    f0_hz = 10.0 * (2.0 ** (pitches * 7.5))

    # Voice/unvoiced: uncertainty < 0.9 is voiced
    voiced = uncertainties < 0.9
    # Filter to vocal range C2–C6
    vocal_lo = librosa.note_to_hz("C2")
    vocal_hi = librosa.note_to_hz("C6")
    voiced = voiced & (f0_hz >= vocal_lo) & (f0_hz <= vocal_hi)

    hop_size_s = 0.032  # SPICE outputs one value per 32 ms
    times = np.arange(len(pitches)) * hop_size_s
    mid = pitch_track_to_midi(times, f0_hz, voiced, confidences)
    mid.save(midi_out)

    results = {
        "tool": "spice",
        "input": mix_path,
        "duration_s": round(duration, 2),
        "elapsed_s": round(elapsed, 2),
        "rtf": round(rtf, 4),
        "voiced_frames": int(voiced.sum()),
        "total_frames": len(pitches),
        "voiced_fraction": round(float(voiced.sum()) / len(pitches), 4),
        "midi_notes": sum(
            1 for t in mid.tracks[0] if hasattr(t, "type") and t.type == "note_on" and t.velocity > 0
        ),
    }
    with open(json_out, "w") as fh:
        json.dump(results, fh, indent=2)
    return results

if __name__ == "__main__":
    import tensorflow as tf
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default=MIX_PATH)
    ap.add_argument("--midi-out", default="research/2026-04-17_vocal-pitch-tracker-research/spice_output.mid")
    ap.add_argument("--json-out", default="research/2026-04-17_vocal-pitch-tracker-research/spice_results.json")
    args = ap.parse_args()
    r = run(args.mix, args.midi_out, args.json_out)
    print(json.dumps(r, indent=2))
```

---

## Phase 3 — PESTO on htdemucs Stem

### Install
```bash
pip install pesto-pitch
```

### Script: `research/2026-04-17_vocal-pitch-tracker-research/run_pesto.py`

```python
"""PESTO vocal pitch tracker on htdemucs vocals stem."""
import argparse
import json
import time
import numpy as np
import librosa
import pesto
from pitch_to_midi import pitch_track_to_midi

VOCALS_STEM = (
    "output/2026-04-13_escene-htdemucs-4s-wav/"
    "E.scene \uff02\u610f\u8b58\uff02 Music Video/vocals.wav"
)

def run(vocals_path: str, midi_out: str, json_out: str) -> dict:
    y, sr = librosa.load(vocals_path, sr=None, mono=True)
    duration = len(y) / sr

    t_start = time.perf_counter()
    # pesto.predict returns (times, pitches_hz, confidences, activation)
    # step_size in milliseconds
    times, f0_hz, confidences, _ = pesto.predict(y, sr, step_size=10.0)
    elapsed = time.perf_counter() - t_start
    rtf = elapsed / duration

    voiced = confidences > 0.5
    vocal_lo = librosa.note_to_hz("C2")
    vocal_hi = librosa.note_to_hz("C6")
    voiced = voiced & (f0_hz >= vocal_lo) & (f0_hz <= vocal_hi)

    mid = pitch_track_to_midi(times, f0_hz, voiced, confidences)
    mid.save(midi_out)

    results = {
        "tool": "pesto",
        "input": vocals_path,
        "duration_s": round(duration, 2),
        "elapsed_s": round(elapsed, 2),
        "rtf": round(rtf, 4),
        "voiced_frames": int(voiced.sum()),
        "total_frames": len(f0_hz),
        "voiced_fraction": round(float(voiced.sum()) / len(f0_hz), 4),
        "midi_notes": sum(
            1 for t in mid.tracks[0] if hasattr(t, "type") and t.type == "note_on" and t.velocity > 0
        ),
        "mean_confidence": round(float(confidences[voiced].mean()) if voiced.sum() > 0 else 0.0, 4),
    }
    with open(json_out, "w") as fh:
        json.dump(results, fh, indent=2)
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocals-stem", default=VOCALS_STEM)
    ap.add_argument("--midi-out", default="research/2026-04-17_vocal-pitch-tracker-research/pesto_output.mid")
    ap.add_argument("--json-out", default="research/2026-04-17_vocal-pitch-tracker-research/pesto_results.json")
    args = ap.parse_args()
    r = run(args.vocals_stem, args.midi_out, args.json_out)
    print(json.dumps(r, indent=2))
```

---

## Phase 4 — FCPE on htdemucs Stem

### Install
```bash
pip install torchfcpe
```

### Script: `research/2026-04-17_vocal-pitch-tracker-research/run_fcpe.py`

```python
"""FCPE vocal pitch tracker on htdemucs vocals stem."""
import argparse
import json
import time
import numpy as np
import librosa
import torch
import torchfcpe
from pitch_to_midi import pitch_track_to_midi

VOCALS_STEM = (
    "output/2026-04-13_escene-htdemucs-4s-wav/"
    "E.scene \uff02\u610f\u8b58\uff02 Music Video/vocals.wav"
)

def run(vocals_path: str, midi_out: str, json_out: str) -> dict:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    y, sr = librosa.load(vocals_path, sr=16000, mono=True)
    duration = len(y) / 16000

    wav_tensor = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    model = torchfcpe.spawn_bundled_infer_model(device=device)

    t_start = time.perf_counter()
    # f0: (1, T', 1) tensor
    f0_tensor = model.infer(
        wav_tensor.to(device),
        sr=16000,
        decoder_mode="local_argmax",
        threshold=0.006,
    )
    elapsed = time.perf_counter() - t_start
    rtf = elapsed / duration

    f0_hz = f0_tensor.squeeze().cpu().numpy()
    # torchfcpe returns 0.0 for unvoiced frames
    voiced = f0_hz > 0.0
    vocal_lo = librosa.note_to_hz("C2")
    vocal_hi = librosa.note_to_hz("C6")
    voiced = voiced & (f0_hz >= vocal_lo) & (f0_hz <= vocal_hi)
    hop_s = 160.0 / 16000.0  # 10 ms default hop
    times = np.arange(len(f0_hz)) * hop_s

    confidence = np.where(voiced, 0.8, 0.0)  # FCPE does not output confidence; use fixed value
    mid = pitch_track_to_midi(times, f0_hz, voiced, confidence)
    mid.save(midi_out)

    results = {
        "tool": "fcpe",
        "input": vocals_path,
        "duration_s": round(duration, 2),
        "elapsed_s": round(elapsed, 2),
        "rtf": round(rtf, 4),
        "voiced_frames": int(voiced.sum()),
        "total_frames": len(f0_hz),
        "voiced_fraction": round(float(voiced.sum()) / len(f0_hz), 4),
        "midi_notes": sum(
            1 for t in mid.tracks[0] if hasattr(t, "type") and t.type == "note_on" and t.velocity > 0
        ),
    }
    with open(json_out, "w") as fh:
        json.dump(results, fh, indent=2)
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocals-stem", default=VOCALS_STEM)
    ap.add_argument("--midi-out", default="research/2026-04-17_vocal-pitch-tracker-research/fcpe_output.mid")
    ap.add_argument("--json-out", default="research/2026-04-17_vocal-pitch-tracker-research/fcpe_results.json")
    args = ap.parse_args()
    r = run(args.vocals_stem, args.midi_out, args.json_out)
    print(json.dumps(r, indent=2))
```

---

## Validation Criteria

After running each tool, record the following in `results.md`:

| Metric | Target | Rationale |
|--------|--------|-----------|
| MIDI note count | 200–800 notes | Plausible for 4-min vocal track; basic-pitch baseline: 650 |
| Notes per second | 1–5 notes/s | Typical sung phrase rate |
| Voiced fraction | 40–70% | Track has 36.6% silence [measured]; voiced should be majority of non-silent |
| Pitch range (MIDI) | MIDI 36–84 (C2–C6) | Expected vocal range for J-pop female lead |
| RTF | ≤ 3.0× | Project constraint; ~12 min per 4-min track |
| Subjective MIDI quality | Better than basic-pitch baseline | Primary quality criterion (no ground-truth MIDI) |

---

## Success Criteria

At least one tool must produce MIDI that:
1. Passes all quantitative thresholds in the table above [measurable]
2. Is subjectively better than the basic-pitch htdemucs-stem baseline (listener assessment) [measured by user]
3. Runs within the 3× RTF budget on M1 [measurable]

---

## Files to Produce

```
research/2026-04-17_vocal-pitch-tracker-research/
├── notes.md                    ✓ complete
├── implementation-plan.md      ✓ complete
├── pitch_to_midi.py            (Phase 0 — implement before running any phase)
├── run_pyin.py                 (Phase 1)
├── run_spice.py                (Phase 2)
├── run_pesto.py                (Phase 3)
├── run_fcpe.py                 (Phase 4)
├── validate.sh                 (orchestrates all phases)
├── pyin_results.json           (after Phase 1)
├── spice_results.json          (after Phase 2)
├── pesto_results.json          (after Phase 3)
├── fcpe_results.json           (after Phase 4)
└── results.md                  (comparison table; after all phases)
```
