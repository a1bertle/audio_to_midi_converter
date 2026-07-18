"""
Vocal-to-MIDI bleed tolerance evaluator.

Runs basic-pitch (the project's vocal/guitar transcription backend) on the
htdemucs vocals stem and measures:

  - Note count, pitch range, mean confidence
  - Pitch distribution (octave histogram)
  - Confidence distribution (low < 0.3, mid 0.3–0.6, high > 0.6)
  - Suspect notes: pitches outside typical vocal range (C2–C6, MIDI 36–84)
    that likely originate from bleed (drums transients, bass fundamentals,
    high-frequency synth artefacts)
  - Silence-vs-note density (notes per second of non-silent audio)

These metrics characterise how much the −5.18 dB vocals/other bleed from
htdemucs affects downstream MIDI quality, without needing a ground-truth
MIDI reference.

Usage
-----
  python evaluate_transcription.py \\
      --vocals-stem path/to/vocals.wav \\
      [--json-out path/to/results.json]

Exit codes
----------
  0  success
  1  error
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = _REPO_ROOT / ".logs"
LOG_DIR.mkdir(exist_ok=True)

_fmt = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s [PID %(process)d %(threadName)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
_fh = logging.handlers.RotatingFileHandler(
    LOG_DIR / "error.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_fh.setLevel(logging.ERROR)
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stderr)
_sh.setLevel(logging.ERROR)
_sh.setFormatter(_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_fh, _sh])
log = logging.getLogger("evaluate_transcription")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Typical soprano–tenor vocal range: C2 (36) – C6 (84)
VOCAL_MIDI_LOW = 36
VOCAL_MIDI_HIGH = 84

# Confidence thresholds
CONF_LOW = 0.3
CONF_HIGH = 0.6

# Basic-pitch guitar kwargs (broader range than piano; better for vocals)
BASIC_PITCH_KWARGS = {
    "minimum_frequency": 80.0,    # ~E2 — below lowest expected vocal
    "maximum_frequency": 1400.0,  # ~F6 — above highest expected vocal
    "multiple_pitch_bends": False,
    "frame_threshold": 0.3,
    "onset_threshold": 0.4,
    "minimum_note_length": 100,   # ms
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def midi_to_note_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"


def octave_histogram(pitches: list[int]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for p in pitches:
        oct_label = f"oct{p // 12 - 1}"
        hist[oct_label] = hist.get(oct_label, 0) + 1
    return dict(sorted(hist.items()))


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def evaluate(vocals_stem: Path) -> dict:
    # ── Load audio metadata ───────────────────────────────────────────────────
    try:
        import librosa
    except ImportError:
        log.error("librosa required")
        sys.exit(1)

    print(f"\nLoading vocals stem: {vocals_stem}", flush=True)
    y, sr = librosa.load(str(vocals_stem), sr=None, mono=True)
    duration_s = len(y) / sr
    rms_db = float(20.0 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-10))
    print(f"  Duration:    {duration_s:.2f} s", flush=True)
    print(f"  Sample rate: {sr} Hz", flush=True)
    print(f"  RMS:         {rms_db:.2f} dBFS", flush=True)

    # ── Run basic-pitch ───────────────────────────────────────────────────────
    print("\nRunning basic-pitch transcription...", flush=True)
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except ImportError as e:
        log.error("basic-pitch not installed: %s", e)
        sys.exit(1)

    onnx_path = Path(str(ICASSP_2022_MODEL_PATH) + ".onnx")
    model_path = str(onnx_path) if onnx_path.exists() else ICASSP_2022_MODEL_PATH

    _model_out, _midi, note_events = predict(
        str(vocals_stem), model_path, **BASIC_PITCH_KWARGS
    )

    print(f"  Raw note events returned: {len(note_events)}", flush=True)

    # ── Parse note events ─────────────────────────────────────────────────────
    notes = []
    for ev in note_events:
        start, end, pitch, confidence = float(ev[0]), float(ev[1]), int(ev[2]), float(ev[3])
        notes.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "pitch": pitch,
            "name": midi_to_note_name(pitch),
            "confidence": round(confidence, 4),
            "duration": round(end - start, 3),
        })

    if not notes:
        print("  WARNING: no notes detected.", flush=True)
        return {"vocals_stem": str(vocals_stem), "note_count": 0}

    pitches = [n["pitch"] for n in notes]
    confidences = [n["confidence"] for n in notes]
    durations = [n["duration"] for n in notes]

    # ── Statistics ────────────────────────────────────────────────────────────
    note_count = len(notes)
    pitch_min, pitch_max = min(pitches), max(pitches)
    pitch_mean = float(np.mean(pitches))
    conf_mean = float(np.mean(confidences))
    conf_median = float(np.median(confidences))
    dur_mean = float(np.mean(durations))

    out_of_range = [n for n in notes if n["pitch"] < VOCAL_MIDI_LOW or n["pitch"] > VOCAL_MIDI_HIGH]
    in_range = [n for n in notes if VOCAL_MIDI_LOW <= n["pitch"] <= VOCAL_MIDI_HIGH]

    low_conf = [n for n in notes if n["confidence"] < CONF_LOW]
    mid_conf = [n for n in notes if CONF_LOW <= n["confidence"] < CONF_HIGH]
    high_conf = [n for n in notes if n["confidence"] >= CONF_HIGH]

    oct_hist = octave_histogram(pitches)
    notes_per_second = note_count / duration_s

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n--- Transcription Summary ---", flush=True)
    print(f"  Note count:              {note_count}", flush=True)
    print(f"  Pitch range:             {midi_to_note_name(pitch_min)} ({pitch_min}) – "
          f"{midi_to_note_name(pitch_max)} ({pitch_max})", flush=True)
    print(f"  Pitch mean:              {pitch_mean:.1f} ({midi_to_note_name(int(pitch_mean))})", flush=True)
    print(f"  Mean duration:           {dur_mean*1000:.0f} ms", flush=True)
    print(f"  Notes per second:        {notes_per_second:.2f}", flush=True)
    print(f"\n  Confidence mean:         {conf_mean:.3f}", flush=True)
    print(f"  Confidence median:       {conf_median:.3f}", flush=True)
    print(f"  Low conf (<{CONF_LOW}):       {len(low_conf)} ({100*len(low_conf)/note_count:.1f}%)", flush=True)
    print(f"  Mid conf ({CONF_LOW}–{CONF_HIGH}):     {len(mid_conf)} ({100*len(mid_conf)/note_count:.1f}%)", flush=True)
    print(f"  High conf (>{CONF_HIGH}):      {len(high_conf)} ({100*len(high_conf)/note_count:.1f}%)", flush=True)
    print(f"\n  In vocal range ({midi_to_note_name(VOCAL_MIDI_LOW)}–{midi_to_note_name(VOCAL_MIDI_HIGH)}): "
          f"{len(in_range)} ({100*len(in_range)/note_count:.1f}%)", flush=True)
    print(f"  Out of vocal range:      {len(out_of_range)} ({100*len(out_of_range)/note_count:.1f}%)", flush=True)

    print(f"\n  Octave histogram:", flush=True)
    for oct_label, count in oct_hist.items():
        bar = "█" * (count * 40 // note_count)
        print(f"    {oct_label}: {count:4d}  {bar}", flush=True)

    # Top out-of-range pitches
    if out_of_range:
        from collections import Counter
        oor_counts = Counter(n["pitch"] for n in out_of_range)
        print(f"\n  Top out-of-range pitches:", flush=True)
        for pitch, cnt in oor_counts.most_common(10):
            print(f"    {midi_to_note_name(pitch)} ({pitch}): {cnt} notes", flush=True)

    result = {
        "vocals_stem": str(vocals_stem),
        "duration_s": round(duration_s, 2),
        "sample_rate": sr,
        "rms_db": round(rms_db, 2),
        "basic_pitch_kwargs": BASIC_PITCH_KWARGS,
        "note_count": note_count,
        "notes_per_second": round(notes_per_second, 3),
        "pitch": {
            "min": pitch_min,
            "max": pitch_max,
            "mean": round(pitch_mean, 1),
            "min_name": midi_to_note_name(pitch_min),
            "max_name": midi_to_note_name(pitch_max),
        },
        "duration_ms": {
            "mean": round(dur_mean * 1000, 1),
        },
        "confidence": {
            "mean": round(conf_mean, 4),
            "median": round(conf_median, 4),
            "low_fraction": round(len(low_conf) / note_count, 4),
            "mid_fraction": round(len(mid_conf) / note_count, 4),
            "high_fraction": round(len(high_conf) / note_count, 4),
        },
        "vocal_range": {
            "low_midi": VOCAL_MIDI_LOW,
            "high_midi": VOCAL_MIDI_HIGH,
            "in_range_count": len(in_range),
            "in_range_fraction": round(len(in_range) / note_count, 4),
            "out_of_range_count": len(out_of_range),
            "out_of_range_fraction": round(len(out_of_range) / note_count, 4),
        },
        "octave_histogram": oct_hist,
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Vocal-to-MIDI bleed tolerance evaluator.")
    parser.add_argument("--vocals-stem", required=True, type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not args.vocals_stem.exists():
        log.error("Vocals stem not found: %s", args.vocals_stem)
        sys.exit(1)

    try:
        results = evaluate(args.vocals_stem)
    except Exception:
        log.exception("Evaluation failed")
        sys.exit(1)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nJSON written to: {args.json_out}", flush=True)

    print("\n--- JSON summary ---")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
