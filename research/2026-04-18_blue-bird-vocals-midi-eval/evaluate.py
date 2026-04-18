#!/usr/bin/env python3
"""
evaluate.py — MIDI transcription quality assessment for vocal-to-MIDI output.

Usage:
    python evaluate.py <path/to/file.mid>

Metrics:
    - File properties (duration, tempo, track count, note count)
    - Note density (notes per second)
    - Polyphony: simultaneous notes (should be 0 for monophonic vocal)
    - Pitch distribution: min/max/mean pitch, out-of-vocal-range notes
    - Duration statistics: min/max/mean/std note duration
    - Quantization regularity: deviation from nearest 32nd-note grid
    - Pitch gap analysis: semitone intervals between consecutive notes

Vocal range reference: C3 (MIDI 48) to C6 (MIDI 84)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pretty_midi

VOCAL_LOW = 48   # C3
VOCAL_HIGH = 84  # C6
SOPRANO_LOW = 60  # C4
SOPRANO_HIGH = 81  # A5

LOG_DIR = Path(__file__).resolve().parents[2] / ".logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOG_DIR / "error.log"),
    ],
)
logger = logging.getLogger("evaluate")


def compute_polyphony_events(notes: list) -> int:
    """Count number of time points where >1 note is simultaneously active."""
    events = []
    for n in notes:
        events.append((n.start, +1))
        events.append((n.end, -1))
    events.sort(key=lambda x: (x[0], x[1]))
    active = 0
    overlap_count = 0
    for _, delta in events:
        active += delta
        if active > 1:
            overlap_count += 1
    return overlap_count


def nearest_grid_deviation(onset: float, bpm: float) -> float:
    """Return deviation in ms from nearest 32nd-note grid position."""
    beat_dur = 60.0 / bpm
    grid = beat_dur / 8  # 32nd note = beat/8
    mod = onset % grid
    deviation = min(mod, grid - mod)
    return deviation * 1000  # ms


def main(midi_path: str) -> None:
    path = Path(midi_path).resolve()
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    print(f"\n=== MIDI Evaluation: {path.name} ===\n")

    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:
        logger.error("Failed to parse MIDI: %s", exc, exc_info=True)
        sys.exit(1)

    # --- File properties ---
    duration = pm.get_end_time()
    tempo_change_times, tempos_arr = pm.get_tempo_changes()
    initial_tempo = float(tempos_arr[0]) if len(tempos_arr) > 0 else 120.0
    track_count = len(pm.instruments)
    file_size = path.stat().st_size

    all_notes = []
    for inst in pm.instruments:
        all_notes.extend(inst.notes)
    all_notes.sort(key=lambda n: n.start)
    note_count = len(all_notes)

    print("--- File Properties ---")
    print(f"file_size_bytes: {file_size}")
    print(f"duration_s: {duration:.3f}")
    print(f"estimated_tempo_bpm: {initial_tempo:.2f}")
    print(f"track_count: {track_count}")
    print(f"note_count: {note_count}")
    print(f"note_density_per_s: {note_count / duration:.3f}")

    # --- Polyphony ---
    poly_events = compute_polyphony_events(all_notes)
    print(f"\n--- Polyphony ---")
    print(f"simultaneous_note_events: {poly_events}")

    # --- Pitch distribution ---
    pitches = np.array([n.pitch for n in all_notes])
    out_of_vocal_range = int(np.sum((pitches < VOCAL_LOW) | (pitches > VOCAL_HIGH)))
    out_of_soprano_range = int(np.sum((pitches < SOPRANO_LOW) | (pitches > SOPRANO_HIGH)))
    unique_pitches = len(np.unique(pitches))

    print(f"\n--- Pitch Distribution ---")
    print(f"pitch_min_midi: {int(pitches.min())}")
    print(f"pitch_max_midi: {int(pitches.max())}")
    print(f"pitch_mean_midi: {pitches.mean():.2f}")
    print(f"pitch_std_midi: {pitches.std():.2f}")
    print(f"unique_pitches: {unique_pitches}")
    print(f"notes_out_of_vocal_range_c3_c6: {out_of_vocal_range}")
    print(f"notes_out_of_soprano_range_c4_a5: {out_of_soprano_range}")
    print(f"pct_out_of_vocal_range: {100.0 * out_of_vocal_range / note_count:.1f}%")

    # --- Duration statistics ---
    durations = np.array([n.end - n.start for n in all_notes])
    print(f"\n--- Note Duration Statistics ---")
    print(f"duration_min_s: {durations.min():.4f}")
    print(f"duration_max_s: {durations.max():.4f}")
    print(f"duration_mean_s: {durations.mean():.4f}")
    print(f"duration_std_s: {durations.std():.4f}")
    print(f"duration_median_s: {np.median(durations):.4f}")

    # --- Quantization regularity ---
    # Use the tempo embedded in the MIDI file (set by the pipeline).
    assumed_bpm = initial_tempo if initial_tempo > 0 else 120.0
    onsets = np.array([n.start for n in all_notes])
    deviations = np.array([nearest_grid_deviation(o, assumed_bpm) for o in onsets])
    print(f"\n--- Quantization (32nd-note grid @ {assumed_bpm} BPM) ---")
    print(f"grid_interval_ms: {(60.0 / assumed_bpm / 8) * 1000:.2f}")
    print(f"onset_deviation_mean_ms: {deviations.mean():.2f}")
    print(f"onset_deviation_max_ms: {deviations.max():.2f}")
    print(f"onset_deviation_std_ms: {deviations.std():.2f}")
    # Perfect quantization = 0 deviation; high deviation = grid mismatch
    pct_off_grid = 100.0 * np.sum(deviations > 10) / len(deviations)
    print(f"pct_onsets_off_grid_gt10ms: {pct_off_grid:.1f}%")

    # --- Interval analysis ---
    if len(pitches) > 1:
        intervals = np.abs(np.diff(pitches.astype(int)))
        large_leaps = int(np.sum(intervals > 12))  # > 1 octave
        unison_repeats = int(np.sum(intervals == 0))
        print(f"\n--- Melodic Intervals ---")
        print(f"interval_mean_semitones: {intervals.mean():.2f}")
        print(f"interval_max_semitones: {int(intervals.max())}")
        print(f"leaps_greater_than_octave: {large_leaps}")
        print(f"unison_repeats: {unison_repeats}")
        print(f"pct_large_leaps: {100.0 * large_leaps / len(intervals):.1f}%")

    # --- JSON summary ---
    summary = {
        "file": path.name,
        "duration_s": round(duration, 3),
        "estimated_tempo_bpm": round(initial_tempo, 2),
        "note_count": note_count,
        "note_density_per_s": round(note_count / duration, 3),
        "polyphony_events": poly_events,
        "pitch_min": int(pitches.min()),
        "pitch_max": int(pitches.max()),
        "pitch_mean": round(float(pitches.mean()), 2),
        "unique_pitches": unique_pitches,
        "notes_out_of_vocal_range": out_of_vocal_range,
        "pct_out_of_vocal_range": round(100.0 * out_of_vocal_range / note_count, 1),
        "duration_mean_s": round(float(durations.mean()), 4),
        "duration_std_s": round(float(durations.std()), 4),
        "onset_deviation_mean_ms": round(float(deviations.mean()), 2),
        "pct_off_grid_gt10ms": round(pct_off_grid, 1),
    }

    print(f"\n--- JSON Summary ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path/to/file.mid>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
