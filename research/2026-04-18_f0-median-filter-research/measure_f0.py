#!/usr/bin/env python3
"""
measure_f0.py — Measure raw F0 contour characteristics and evaluate median
filter window sizes for vibrato suppression on the vocals stem.

Usage:
    python measure_f0.py <vocals_stem.wav> [--rmvpe-checkpoint <path>]

Outputs:
    - Raw F0 statistics (vibrato rate, oscillation amplitude, pitch change rate)
    - Note count and unison repeat count for window sizes: 0 (off), 3, 5, 7, 11, 21
    - JSON summary

Requires: project .venv with audio2midi installed, scipy
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

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
logger = logging.getLogger("measure_f0")

_TARGET_SR = 16000
_HOP_S = 160.0 / _TARGET_SR  # 10 ms per frame — matches rmvpe.py [source-code]
_DEFAULT_CHECKPOINT = Path.home() / ".cache" / "audio2midi" / "rmvpe.pt"


def load_rmvpe(checkpoint: Path, device: str = "cpu"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from audio2midi.transcribers._rmvpe_impl import RMVPE0Predictor  # type: ignore
    return RMVPE0Predictor(str(checkpoint), device=device)


def semitone_snap(f0_hz: np.ndarray) -> np.ndarray:
    """Convert voiced F0 frames to nearest MIDI semitone (unvoiced → -1)."""
    lo = librosa.note_to_hz("C2")
    hi = librosa.note_to_hz("C6")
    voiced = (f0_hz > 0) & (f0_hz >= lo) & (f0_hz <= hi)
    midi = np.where(
        voiced,
        np.round(12 * np.log2(np.maximum(f0_hz, 1e-6) / 440.0) + 69).astype(int),
        -1,
    )
    return midi


def segment_notes(midi_track: np.ndarray, min_dur_frames: int) -> list[tuple[int, int, int]]:
    """Segment midi_track into (start_frame, end_frame, pitch) tuples."""
    notes = []
    in_note = False
    note_start = 0
    note_midi = 0
    for i, midi in enumerate(midi_track):
        if midi >= 0:
            if not in_note:
                in_note = True
                note_start = i
                note_midi = int(midi)
            elif int(midi) != note_midi:
                if i - note_start >= min_dur_frames:
                    notes.append((note_start, i, note_midi))
                note_start = i
                note_midi = int(midi)
        else:
            if in_note:
                if i - note_start >= min_dur_frames:
                    notes.append((note_start, i, note_midi))
                in_note = False
    return notes


def count_unison_repeats(notes: list[tuple[int, int, int]]) -> int:
    return sum(1 for i in range(len(notes) - 1) if notes[i][2] == notes[i + 1][2])


def measure_vibrato(f0_hz: np.ndarray, sr_frames: float) -> dict:
    """Measure vibrato characteristics on the raw continuous F0 contour."""
    voiced = f0_hz[f0_hz > 0]
    if len(voiced) < 10:
        return {}

    # Convert to cents relative to median for oscillation analysis
    median_hz = np.median(voiced)
    cents = 1200 * np.log2(voiced / median_hz)

    # Oscillation amplitude: std dev of cents across voiced frames
    osc_std_cents = float(np.std(cents))

    # Pitch change rate: fraction of consecutive voiced frames with > 50 cent change
    diffs = np.abs(np.diff(cents))
    rapid_change_rate = float(np.mean(diffs > 50))

    # Vibrato rate estimation via autocorrelation of cent contour
    # Typical vocal vibrato: 5–7 Hz → period 140–200 ms → 14–20 frames at 10 ms
    ac = np.correlate(cents - cents.mean(), cents - cents.mean(), mode="full")
    ac = ac[len(ac) // 2:]
    ac /= ac[0] + 1e-8
    # Look for first peak beyond lag 5 (50 ms) up to lag 30 (300 ms)
    peaks, _ = find_peaks(ac[5:30], height=0.1)
    vibrato_period_ms = float((peaks[0] + 5) * 10) if len(peaks) > 0 else float("nan")
    vibrato_rate_hz = 1000.0 / vibrato_period_ms if not np.isnan(vibrato_period_ms) else float("nan")

    return {
        "voiced_frames": int(len(voiced)),
        "osc_std_cents": round(osc_std_cents, 2),
        "rapid_change_rate_gt50cents": round(rapid_change_rate, 4),
        "vibrato_period_ms": round(vibrato_period_ms, 1),
        "vibrato_rate_hz": round(vibrato_rate_hz, 2),
    }


def evaluate_window(
    f0_raw: np.ndarray,
    window: int,
    min_dur_frames: int,
) -> dict:
    """Apply median filter of given window size and measure note segmentation."""
    if window > 1:
        # Apply per-voiced-segment to avoid smearing across silences
        f0_filtered = f0_raw.copy()
        voiced_mask = f0_raw > 0
        # Simple approach: filter entire array, then restore silence frames
        f0_filtered = median_filter(f0_raw.astype(float), size=window)
        f0_filtered[~voiced_mask] = 0.0
    else:
        f0_filtered = f0_raw.copy()

    midi_track = semitone_snap(f0_filtered)
    notes = segment_notes(midi_track, min_dur_frames)
    unison = count_unison_repeats(notes)

    # Pitch smoothness: std of frame-to-frame semitone changes on voiced frames
    voiced_midi = midi_track[midi_track >= 0]
    pitch_smoothness = float(np.std(np.diff(voiced_midi.astype(float)))) if len(voiced_midi) > 1 else 0.0

    return {
        "window_frames": window,
        "window_ms": round(window * 10, 1),
        "note_count": len(notes),
        "unison_repeats": unison,
        "pct_unison": round(100.0 * unison / max(len(notes) - 1, 1), 1),
        "pitch_smoothness_std": round(pitch_smoothness, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure F0 median filter characteristics")
    parser.add_argument("vocals_wav", help="Path to separated vocals WAV stem")
    parser.add_argument("--rmvpe-checkpoint", default=str(_DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--bpm", type=float, default=152.05,
        help="BPM for 32nd-note minimum duration calculation",
    )
    args = parser.parse_args()

    vocals_path = Path(args.vocals_wav)
    if not vocals_path.exists():
        logger.error("File not found: %s", vocals_path)
        sys.exit(1)

    checkpoint = Path(args.rmvpe_checkpoint)
    logger.info("Loading RMVPE from %s", checkpoint)
    model = load_rmvpe(checkpoint, args.device)

    logger.info("Loading audio: %s", vocals_path)
    y, _ = librosa.load(str(vocals_path), sr=_TARGET_SR, mono=True)

    logger.info("Running RMVPE pitch inference...")
    f0 = model.infer_from_audio(y, thred=0.03)

    duration_s = len(f0) * _HOP_S
    voiced_count = int(np.sum(f0 > 0))
    logger.info("F0 frames: %d (%.1f s), voiced: %d (%.1f%%)",
                len(f0), duration_s, voiced_count, 100.0 * voiced_count / len(f0))

    print("\n=== Raw F0 Statistics ===")
    vib = measure_vibrato(f0, 1.0 / _HOP_S)
    for k, v in vib.items():
        print(f"{k}: {v}")

    # Semitone change rate on raw F0
    midi_raw = semitone_snap(f0)
    voiced_midi = midi_raw[midi_raw >= 0]
    raw_diffs = np.abs(np.diff(voiced_midi.astype(float)))
    print(f"raw_semitone_change_rate_per_frame: {np.mean(raw_diffs > 0):.4f}")
    print(f"raw_mean_semitone_jump: {np.mean(raw_diffs):.4f}")

    # 32nd note minimum duration in frames
    beat_s = 60.0 / args.bpm
    min_note_s = beat_s / 8.0
    min_dur_frames = max(1, int(min_note_s / _HOP_S))
    print(f"\nbpm: {args.bpm} [assumed — from bpm_detect output]")
    print(f"min_note_duration_s: {min_note_s:.4f} [back-calc: 60/{args.bpm}/8]")
    print(f"min_dur_frames: {min_dur_frames} [back-calc]")

    # Evaluate window sizes
    windows = [1, 3, 5, 7, 11, 21, 31]
    print("\n=== Median Filter Window Sweep ===")
    print(f"{'window_ms':>12} {'notes':>8} {'unison':>8} {'pct_unison':>12} {'pitch_smooth':>14}")
    results = []
    for w in windows:
        r = evaluate_window(f0, w, min_dur_frames)
        results.append(r)
        print(f"{r['window_ms']:>12} {r['note_count']:>8} {r['unison_repeats']:>8} "
              f"{r['pct_unison']:>11}% {r['pitch_smoothness_std']:>14}")

    summary = {
        "input": str(vocals_path),
        "bpm": args.bpm,
        "min_dur_frames": min_dur_frames,
        "f0_frames": len(f0),
        "voiced_frames": voiced_count,
        "vibrato": vib,
        "window_sweep": results,
    }
    print("\n=== JSON Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
