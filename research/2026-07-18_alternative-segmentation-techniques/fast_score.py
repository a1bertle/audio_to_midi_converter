#!/usr/bin/env python3
"""Fast local scorer replicating scripts/eval_vocal_midi.py's coverage,
hallucination, pitch-accuracy, and IOI-CV metrics, using the cached
eval-grid pYIN array so no pYIN recomputation is needed per iteration.

Median note duration is computed directly from the note list (matches
midi_properties() in the eval script).

Always cross-check a final candidate against the real eval_vocal_midi.py
before trusting these numbers for a decision -- this is an iteration tool,
not a replacement for the canonical eval.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def score_notes(
    notes: list[tuple[float, float, int]],
    eval_f0_midi: np.ndarray,
    sr: int,
    hop: int = 512,
) -> dict:
    n_frames = len(eval_f0_midi)
    midi_roll = np.full(n_frames, np.nan)
    for start, end, pitch in notes:
        s = int(round(start * sr / hop))
        e = int(round(end * sr / hop))
        s = max(0, min(s, n_frames))
        e = max(0, min(e, n_frames))
        if e > s:
            midi_roll[s:e] = pitch

    voiced = np.isfinite(eval_f0_midi)
    midi_active = np.isfinite(midi_roll)
    both = voiced & midi_active
    coverage = float(np.sum(both) / np.sum(voiced)) if np.sum(voiced) > 0 else 0.0
    hallucination = (
        float(np.sum(midi_active & ~voiced) / np.sum(midi_active))
        if np.sum(midi_active) > 0
        else 0.0
    )
    pitch_err = np.abs(eval_f0_midi[both] - midi_roll[both])
    within_half = float(np.mean(pitch_err < 0.5)) * 100 if len(pitch_err) else 0.0

    durations = [end - start for start, end, _ in notes]
    median_dur_ms = float(np.median(durations)) * 1000 if durations else 0.0

    onsets = sorted(start for start, _, _ in notes)
    if len(onsets) >= 2:
        iois = np.diff(onsets)
        ioi_cv = float(np.std(iois) / np.mean(iois)) if np.mean(iois) > 0 else None
    else:
        ioi_cv = None

    return {
        "note_count": len(notes),
        "coverage_pct": round(coverage * 100, 2),
        "hallucination_pct": round(hallucination * 100, 2),
        "pitch_within_half_st_pct": round(within_half, 2),
        "median_note_duration_ms": round(median_dur_ms, 1),
        "ioi_cv": round(ioi_cv, 3) if ioi_cv is not None else None,
    }


def load_eval_cache(cache_dir: Path) -> tuple[np.ndarray, int]:
    eval_f0_midi = np.load(cache_dir / "eval_f0_midi.npy")
    sr = int(np.load(cache_dir / "eval_sr.npy")[0])
    return eval_f0_midi, sr
