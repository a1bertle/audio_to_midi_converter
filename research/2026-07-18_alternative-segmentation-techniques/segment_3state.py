#!/usr/bin/env python3
"""3-state (silence / onset / sustain) Viterbi note segmentation.

Improves on the 2-state prototype (run_viterbi_smoothing.py) by giving
"sustain" a pitch-change tolerance: small semitone wobble does not force a
state transition, only a genuine new onset (large pitch jump, or a
silence gap) does. This directly targets the 2-state prototype's measured
weakness: it re-cut notes on every semitone change same as the raw
heuristic, producing worse fragmentation than the accepted pipeline.

States:
  0 = silence  (unvoiced)
  1 = onset    (voiced, first frame(s) of a new note)
  2 = sustain  (voiced, continuing the current note; tolerant of small
                pitch deviation from the note's anchor pitch)

This is a *self-transition-biased* Viterbi decode over voicing, with a
separate deterministic note-boundary rule layered on top: a transition
into "onset" (from silence, or from sustain when pitch has drifted past
tolerance) starts a new note; consecutive sustain frames extend it.
"""
from __future__ import annotations

import numpy as np


def apply_median_filter(f0_hz: np.ndarray, window_frames: int) -> np.ndarray:
    if window_frames <= 1:
        return f0_hz
    from scipy.signal import medfilt

    voiced = f0_hz > 0
    out = f0_hz.copy()
    if np.any(voiced):
        # Median-filter only in log-pitch space over voiced frames, leaving
        # unvoiced frames at 0 -- matches _apply_f0_median_filter's intent
        # without importing the private pipeline function.
        log_f0 = np.where(voiced, np.log2(np.maximum(f0_hz, 1e-6)), 0.0)
        filtered = medfilt(log_f0, kernel_size=window_frames if window_frames % 2 else window_frames + 1)
        out = np.where(voiced, 2.0 ** filtered, 0.0)
    return out


def segment_3state(
    f0_hz: np.ndarray,
    hop_s: float,
    bpm: float,
    f0_filter_frames: int = 7,
    stay_prob: float = 0.985,
    pitch_tolerance_semitones: float = 0.75,
    min_note_s: float | None = None,
) -> list[tuple[float, float, int]]:
    """Segment an F0 contour into notes via voicing-Viterbi + pitch-tolerant
    sustain tracking.

    stay_prob: self-transition probability for the voiced/unvoiced Viterbi
        smoothing pass (same role as the 2-state prototype's knob).
    pitch_tolerance_semitones: how far the running pitch estimate may drift
        from a note's anchor pitch before a new onset is declared. This is
        the new parameter that should fix the fragmentation regression seen
        in the 2-state prototype.
    """
    f0_hz = apply_median_filter(f0_hz, f0_filter_frames)
    times = np.arange(len(f0_hz)) * hop_s
    voiced_raw = f0_hz > 0

    lo_bound_midi = 12 * np.log2(65.0 / 440.0) + 69  # ~C2
    hi_bound_midi = 12 * np.log2(1050.0 / 440.0) + 69  # ~C6
    with np.errstate(divide="ignore", invalid="ignore"):
        pitch_midi_cont = 12 * np.log2(np.maximum(f0_hz, 1e-6) / 440.0) + 69
    in_range = (pitch_midi_cont >= lo_bound_midi) & (pitch_midi_cont <= hi_bound_midi)
    voiced_raw = voiced_raw & in_range

    p_voiced = np.where(voiced_raw, 0.99, 0.01)
    obs = np.vstack([1 - p_voiced, p_voiced])
    transition = np.array([[stay_prob, 1 - stay_prob], [1 - stay_prob, stay_prob]])

    import librosa.sequence

    state_path = librosa.sequence.viterbi_discriminative(
        obs, transition, p_init=np.array([0.5, 0.5])
    )
    smoothed_voiced = state_path.astype(bool)

    if min_note_s is None:
        min_note_s = (60.0 / bpm) / 8.0  # 32nd note floor, matches accepted pipeline

    notes: list[tuple[float, float, int]] = []
    in_note = False
    note_start = 0.0
    anchor_midi = 0.0
    note_pitches: list[float] = []

    def close_note(end_t: float) -> None:
        nonlocal in_note
        if not in_note:
            return
        if end_t - note_start >= min_note_s and note_pitches:
            final_pitch = int(round(float(np.median(note_pitches))))
            notes.append((note_start, end_t, final_pitch))
        in_note = False

    for i, t in enumerate(times):
        active = bool(smoothed_voiced[i]) and voiced_raw[i]
        if active:
            cur_midi = pitch_midi_cont[i]
            if not in_note:
                in_note = True
                note_start = float(t)
                anchor_midi = cur_midi
                note_pitches = [cur_midi]
            else:
                if abs(cur_midi - anchor_midi) > pitch_tolerance_semitones:
                    close_note(float(t))
                    in_note = True
                    note_start = float(t)
                    anchor_midi = cur_midi
                    note_pitches = [cur_midi]
                else:
                    note_pitches.append(cur_midi)
                    # Slowly drift the anchor toward the running median so a
                    # gradual glide (not a jump) doesn't accumulate error and
                    # falsely trip the tolerance gate later in a long note.
                    anchor_midi = float(np.median(note_pitches[-15:]))
        else:
            close_note(float(t))

    if in_note:
        track_end = len(f0_hz) * hop_s
        close_note(track_end)

    return notes
