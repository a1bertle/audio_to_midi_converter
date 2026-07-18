#!/usr/bin/env python3
"""Prototype: HMM/Viterbi note-state smoothing as a replacement for the
heuristic merge/split segmentation stack in rmvpe.py.

Reproduces the same F0 array the accepted pipeline computes (RMVPE +
pYIN gap-bridge + veto, exactly as in RmvpeTranscriber.transcribe), then
decodes note boundaries via Viterbi over a 3-state observation sequence
(silence / onset / sustain) instead of the current silence-merge ->
same-pitch-merge -> beat-gated-split -> gap-fill heuristic chain.

Eval-only script for this research session. Does not modify the accepted
pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audio2midi.transcribers.rmvpe import (  # noqa: E402
    _TARGET_SR,
    _HOP_S,
    _apply_f0_median_filter,
    _load_rmvpe,
    _short_internal_unvoiced_gaps,
)

_BPM = 135.00

_CHECKPOINT = Path.home() / ".cache" / "audio2midi" / "rmvpe.pt"


def compute_accepted_f0(vocals_path: Path) -> np.ndarray:
    """Reproduce the exact F0 array the accepted pipeline feeds into
    _pitch_track_to_notes (RMVPE + pYIN gap-bridge + veto)."""
    model = _load_rmvpe(_CHECKPOINT, "cpu")
    y, _ = librosa.load(str(vocals_path), sr=_TARGET_SR, mono=True)
    f0 = model.infer_from_audio(y, thred=0.03)

    f0_pyin, voiced_flag, _ = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C6")),
        sr=_TARGET_SR,
        hop_length=int(_HOP_S * _TARGET_SR),
        fill_na=0.0,
    )
    f0_pyin = np.where(voiced_flag, f0_pyin, 0.0)
    n = min(len(f0), len(f0_pyin))
    f0_work = f0[:n].copy()
    pyin_n = f0_pyin[:n]

    _BRIDGE_FRAMES = int(round(0.200 / _HOP_S))
    rmvpe_voiced = f0_work > 0
    from scipy.ndimage import binary_dilation

    bridge_zone = binary_dilation(rmvpe_voiced, iterations=_BRIDGE_FRAMES)
    candidate_mask = (~rmvpe_voiced) & bridge_zone & (pyin_n > 0)

    voiced_indices = np.where(rmvpe_voiced)[0]
    if len(voiced_indices) > 0 and np.any(candidate_mask):
        candidate_indices = np.where(candidate_mask)[0]
        ins = np.searchsorted(voiced_indices, candidate_indices)
        ins_left = np.clip(ins - 1, 0, len(voiced_indices) - 1)
        ins_right = np.clip(ins, 0, len(voiced_indices) - 1)
        dist_left = candidate_indices - voiced_indices[ins_left]
        dist_right = voiced_indices[ins_right] - candidate_indices
        nearest_idx = np.where(
            dist_left <= dist_right, voiced_indices[ins_left], voiced_indices[ins_right]
        )
        nearest_f0 = f0_work[nearest_idx]
        pyin_midi = 12 * np.log2(pyin_n[candidate_indices] / 440.0 + 1e-9) + 69
        near_midi = 12 * np.log2(nearest_f0 / 440.0 + 1e-9) + 69
        consistent = np.abs(pyin_midi - near_midi) <= 1.0
        patch_indices = candidate_indices[consistent]
    else:
        patch_indices = np.array([], dtype=int)

    f0_work[patch_indices] = pyin_n[patch_indices]

    pyin_voiced_flag = voiced_flag[:n]
    rmvpe_voiced_updated = f0_work > 0
    short_gap_zone = _short_internal_unvoiced_gaps(pyin_voiced_flag, _BRIDGE_FRAMES)
    veto_mask = rmvpe_voiced_updated & (~pyin_voiced_flag) & (~short_gap_zone)
    f0_work[veto_mask] = 0.0
    return f0_work


def viterbi_smooth_notes(f0_hz: np.ndarray, f0_filter_frames: int = 7) -> list[tuple[float, float, int]]:
    """Decode a smoothed voiced/unvoiced state path via Viterbi, then
    segment the voiced runs into notes on semitone changes.

    States: 0 = unvoiced, 1 = voiced. Self-transition strongly favored
    (penalizes rapid flips) — this directly targets the IOI-CV /
    fragmentation problem by refusing to cut a note unless the pitch
    evidence for "unvoiced" or a different pitch is sustained.
    """
    f0_hz = _apply_f0_median_filter(f0_hz, f0_filter_frames)
    times = np.arange(len(f0_hz)) * _HOP_S
    voiced_raw = f0_hz > 0

    # Observation likelihoods: probability of "voiced" state given the raw
    # voicing decision, softened so isolated flips don't dominate.
    p_voiced = np.where(voiced_raw, 0.9, 0.1)
    obs = np.vstack([1 - p_voiced, p_voiced])  # shape (2, T)

    # Transition matrix strongly favors staying in the same state -- this
    # is the "smoothing strength" knob replacing the heuristic merge/split
    # constants entirely with one interpretable parameter.
    stay_prob = 0.985
    transition = np.array(
        [[stay_prob, 1 - stay_prob], [1 - stay_prob, stay_prob]]
    )

    import librosa.sequence

    state_path = librosa.sequence.viterbi_discriminative(
        obs, transition, p_init=np.array([0.5, 0.5])
    )

    # Snap to semitones within voiced runs.
    midi_track = np.where(
        voiced_raw,
        np.round(12 * np.log2(np.maximum(f0_hz, 1e-6) / 440.0) + 69).astype(int),
        -1,
    )

    lo_bound = librosa.note_to_hz("C2")
    hi_bound = librosa.note_to_hz("C6")
    in_range = (f0_hz >= lo_bound) & (f0_hz <= hi_bound)

    notes: list[tuple[float, float, int]] = []
    in_note = False
    note_start = 0.0
    note_midi = 0
    min_note_s = (60.0 / _BPM) / 8.0  # 32nd note floor, same as accepted pipeline

    for i, (t, state) in enumerate(zip(times, state_path)):
        active = bool(state) and in_range[i] and midi_track[i] >= 0
        midi = int(midi_track[i]) if active else -1
        if active:
            if not in_note:
                in_note = True
                note_start = float(t)
                note_midi = midi
            elif midi != note_midi:
                if float(t) - note_start >= min_note_s:
                    notes.append((note_start, float(t), note_midi))
                note_start = float(t)
                note_midi = midi
        else:
            if in_note:
                if float(t) - note_start >= min_note_s:
                    notes.append((note_start, float(t), note_midi))
                in_note = False

    if in_note:
        track_end = len(f0_hz) * _HOP_S
        if track_end - note_start >= min_note_s:
            notes.append((note_start, track_end, note_midi))

    return notes


def write_midi(notes: list[tuple[float, float, int]], out_path: Path, bpm: float) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0, name="vocals")
    for start, end, midi_note in notes:
        inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=midi_note, start=start, end=end)
        )
    pm.instruments.append(inst)
    pm.write(str(out_path))


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: run_viterbi_smoothing.py <vocals_wav> <out_midi>", file=sys.stderr)
        raise SystemExit(2)
    vocals_path = Path(sys.argv[1])
    out_midi = Path(sys.argv[2])

    print("Computing accepted-pipeline F0 array (RMVPE + pYIN bridge + veto)...")
    f0 = compute_accepted_f0(vocals_path)

    print("Decoding notes via Viterbi state smoothing...")
    notes = viterbi_smooth_notes(f0)

    write_midi(notes, out_midi, _BPM)
    print(f"wrote {out_midi} ({len(notes)} notes)")


if __name__ == "__main__":
    main()
