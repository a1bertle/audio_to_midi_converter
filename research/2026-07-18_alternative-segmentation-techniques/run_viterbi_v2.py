#!/usr/bin/env python3
"""Prototype v2: sustain-aware Viterbi note segmentation over a
coverage-extended F0 array.

Follow-up to Finding 2 in notes.md (2026-07-18 session). Two problems were
found and fixed relative to the first v2 draft:

1. The raw 2-state (voiced/unvoiced) Viterbi prototype fragmented worse than
   the accepted heuristic stack (median duration 139.4ms vs 169ms; IOI CV
   2.19 vs 2.04) because it re-cut a note on every semitone change. A first
   fix attempt (median-of-run reference pitch + wide tolerance band) traded
   this away for a *worse* problem: blurring a note's assigned pitch to the
   run median caused severe pitch-accuracy loss (as low as 52% within
   0.5 semitone, vs 79.5% accepted) as soon as the tolerance band was wide
   enough to matter, because portamento/glide frames far from the median
   were still scored against their own true pitch by eval_vocal_midi.py.
   Fixed here with a *persistence* debounce instead: a new rounded-semitone
   pitch only triggers a cut once it has held for >= persist_frames
   consecutive frames, so single-frame vibrato blips are absorbed but the
   assigned pitch per segment is never blurred away from the true quantized
   value.

2. Grid-search against the F0 array as originally computed (RMVPE + pYIN
   200ms-bridge + veto, i.e. f0_accepted) showed a hard *ceiling*: at most
   84.7% of pYIN-eval-voiced frames are ever voiced in that array, so no
   segmentation algorithm operating on it can reach the 90% coverage target
   -- consistent with the 2026-07-17 report's finding of a persistent
   pYIN-vs-RMVPE voicing gap. Measured fix: extend the F0 array with an
   unconditional pYIN fallback (use pYIN's own pitch wherever pYIN marks a
   frame voiced and RMVPE has nothing there, dropping the existing 200ms
   bridge-window and octave-consistency restrictions that limited the
   accepted pipeline's bridge step) raises the ceiling to 93.5%. This is a
   qualitatively different change from further constant-tuning (this
   session's premise): it makes pYIN a *second, independent* pitch source
   rather than only a confirmation signal within a narrow window of existing
   RMVPE voicing.

Eval-only script for this research session. Does not modify the accepted
pipeline (audio2midi/transcribers/rmvpe.py untouched).
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audio2midi.transcribers.rmvpe import (  # noqa: E402
    _HOP_S,
    _apply_f0_median_filter,
)

_BPM = 135.00


def extend_f0_with_pyin_fallback(
    f0_accepted: np.ndarray, f0_pyin: np.ndarray, pyin_voiced_flag: np.ndarray
) -> np.ndarray:
    """Fill remaining unvoiced frames in f0_accepted with pYIN's own pitch
    wherever pYIN independently marks the frame voiced. Unlike the accepted
    pipeline's bridge step, this is unconditional (no 200ms window, no
    octave-consistency check against nearby RMVPE pitch) -- pYIN is trusted
    as a second pitch source in its own right, not just to confirm RMVPE.
    """
    f0 = f0_accepted.copy()
    fill_mask = (f0 == 0) & pyin_voiced_flag & (f0_pyin > 0)
    f0[fill_mask] = f0_pyin[fill_mask]
    return f0


def viterbi_smooth_notes(
    f0_hz: np.ndarray,
    f0_filter_frames: int = 7,
    stay_prob: float = 0.985,
    p_voiced_active: float = 0.9,
    p_voiced_inactive: float = 0.1,
    persist_frames: int = 3,
    min_note_s: float | None = None,
) -> list[tuple[float, float, int]]:
    """Decode a smoothed voiced/unvoiced state path via Viterbi, then
    segment voiced runs into notes with a persistence-debounced pitch cut:
    a new rounded-semitone pitch must hold for >= persist_frames consecutive
    frames before it is allowed to end the current note. This suppresses
    single-frame vibrato/tracking flicker without blurring the assigned
    pitch of any committed note segment.
    """
    f0_hz = _apply_f0_median_filter(f0_hz, f0_filter_frames)
    times = np.arange(len(f0_hz)) * _HOP_S
    voiced_raw = f0_hz > 0

    p_voiced = np.where(voiced_raw, p_voiced_active, p_voiced_inactive)
    obs = np.vstack([1 - p_voiced, p_voiced])
    transition = np.array(
        [[stay_prob, 1 - stay_prob], [1 - stay_prob, stay_prob]]
    )

    import librosa.sequence

    state_path = librosa.sequence.viterbi_discriminative(
        obs, transition, p_init=np.array([0.5, 0.5])
    )

    lo_bound = librosa.note_to_hz("C2")
    hi_bound = librosa.note_to_hz("C6")
    in_range = (f0_hz >= lo_bound) & (f0_hz <= hi_bound)
    active_mask = state_path.astype(bool) & in_range & voiced_raw

    midi_track = np.where(
        voiced_raw,
        np.round(12 * np.log2(np.maximum(f0_hz, 1e-6) / 440.0) + 69).astype(int),
        -1,
    )

    if min_note_s is None:
        min_note_s = (60.0 / _BPM) / 8.0  # 32nd note floor, same as accepted pipeline

    notes: list[tuple[float, float, int]] = []
    in_note = False
    note_start = 0.0
    note_midi = 0
    # Candidate new pitch pending persistence confirmation.
    cand_midi: int | None = None
    cand_run = 0
    cand_start_t = 0.0

    def flush(end_t: float) -> None:
        nonlocal in_note
        if in_note and end_t - note_start >= min_note_s:
            notes.append((note_start, end_t, note_midi))
        in_note = False

    n = len(f0_hz)
    for i in range(n):
        t = float(times[i])
        active = bool(active_mask[i])
        if active:
            midi = int(midi_track[i])
            if not in_note:
                in_note = True
                note_start = t
                note_midi = midi
                cand_midi = None
                cand_run = 0
            elif midi == note_midi:
                cand_midi = None
                cand_run = 0
            else:
                if cand_midi == midi:
                    cand_run += 1
                else:
                    cand_midi = midi
                    cand_run = 1
                    cand_start_t = t
                if cand_run >= persist_frames:
                    flush(cand_start_t)
                    in_note = True
                    note_start = cand_start_t
                    note_midi = midi
                    cand_midi = None
                    cand_run = 0
        else:
            flush(t)
            cand_midi = None
            cand_run = 0

    if in_note:
        flush(n * _HOP_S)

    return notes


def write_midi(notes: list[tuple[float, float, int]], out_path: Path, bpm: float) -> None:
    import pretty_midi

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
        print("usage: run_viterbi_v2.py <f0_npy> <out_midi>", file=sys.stderr)
        raise SystemExit(2)
    f0 = np.load(sys.argv[1])
    notes = viterbi_smooth_notes(f0)
    write_midi(notes, Path(sys.argv[2]), _BPM)
    print(f"wrote {sys.argv[2]} ({len(notes)} notes)")


if __name__ == "__main__":
    main()
