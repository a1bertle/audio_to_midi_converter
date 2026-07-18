#!/usr/bin/env python3
"""Tolerance-window note-matching evaluation (Finding 4, notes.md).

scripts/eval_vocal_midi.py's coverage/hallucination metrics are frame-level
overlap against pYIN, which the 2026-07-17 report and this session's
Finding 4 identified as a known-imperfect proxy for singing voice: standard
SVT research (COnPOff methodology) instead uses tolerance-window note
matching (onset error < 50ms, pitch error < 50 cents, offset error <
max(50ms, 20% of note duration) -- mir_eval.transcription's defaults,
which match COnPOff exactly).

This script builds a pYIN-derived PSEUDO-reference note list (NOT
hand-annotated ground truth -- a real limitation, stated explicitly) by
segmenting the eval-grid pYIN F0 contour into notes the same simple way any
of this session's prototypes do (contiguous voiced run, split on semitone
change, 30ms minimum), then scores candidate MIDI files against it with
mir_eval.transcription.evaluate(), which implements the COnPOff on/off/pitch
tolerance-window metrics used in the literature.

This is a metric-methodology comparison, not a pipeline change. It does not
modify scripts/eval_vocal_midi.py or replace the accepted frame-level
metric -- both are reported side by side per the plan's requirement not to
silently redefine the published targets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_score import load_eval_cache  # noqa: E402

SESSION = Path(__file__).resolve().parent


def build_pseudo_reference(eval_f0_midi: np.ndarray, sr: int, hop: int = 512,
                            min_note_s: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """Segment the eval-grid pYIN F0 contour into (intervals, pitches_hz)
    the same way this session's Viterbi prototypes segment RMVPE F0: a note
    continues while pitch (rounded to nearest semitone) is unchanged and the
    frame stays voiced.
    """
    hop_s = hop / sr
    midi_round = np.where(
        np.isfinite(eval_f0_midi), np.round(eval_f0_midi).astype(int), -999
    )
    intervals = []
    pitches_hz = []
    in_note = False
    start_i = 0
    cur = -999
    n = len(midi_round)
    for i in range(n + 1):
        m = midi_round[i] if i < n else -999
        if m != -999 and (not in_note or m != cur):
            if in_note:
                end_t = i * hop_s
                start_t = start_i * hop_s
                if end_t - start_t >= min_note_s:
                    intervals.append((start_t, end_t))
                    pitches_hz.append(librosa_midi_to_hz(cur))
            start_i = i
            cur = m
            in_note = True
        elif m == -999 and in_note:
            end_t = i * hop_s
            start_t = start_i * hop_s
            if end_t - start_t >= min_note_s:
                intervals.append((start_t, end_t))
                pitches_hz.append(librosa_midi_to_hz(cur))
            in_note = False
    return np.array(intervals), np.array(pitches_hz)


def librosa_midi_to_hz(midi_pitch: int) -> float:
    return 440.0 * 2 ** ((midi_pitch - 69) / 12.0)


def midi_to_intervals_pitches(midi_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [n for inst in pm.instruments for n in inst.notes]
    intervals = np.array([[n.start, n.end] for n in notes])
    pitches_hz = np.array([librosa_midi_to_hz(n.pitch) for n in notes])
    return intervals, pitches_hz


def main() -> None:
    eval_f0_midi, sr = load_eval_cache(SESSION / "cache")
    ref_intervals, ref_pitches = build_pseudo_reference(eval_f0_midi, sr)
    print(f"pseudo-reference: {len(ref_intervals)} notes (pYIN-derived, NOT hand-annotated)")

    candidates = {
        "accepted_2026-07-17": Path(
            "../../outputs/2026-07-17_Adounravel_segfix/Adounravel_歌いました.mid"
        ),
        "basic_pitch": SESSION / "adounravel_basic_pitch.mid",
        "viterbi_raw_2state": SESSION / "adounravel_viterbi.mid",
    }
    for extra in sorted(SESSION.glob("adounravel_v2_*.mid")):
        candidates[extra.stem] = extra

    import json

    all_scores = {}
    for name, path in candidates.items():
        if not path.exists():
            print(f"{name}: (missing: {path})")
            continue
        est_intervals, est_pitches = midi_to_intervals_pitches(path)
        if len(est_intervals) == 0:
            print(f"{name}: (no notes)")
            continue
        scores = mir_eval.transcription.evaluate(
            ref_intervals, ref_pitches, est_intervals, est_pitches
        )
        all_scores[name] = scores
        print(f"\n=== {name} ({len(est_intervals)} notes) ===")
        for k, v in scores.items():
            print(f"  {k}: {v:.4f}")
        out = SESSION / f"tolerance-window-{name}.json"
        out.write_text(json.dumps({k: float(v) for k, v in scores.items()}, indent=2) + "\n")

    print("\n=== Summary (Onset+Offset+Pitch F-measure = closest standard analog to COnPOff) ===")
    key = "Precision" if "Precision" in next(iter(all_scores.values()), {}) else None
    for name, scores in all_scores.items():
        print(f"{name:>28}  {json.dumps({k: round(v,3) for k,v in scores.items() if 'F-measure' in k})}")


if __name__ == "__main__":
    main()
