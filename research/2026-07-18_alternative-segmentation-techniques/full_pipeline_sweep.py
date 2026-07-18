#!/usr/bin/env python3
"""Full candidate pipeline: RMVPE (swept thred) -> pYIN gap-bridge+veto
(reused exactly from the accepted pipeline) -> 3-state Viterbi segmentation.

Uses cached RMVPE hidden representation and cached pYIN track so only the
decode() + bridge/veto + segmentation steps re-run per candidate -- fast
enough to grid search.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio2midi.transcribers.rmvpe import _HOP_S, _short_internal_unvoiced_gaps  # noqa: E402
from audio2midi.transcribers._rmvpe_impl import RMVPE0Predictor  # noqa: E402
from fast_score import score_notes, load_eval_cache  # noqa: E402
from segment_3state import segment_3state  # noqa: E402

_BPM = 135.00


class _Decoder:
    """Bare object exposing RMVPE0Predictor.decode() without reloading the model."""

    def __init__(self) -> None:
        n_class = 360
        cents_mapping = 20 * np.arange(n_class) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    to_local_average_cents = RMVPE0Predictor.to_local_average_cents
    decode = RMVPE0Predictor.decode


def build_f0(
    hidden: np.ndarray,
    thred: float,
    pyin_n: np.ndarray,
    voiced_flag: np.ndarray,
    global_consistency_gate_st: float | None = None,
) -> np.ndarray:
    """Reproduce the accepted pipeline's bridge+veto logic, parameterized by
    RMVPE thred instead of the fixed 0.03.

    global_consistency_gate_st: if set, zero out ANY RMVPE-voiced frame
    (not just newly-bridged ones) where pYIN also votes voiced but disagrees
    on pitch by more than this many semitones. This targets low-thred's
    measured failure mode: recovered low-confidence frames that are
    pYIN-voiced but pitch-wrong (octave errors), which the original
    bridge-only veto does not catch since it only vetoes pYIN-unvoiced
    frames.
    """
    decoder = _Decoder()
    f0_rmvpe = decoder.decode(hidden, thred=thred)

    n = min(len(f0_rmvpe), len(pyin_n))
    f0_work = f0_rmvpe[:n].copy()
    pyin_n = pyin_n[:n]
    voiced_flag_n = voiced_flag[:n]

    if global_consistency_gate_st is not None:
        both_voiced = (f0_work > 0) & (pyin_n > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rmvpe_midi = 12 * np.log2(np.maximum(f0_work, 1e-9) / 440.0) + 69
            pyin_midi_all = 12 * np.log2(np.maximum(pyin_n, 1e-9) / 440.0) + 69
        disagree = both_voiced & (np.abs(rmvpe_midi - pyin_midi_all) > global_consistency_gate_st)
        f0_work[disagree] = 0.0

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

    rmvpe_voiced_updated = f0_work > 0
    short_gap_zone = _short_internal_unvoiced_gaps(voiced_flag_n, _BRIDGE_FRAMES)
    veto_mask = rmvpe_voiced_updated & (~voiced_flag_n) & (~short_gap_zone)
    f0_work[veto_mask] = 0.0
    return f0_work


def main() -> None:
    cache_dir = Path(__file__).resolve().parent / "cache"
    hidden = np.load(cache_dir / "rmvpe_hidden.npy")
    f0_pyin = np.load(cache_dir / "f0_pyin.npy")
    voiced_flag = np.load(cache_dir / "pyin_voiced_flag.npy")
    eval_f0_midi, eval_sr = load_eval_cache(cache_dir)

    thred_values = [0.03, 0.01, 0.005, 0.002, 0.001]
    tolerances = [0.5, 0.75, 1.0, 1.5]
    gates = [None, 1.0, 0.5]

    print(f"{'thred':>7} {'gate':>6} {'tol_st':>7} {'notes':>6} {'cov%':>7} {'halluc%':>8} "
          f"{'pitch.5%':>9} {'med_ms':>7} {'ioi_cv':>7}")
    results = []
    for thred in thred_values:
        for gate in gates:
            f0 = build_f0(hidden, thred, f0_pyin, voiced_flag, global_consistency_gate_st=gate)
            for tol in tolerances:
                notes = segment_3state(
                    f0, hop_s=_HOP_S, bpm=_BPM,
                    stay_prob=0.999, pitch_tolerance_semitones=tol,
                )
                metrics = score_notes(notes, eval_f0_midi, eval_sr)
                results.append({"thred": thred, "gate": gate, "tolerance": tol, **metrics})
                ioi_disp = metrics['ioi_cv'] if metrics['ioi_cv'] is not None else float('nan')
                gate_disp = f"{gate:.2f}" if gate is not None else "none"
                print(f"{thred:>7.4f} {gate_disp:>6} {tol:>7.2f} {metrics['note_count']:>6} "
                      f"{metrics['coverage_pct']:>7.2f} {metrics['hallucination_pct']:>8.2f} "
                      f"{metrics['pitch_within_half_st_pct']:>9.2f} "
                      f"{metrics['median_note_duration_ms']:>7.1f} {ioi_disp:>7.2f}")

    import json
    with open(Path(__file__).resolve().parent / "full_pipeline_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved full_pipeline_sweep_results.json")


if __name__ == "__main__":
    main()
