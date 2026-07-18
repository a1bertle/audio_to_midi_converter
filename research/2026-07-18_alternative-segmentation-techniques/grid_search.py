#!/usr/bin/env python3
"""Grid search stay_prob x pitch_tolerance for the 3-state Viterbi segmenter,
scored against cached eval-grid pYIN ground truth (fast_score.py) so no
pYIN/RMVPE recomputation happens per candidate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fast_score import score_notes, load_eval_cache  # noqa: E402
from segment_3state import segment_3state  # noqa: E402

_HOP_S = 160.0 / 16000.0  # pipeline hop: 10 ms
_BPM = 135.00


def main() -> None:
    cache_dir = Path(__file__).resolve().parent / "cache"
    f0_accepted = np.load(cache_dir / "f0_accepted.npy")
    eval_f0_midi, eval_sr = load_eval_cache(cache_dir)

    stay_probs = [0.985, 0.999, 0.9995, 0.9999, 0.99995]
    tolerances = [0.5, 0.75, 1.0, 1.5, 2.0]

    print(f"{'stay_prob':>10} {'tol_st':>7} {'notes':>6} {'cov%':>7} {'halluc%':>8} "
          f"{'pitch.5%':>9} {'med_ms':>7} {'ioi_cv':>7}")
    results = []
    for stay_prob in stay_probs:
        for tol in tolerances:
            notes = segment_3state(
                f0_accepted, hop_s=_HOP_S, bpm=_BPM,
                stay_prob=stay_prob, pitch_tolerance_semitones=tol,
            )
            metrics = score_notes(notes, eval_f0_midi, eval_sr)
            results.append({"stay_prob": stay_prob, "tolerance": tol, **metrics})
            print(f"{stay_prob:>10.3f} {tol:>7.2f} {metrics['note_count']:>6} "
                  f"{metrics['coverage_pct']:>7.2f} {metrics['hallucination_pct']:>8.2f} "
                  f"{metrics['pitch_within_half_st_pct']:>9.2f} "
                  f"{metrics['median_note_duration_ms']:>7.1f} "
                  f"{metrics['ioi_cv'] if metrics['ioi_cv'] is not None else float('nan'):>7.2f}")

    import json
    with open(Path(__file__).resolve().parent / "grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved grid_search_results.json")


if __name__ == "__main__":
    main()
