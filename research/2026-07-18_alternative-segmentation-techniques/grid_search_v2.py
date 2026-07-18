#!/usr/bin/env python3
"""Grid-search run_viterbi_v2 parameters against the cached eval-grid pYIN
ground truth, using fast_score.py's local metric replicas (no per-iteration
pYIN recomputation). Prints candidates ranked by how many of the four
unmet targets they clear, subject to the two passing guards.

Targets: coverage >= 90%, median_note_duration_ms >= 300, ioi_cv <= 1.5,
hallucination_pct <= 5%.
Guards (must not regress): pitch_within_half_st_pct >= 79%,
duration_mismatch not checked here (checked in the final real-eval pass).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fast_score import load_eval_cache, score_notes  # noqa: E402
from run_viterbi_v2 import extend_f0_with_pyin_fallback, viterbi_smooth_notes  # noqa: E402

SESSION = Path(__file__).resolve().parent


def targets_cleared(m: dict) -> int:
    n = 0
    if m["coverage_pct"] >= 90:
        n += 1
    if m["median_note_duration_ms"] >= 300:
        n += 1
    if m["ioi_cv"] is not None and m["ioi_cv"] <= 1.5:
        n += 1
    if m["hallucination_pct"] <= 5:
        n += 1
    return n


def guards_ok(m: dict) -> bool:
    return m["pitch_within_half_st_pct"] >= 79.0


def main() -> None:
    f0_accepted = np.load(SESSION / "cache" / "f0_accepted.npy")
    f0_pyin = np.load(SESSION / "cache" / "f0_pyin.npy")
    pyin_voiced_flag = np.load(SESSION / "cache" / "pyin_voiced_flag.npy")
    f0 = extend_f0_with_pyin_fallback(f0_accepted, f0_pyin, pyin_voiced_flag)
    eval_f0_midi, sr = load_eval_cache(SESSION / "cache")

    stay_probs = [0.97, 0.993]
    p_voiced_actives = [0.9]
    p_voiced_inactives = [0.1]
    persist_frames_opts = [1, 2, 3, 5, 8]
    f0_filter_frames_opts = [3, 5, 7, 11]

    results = []
    total = (
        len(stay_probs) * len(p_voiced_actives) * len(p_voiced_inactives)
        * len(persist_frames_opts) * len(f0_filter_frames_opts)
    )
    done = 0
    for stay_prob, pva, pvi, persist, filt in itertools.product(
        stay_probs, p_voiced_actives, p_voiced_inactives, persist_frames_opts, f0_filter_frames_opts
    ):
        notes = viterbi_smooth_notes(
            f0,
            f0_filter_frames=filt,
            stay_prob=stay_prob,
            p_voiced_active=pva,
            p_voiced_inactive=pvi,
            persist_frames=persist,
        )
        m = score_notes(notes, eval_f0_midi, sr)
        m["params"] = dict(
            stay_prob=stay_prob, p_voiced_active=pva, p_voiced_inactive=pvi,
            persist_frames=persist, f0_filter_frames=filt,
        )
        m["cleared"] = targets_cleared(m)
        m["guards_ok"] = guards_ok(m)
        results.append(m)
        done += 1
        if done % 10 == 0:
            print(f"...{done}/{total}", file=sys.stderr)

    results.sort(key=lambda m: (m["guards_ok"], m["cleared"]), reverse=True)

    print(f"{'cleared':>7} {'guard':>5} {'cov%':>6} {'med_ms':>7} {'ioi_cv':>7} {'hall%':>6} {'pitch%':>7} {'notes':>6}  params")
    for m in results[:25]:
        print(
            f"{m['cleared']:>7} {str(m['guards_ok']):>5} {m['coverage_pct']:>6.1f} "
            f"{m['median_note_duration_ms']:>7.1f} {m['ioi_cv'] if m['ioi_cv'] is not None else float('nan'):>7.2f} "
            f"{m['hallucination_pct']:>6.2f} {m['pitch_within_half_st_pct']:>7.2f} {m['note_count']:>6}  {m['params']}"
        )


if __name__ == "__main__":
    main()
