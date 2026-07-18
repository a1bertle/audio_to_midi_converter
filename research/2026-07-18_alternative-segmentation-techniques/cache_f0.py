#!/usr/bin/env python3
"""Compute and cache the accepted-pipeline F0 array + raw pYIN track once,
so segmentation-algorithm iteration doesn't re-pay RMVPE + pYIN cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audio2midi.transcribers.rmvpe import (  # noqa: E402
    _TARGET_SR,
    _HOP_S,
    _load_rmvpe,
    _short_internal_unvoiced_gaps,
)

_CHECKPOINT = Path.home() / ".cache" / "audio2midi" / "rmvpe.pt"


def main() -> None:
    vocals_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_rmvpe(_CHECKPOINT, "cpu")
    y, _ = librosa.load(str(vocals_path), sr=_TARGET_SR, mono=True)
    f0_rmvpe = model.infer_from_audio(y, thred=0.03)

    f0_pyin, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C6")),
        sr=_TARGET_SR,
        hop_length=int(_HOP_S * _TARGET_SR),
        fill_na=0.0,
    )
    f0_pyin = np.where(voiced_flag, f0_pyin, 0.0)

    n = min(len(f0_rmvpe), len(f0_pyin))
    f0_work = f0_rmvpe[:n].copy()
    pyin_n = f0_pyin[:n]
    voiced_flag_n = voiced_flag[:n]
    voiced_prob_n = voiced_prob[:n]

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
    f0_accepted = f0_work.copy()
    f0_accepted[veto_mask] = 0.0

    np.save(out_dir / "f0_rmvpe_raw.npy", f0_rmvpe[:n])
    np.save(out_dir / "f0_accepted.npy", f0_accepted)
    np.save(out_dir / "f0_pyin.npy", pyin_n)
    np.save(out_dir / "pyin_voiced_flag.npy", voiced_flag_n)
    np.save(out_dir / "pyin_voiced_prob.npy", voiced_prob_n)
    print(f"cached {n} frames to {out_dir}")


if __name__ == "__main__":
    main()
