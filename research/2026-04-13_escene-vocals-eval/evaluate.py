"""
Vocal stem separation quality evaluator.

Evaluates the quality of a vocals stem produced by a source-separation tool
(e.g. Demucs htdemucs 4-stem) against the original mix. Accepts .mp3 or .wav
for all inputs.

Metrics
-------
Reconstruction (mix-level):
  mix_energy_db        — total energy of original mix (dBFS scale)
  stem_sum_energy_db   — total energy of all stems summed
  residual_energy_db   — energy of (mix − sum(stems))
  leakage_ratio_db     — residual_energy_db − mix_energy_db; closer to −∞ = better reconstruction

Per-stem (vocals focus, others for cross-leakage):
  rms_db               — RMS loudness in dBFS
  srr_db               — stem-to-residual ratio; higher = cleaner isolation
  spectral_centroid_hz — mean spectral centroid; tonal character indicator
  silence_fraction     — fraction of frames below −60 dBFS (over-suppression proxy)
  cross_leakage_db     — per-pair spectral cosine similarity → dB; closer to −∞ = less bleed

Vocals-specific:
  vocal_presence_ratio — vocals RMS energy / mix RMS energy (expected: 0.2–0.5 for lead vocals)
  bleed_fraction       — cross-leakage of vocals vs each non-vocal stem (absolute similarity)

Usage
-----
  python evaluate.py \\
      --mix   path/to/original_mix.mp3 \\
      --stems-dir path/to/stems_folder/ \\
      [--stems vocals drums bass other] \\
      [--json-out path/to/report.json] \\
      [--sample-rate 44100]

  stems-dir must contain <stem_name>.mp3 or <stem_name>.wav files.

Exit codes
----------
  0   Evaluation completed.
  1   Measurement error (details in stderr and .logs/error.log).
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = _REPO_ROOT / ".logs"
LOG_DIR.mkdir(exist_ok=True)

_fmt = logging.Formatter(
    fmt=(
        "%(asctime)s %(levelname)s %(name)s "
        "[PID %(process)d %(threadName)s] %(message)s"
    ),
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)

_fh = logging.handlers.RotatingFileHandler(
    LOG_DIR / "error.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_fh.setLevel(logging.ERROR)
_fh.setFormatter(_fmt)

_sh = logging.StreamHandler(sys.stderr)
_sh.setLevel(logging.ERROR)
_sh.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_fh, _sh])
log = logging.getLogger("evaluate_vocals")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STEMS = ["vocals", "drums", "bass", "other"]
SILENCE_THRESHOLD_DBFS = -60.0
FRAME_HOP = 512  # samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 at target_sr. Accepts .mp3 and .wav."""
    try:
        import librosa
    except ImportError:
        log.error("librosa is required: pip install librosa")
        sys.exit(1)
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return np.asarray(y, dtype=np.float32), int(sr)


def _resolve_stem(stems_dir: Path, name: str) -> Path | None:
    """Find <name>.mp3 or <name>.wav in stems_dir."""
    for ext in (".mp3", ".wav"):
        p = stems_dir / f"{name}{ext}"
        if p.exists():
            return p
    return None


def _energy(x: np.ndarray) -> float:
    return float(np.sum(x.astype(np.float64) ** 2))


def _rms_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return float(20.0 * np.log10(rms)) if rms > 0 else -120.0


def _energy_db(energy: float) -> float:
    return float(10.0 * np.log10(energy)) if energy > 0 else -120.0


def _align(*arrays: np.ndarray) -> list[np.ndarray]:
    min_len = min(len(a) for a in arrays)
    return [a[:min_len] for a in arrays]


def _spectral_centroid_hz(y: np.ndarray, sr: int) -> float:
    import librosa
    cents = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=FRAME_HOP)
    return float(np.mean(cents))


def _silence_fraction(y: np.ndarray) -> float:
    frames = np.array_split(y, max(len(y) // FRAME_HOP, 1))
    db_vals = []
    for f in frames:
        rms = float(np.sqrt(np.mean(f.astype(np.float64) ** 2)))
        db_vals.append(20.0 * np.log10(rms) if rms > 0 else -120.0)
    silent = sum(1 for d in db_vals if d < SILENCE_THRESHOLD_DBFS)
    return float(silent / max(len(db_vals), 1))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    fa = np.abs(np.fft.rfft(a[:n].astype(np.float64)))
    fb = np.abs(np.fft.rfft(b[:n].astype(np.float64)))
    denom = np.linalg.norm(fa) * np.linalg.norm(fb)
    return float(np.dot(fa, fb) / denom) if denom > 0 else 0.0


def _cross_leakage_db(sim: float) -> float:
    return float(20.0 * np.log10(max(sim, 1e-10)))


def _peak_db(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x)))
    return float(20.0 * np.log10(peak)) if peak > 0 else -120.0


def _clipping_fraction(x: np.ndarray, threshold: float = 0.99) -> float:
    return float(np.mean(np.abs(x) >= threshold))


def _lufs_estimate(y: np.ndarray, sr: int) -> float:
    """Very rough integrated loudness estimate (K-weighted approximation omitted;
    uses RMS over full signal as a proxy — labelled [assumed] in report)."""
    return _rms_db(y) - 0.691  # EBU R 128 offset approximation [assumed]


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    mix_path: Path,
    stems_dir: Path,
    stem_names: list[str],
    sample_rate: int,
) -> dict:
    # ── Load mix ──────────────────────────────────────────────────────────────
    print(f"\nLoading mix:  {mix_path}", flush=True)
    mix, sr = _load(mix_path, sample_rate)

    mix_duration_s = len(mix) / sr
    mix_rms = _rms_db(mix)
    mix_peak = _peak_db(mix)
    mix_clip = _clipping_fraction(mix)
    mix_lufs = _lufs_estimate(mix, sr)
    mix_centroid = _spectral_centroid_hz(mix, sr)

    print(f"  Duration:      {mix_duration_s:.2f} s", flush=True)
    print(f"  Sample rate:   {sr} Hz", flush=True)
    print(f"  RMS:           {mix_rms:.2f} dBFS", flush=True)
    print(f"  Peak:          {mix_peak:.2f} dBFS", flush=True)
    print(f"  Clipping frac: {mix_clip:.4%}", flush=True)
    print(f"  LUFS est.:     {mix_lufs:.2f} LUFS (approx)", flush=True)
    print(f"  Centroid:      {mix_centroid:.1f} Hz", flush=True)

    # ── Load stems ────────────────────────────────────────────────────────────
    stems: dict[str, np.ndarray] = {}
    for name in stem_names:
        p = _resolve_stem(stems_dir, name)
        if p is None:
            print(f"  [skip] {name} not found in {stems_dir}", flush=True)
            log.warning("Stem not found: %s in %s", name, stems_dir)
            continue
        print(f"Loading stem: {p}", flush=True)
        y, _ = _load(p, sample_rate)
        stems[name] = y

    if not stems:
        log.error("No stems found in %s for %s", stems_dir, stem_names)
        sys.exit(1)

    # ── Align lengths ─────────────────────────────────────────────────────────
    all_arrays = [mix] + list(stems.values())
    aligned = _align(*all_arrays)
    mix_a = aligned[0]
    stem_arrays: dict[str, np.ndarray] = {
        name: aligned[i + 1] for i, name in enumerate(stems.keys())
    }

    # ── Reconstruction ────────────────────────────────────────────────────────
    stem_sum = sum(stem_arrays.values())
    residual = mix_a - stem_sum

    e_mix = _energy(mix_a)
    e_sum = _energy(stem_sum)
    e_res = _energy(residual)
    leakage_ratio_db = _energy_db(e_res) - _energy_db(e_mix)

    print("\n--- Reconstruction ---", flush=True)
    print(f"  Mix energy:       {_energy_db(e_mix):.2f} dB", flush=True)
    print(f"  Stem-sum energy:  {_energy_db(e_sum):.2f} dB", flush=True)
    print(f"  Residual energy:  {_energy_db(e_res):.2f} dB", flush=True)
    print(
        f"  Leakage ratio:    {leakage_ratio_db:.2f} dB  "
        "(residual/mix; closer to −∞ is better)",
        flush=True,
    )

    # ── Per-stem metrics ──────────────────────────────────────────────────────
    stem_metrics: dict[str, dict] = {}
    print("\n--- Per-stem metrics ---", flush=True)

    for name, arr in stem_arrays.items():
        e_stem = _energy(arr)
        srr = _energy_db(e_stem) - _energy_db(e_res)
        rms = _rms_db(arr)
        peak = _peak_db(arr)
        centroid = _spectral_centroid_hz(arr, sr)
        silence = _silence_fraction(arr)
        clip = _clipping_fraction(arr)

        cross: dict[str, float] = {}
        for other_name, other_arr in stem_arrays.items():
            if other_name == name:
                continue
            sim = _cosine_similarity(arr, other_arr)
            cross[other_name] = round(_cross_leakage_db(sim), 2)

        # Vocal-specific: presence ratio relative to mix
        vocal_presence_ratio = None
        if name == "vocals":
            mix_e_linear = float(np.sqrt(np.mean(mix_a.astype(np.float64) ** 2)))
            voc_e_linear = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
            vocal_presence_ratio = (
                round(voc_e_linear / mix_e_linear, 4) if mix_e_linear > 0 else None
            )

        stem_metrics[name] = {
            "rms_db": round(rms, 2),
            "peak_db": round(peak, 2),
            "srr_db": round(srr, 2),
            "spectral_centroid_hz": round(centroid, 1),
            "silence_fraction": round(silence, 4),
            "clipping_fraction": round(clip, 6),
            "cross_leakage_db": cross,
        }
        if vocal_presence_ratio is not None:
            stem_metrics[name]["vocal_presence_ratio"] = vocal_presence_ratio

        print(f"\n  [{name}]", flush=True)
        print(f"    RMS:                    {rms:.2f} dBFS", flush=True)
        print(f"    Peak:                   {peak:.2f} dBFS", flush=True)
        print(f"    SRR:                    {srr:.2f} dB  (higher = cleaner)", flush=True)
        print(f"    Spectral centroid:      {centroid:.1f} Hz", flush=True)
        print(f"    Silence fraction:       {silence:.2%}", flush=True)
        print(f"    Clipping fraction:      {clip:.4%}", flush=True)
        if vocal_presence_ratio is not None:
            print(f"    Vocal presence ratio:   {vocal_presence_ratio:.4f}  (RMS voc/mix)", flush=True)
        if cross:
            print("    Cross-leakage (dB):", flush=True)
            for other, val in cross.items():
                print(
                    f"      vs {other}: {val:.2f} dB  (closer to −∞ = less bleed)",
                    flush=True,
                )

    result = {
        "mix": {
            "path": str(mix_path),
            "duration_s": round(mix_duration_s, 2),
            "sample_rate": sr,
            "rms_db": round(mix_rms, 2),
            "peak_db": round(mix_peak, 2),
            "clipping_fraction": round(mix_clip, 6),
            "lufs_estimate": round(mix_lufs, 2),
            "spectral_centroid_hz": round(mix_centroid, 1),
        },
        "reconstruction": {
            "mix_energy_db": round(_energy_db(e_mix), 2),
            "stem_sum_energy_db": round(_energy_db(e_sum), 2),
            "residual_energy_db": round(_energy_db(e_res), 2),
            "leakage_ratio_db": round(leakage_ratio_db, 2),
        },
        "stems": stem_metrics,
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vocal stem separation quality evaluator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mix", required=True, type=Path)
    parser.add_argument("--stems-dir", required=True, type=Path)
    parser.add_argument("--stems", nargs="+", default=DEFAULT_STEMS, metavar="STEM")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()

    if not args.mix.exists():
        log.error("Mix file not found: %s", args.mix)
        sys.exit(1)
    if not args.stems_dir.is_dir():
        log.error("Stems directory not found: %s", args.stems_dir)
        sys.exit(1)

    try:
        results = evaluate(
            mix_path=args.mix,
            stems_dir=args.stems_dir,
            stem_names=args.stems,
            sample_rate=args.sample_rate,
        )
    except Exception:
        log.exception("Evaluation failed")
        sys.exit(1)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nJSON report written to: {args.json_out}", flush=True)

    print("\n--- JSON summary ---")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
