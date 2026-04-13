"""
Stem separation quality evaluator.

Measures blind (no reference) quality metrics for a set of separated stems
produced by a source-separation tool (e.g. Demucs).

Metrics
-------
For each stem WAV file:

  leakage_ratio_db
      10 * log10(energy of residual / energy of stem).
      Residual = mix - sum(all stems).  A large negative value (e.g. -30 dB)
      means the stems reconstruct the mix well.  A value close to 0 means
      reconstruction is poor or cancellation is incomplete.

  stem_to_residual_ratio_db  (SRR)
      10 * log10(energy of this stem / energy of residual).
      Higher is better — means this stem carries more signal than leftover
      artefacts.

  cross_leakage_db  (per other stem)
      Spectral cosine similarity converted to a dB-like leakage estimate
      between this stem and every other stem.  Values closer to -inf mean
      the stems are more orthogonal (less bleed-through).

  rms_db
      RMS level of the stem in dBFS.

  spectral_centroid_hz
      Mean spectral centroid of the stem (rough tonal character indicator).

  silence_fraction
      Fraction of frames where the stem is below -60 dBFS RMS (proxy for
      over-suppression).

Usage
-----
  python evaluate_stems.py \\
      --mix  path/to/mix.mp3 \\
      --stems-dir path/to/stems_folder/ \\
      [--stems vocals drums bass guitar piano other] \\
      [--json-out path/to/report.json] \\
      [--sample-rate 44100]

  The stems folder must contain <stem_name>.wav files.
  --mix is required to compute reconstruction metrics.

Exit codes
----------
  0   Evaluation completed without errors.
  1   A measurement error occurred (details in stderr and .logs/error.log).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import logging.handlers
import numpy as np

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).resolve().parents[2] / ".logs"
LOG_DIR.mkdir(exist_ok=True)

_log_formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s %(name)s [PID %(process)d %(threadName)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)

_file_handler = logging.handlers.RotatingFileHandler(
    LOG_DIR / "error.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_file_handler.setLevel(logging.ERROR)
_file_handler.setFormatter(_log_formatter)

_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.ERROR)
_stderr_handler.setFormatter(_log_formatter)

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _stderr_handler])
log = logging.getLogger("evaluate_stems")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]
SILENCE_THRESHOLD_DBFS = -60.0
FRAME_HOP = 512  # samples


def _load(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 at target_sr."""
    try:
        import librosa  # type: ignore
    except ImportError:
        log.error("librosa is required. Install it with: pip install librosa")
        sys.exit(1)
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return np.asarray(y, dtype=np.float32), int(sr)


def _energy(x: np.ndarray) -> float:
    return float(np.sum(x.astype(np.float64) ** 2))


def _rms_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if rms <= 0.0:
        return -120.0
    return float(20.0 * np.log10(rms))


def _energy_db(energy: float) -> float:
    if energy <= 0.0:
        return -120.0
    return float(10.0 * np.log10(energy))


def _align_length(*arrays: np.ndarray) -> list[np.ndarray]:
    """Trim all arrays to the length of the shortest."""
    min_len = min(len(a) for a in arrays)
    return [a[:min_len] for a in arrays]


def _spectral_centroid_hz(y: np.ndarray, sr: int) -> float:
    try:
        import librosa
    except ImportError:
        return float("nan")
    cents = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=FRAME_HOP)
    return float(np.mean(cents))


def _silence_fraction(y: np.ndarray, sr: int) -> float:
    """Fraction of FRAME_HOP-length frames below SILENCE_THRESHOLD_DBFS."""
    import librosa  # type: ignore
    frames = librosa.util.frame(y, frame_length=FRAME_HOP, hop_length=FRAME_HOP)
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=0))
    with np.errstate(divide="ignore"):
        frame_db = np.where(frame_rms > 0, 20.0 * np.log10(frame_rms), -120.0)
    silent = np.sum(frame_db < SILENCE_THRESHOLD_DBFS)
    return float(silent / max(len(frame_db), 1))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Spectral cosine similarity between two mono waveforms (via FFT magnitudes)."""
    n = min(len(a), len(b))
    fa = np.abs(np.fft.rfft(a[:n].astype(np.float64)))
    fb = np.abs(np.fft.rfft(b[:n].astype(np.float64)))
    denom = (np.linalg.norm(fa) * np.linalg.norm(fb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(fa, fb) / denom)


def _cross_leakage_db(similarity: float) -> float:
    """Convert cosine similarity to a dB leakage value.

    similarity=0   → -inf dB (perfectly orthogonal, no leakage)
    similarity=1   →   0 dB  (identical spectra, complete leakage)
    """
    clipped = max(similarity, 1e-10)
    return float(20.0 * np.log10(clipped))


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    mix_path: Path,
    stems_dir: Path,
    stem_names: list[str],
    sample_rate: int,
) -> dict:
    """Run all metrics and return a result dict."""
    import librosa  # noqa: F401 — already imported via _load; kept for _silence_fraction

    # Load mix
    print(f"Loading mix:  {mix_path}", flush=True)
    mix, sr = _load(mix_path, sample_rate)

    # Load stems
    stems: dict[str, np.ndarray] = {}
    for name in stem_names:
        wav_path = stems_dir / f"{name}.wav"
        if not wav_path.exists():
            log.warning("Stem file not found, skipping: %s", wav_path)
            print(f"  [skip] {name}.wav not found", flush=True)
            continue
        print(f"Loading stem: {wav_path}", flush=True)
        y, _ = _load(wav_path, sample_rate)
        stems[name] = y

    if not stems:
        log.error("No stem files found in %s for names %s", stems_dir, stem_names)
        sys.exit(1)

    # Align lengths: mix and all stems to shortest
    all_arrays = [mix] + list(stems.values())
    aligned = _align_length(*all_arrays)
    mix_a = aligned[0]
    stem_arrays = {name: aligned[i + 1] for i, name in enumerate(stems.keys())}

    # Reconstruction residual: mix - sum(stems)
    stem_sum = np.zeros_like(mix_a)
    for arr in stem_arrays.values():
        stem_sum += arr
    residual = mix_a - stem_sum

    energy_mix = _energy(mix_a)
    energy_residual = _energy(residual)
    energy_stem_sum = _energy(stem_sum)

    leakage_ratio_db = _energy_db(energy_residual) - _energy_db(energy_mix)

    print("\n--- Reconstruction ---", flush=True)
    print(f"  Mix energy:        {_energy_db(energy_mix):.2f} dB", flush=True)
    print(f"  Stem-sum energy:   {_energy_db(energy_stem_sum):.2f} dB", flush=True)
    print(f"  Residual energy:   {_energy_db(energy_residual):.2f} dB", flush=True)
    print(f"  Leakage ratio:     {leakage_ratio_db:.2f} dB  (residual / mix; closer to -inf is better)", flush=True)

    # Per-stem metrics
    stem_metrics: dict[str, dict] = {}
    names_list = list(stem_arrays.keys())
    print("\n--- Per-stem metrics ---", flush=True)

    for name, arr in stem_arrays.items():
        e_stem = _energy(arr)
        srr = _energy_db(e_stem) - _energy_db(energy_residual)
        rms = _rms_db(arr)
        centroid = _spectral_centroid_hz(arr, sr)
        silence = _silence_fraction(arr, sr)

        # Cross-leakage with all other stems
        cross: dict[str, float] = {}
        for other_name, other_arr in stem_arrays.items():
            if other_name == name:
                continue
            sim = _cosine_similarity(arr, other_arr)
            cross[other_name] = round(_cross_leakage_db(sim), 2)

        stem_metrics[name] = {
            "rms_db": round(rms, 2),
            "stem_to_residual_ratio_db": round(srr, 2),
            "spectral_centroid_hz": round(centroid, 1),
            "silence_fraction": round(silence, 4),
            "cross_leakage_db": cross,
        }

        print(f"\n  [{name}]", flush=True)
        print(f"    RMS:                       {rms:.2f} dBFS", flush=True)
        print(f"    Stem-to-residual ratio:    {srr:.2f} dB  (higher = cleaner)", flush=True)
        print(f"    Spectral centroid:         {centroid:.1f} Hz", flush=True)
        print(f"    Silence fraction:          {silence:.2%}", flush=True)
        if cross:
            print(f"    Cross-leakage (dB):", flush=True)
            for other, val in cross.items():
                print(f"      vs {other}: {val:.2f} dB  (closer to -inf = less bleed)", flush=True)

    return {
        "mix_path": str(mix_path),
        "stems_dir": str(stems_dir),
        "sample_rate": sr,
        "reconstruction": {
            "mix_energy_db": round(_energy_db(energy_mix), 2),
            "stem_sum_energy_db": round(_energy_db(energy_stem_sum), 2),
            "residual_energy_db": round(_energy_db(energy_residual), 2),
            "leakage_ratio_db": round(leakage_ratio_db, 2),
        },
        "stems": stem_metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blind stem separation quality evaluator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mix", required=True, type=Path,
        help="Path to the original mixed audio file.",
    )
    parser.add_argument(
        "--stems-dir", required=True, type=Path,
        help="Directory containing <stem_name>.wav files.",
    )
    parser.add_argument(
        "--stems", nargs="+", default=DEFAULT_STEMS,
        metavar="STEM",
        help=f"Stem names to evaluate (default: {' '.join(DEFAULT_STEMS)}).",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="Write JSON report to this path.",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=44100,
        help="Sample rate to load audio at (default: 44100).",
    )
    args = parser.parse_args()

    if not args.mix.exists():
        log.error("Mix file not found: %s", args.mix)
        print(f"ERROR: mix file not found: {args.mix}", file=sys.stderr)
        sys.exit(1)
    if not args.stems_dir.is_dir():
        log.error("Stems directory not found: %s", args.stems_dir)
        print(f"ERROR: stems directory not found: {args.stems_dir}", file=sys.stderr)
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
