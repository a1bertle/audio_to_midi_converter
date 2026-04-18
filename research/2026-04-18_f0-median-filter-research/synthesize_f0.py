#!/usr/bin/env python3
"""
synthesize_f0.py — Render raw and filtered F0 contours as sine-wave audio.

Usage:
    python synthesize_f0.py <vocals_stem.wav> [--rmvpe-checkpoint <path>]
                            [--window 7] [--output-dir .]

Produces:
    f0_raw.wav        — sine wave following unfiltered RMVPE F0
    f0_filtered.wav   — sine wave after median filter (default 7 frames / 70 ms)
    f0_raw_mix.wav    — raw F0 sine mixed with original vocals stem (−12 dB each)
    f0_filtered_mix.wav — filtered F0 sine mixed with original vocals stem

All outputs at 16 kHz mono WAV.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter

LOG_DIR = Path(__file__).resolve().parents[2] / ".logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOG_DIR / "error.log"),
    ],
)
logger = logging.getLogger("synthesize_f0")

_SR = 16000
_HOP = 160  # samples — 10 ms, matches RMVPE
_DEFAULT_CHECKPOINT = Path.home() / ".cache" / "audio2midi" / "rmvpe.pt"


def apply_median_filter(f0: np.ndarray, window: int) -> np.ndarray:
    """Apply median filter per voiced segment."""
    if window <= 1:
        return f0.copy()
    result = f0.copy().astype(float)
    voiced = f0 > 0
    changes = np.diff(voiced.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for s, e in zip(starts, ends):
        if e - s >= window:
            result[s:e] = median_filter(f0[s:e].astype(float), size=window)
    return result


def f0_to_sine(f0_frames: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Render an F0 frame sequence as a band-limited sine wave.

    Each frame's frequency is held constant for `hop` samples. Unvoiced
    frames (f0 <= 0) produce silence. Phase is accumulated continuously
    across frames so there are no discontinuities at voiced/unvoiced boundaries.
    """
    n_samples = len(f0_frames) * hop
    audio = np.zeros(n_samples, dtype=np.float32)
    phase = 0.0
    for i, freq in enumerate(f0_frames):
        start = i * hop
        end = start + hop
        if freq > 0:
            t = np.arange(hop) / sr
            audio[start:end] = np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
            phase += 2 * np.pi * freq * hop / sr
            phase %= 2 * np.pi
        else:
            phase = 0.0  # reset phase on silence so next onset starts clean
    return audio


def mix(a: np.ndarray, b: np.ndarray, gain_db: float = -12.0) -> np.ndarray:
    """Mix two signals at equal gain, trimming/padding to same length."""
    n = min(len(a), len(b))
    g = 10 ** (gain_db / 20.0)
    return (a[:n] * g + b[:n] * g).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize F0 contour as audio")
    parser.add_argument("vocals_wav", help="Path to separated vocals WAV stem")
    parser.add_argument("--rmvpe-checkpoint", default=str(_DEFAULT_CHECKPOINT))
    parser.add_argument("--window", type=int, default=7,
                        help="Median filter window in frames (default 7 = 70 ms)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to write output WAV files")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    vocals_path = Path(args.vocals_wav)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.rmvpe_checkpoint)

    if not vocals_path.exists():
        logger.error("File not found: %s", vocals_path)
        sys.exit(1)

    logger.info("Loading RMVPE from %s", checkpoint)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from audio2midi.transcribers._rmvpe_impl import RMVPE0Predictor  # type: ignore
    model = RMVPE0Predictor(str(checkpoint), device=args.device)

    logger.info("Loading vocals stem...")
    y, _ = librosa.load(str(vocals_path), sr=_SR, mono=True)

    logger.info("Running RMVPE pitch inference...")
    f0_raw = model.infer_from_audio(y, thred=0.03)

    voiced = int(np.sum(f0_raw > 0))
    logger.info("F0 frames: %d, voiced: %d (%.1f%%)",
                len(f0_raw), voiced, 100.0 * voiced / len(f0_raw))

    logger.info("Applying median filter (window=%d frames / %d ms)...",
                args.window, args.window * 10)
    f0_filtered = apply_median_filter(f0_raw, args.window)

    logger.info("Synthesizing sine waves...")
    sine_raw = f0_to_sine(f0_raw, _SR, _HOP)
    sine_filtered = f0_to_sine(f0_filtered, _SR, _HOP)

    # Trim vocals stem to same length as F0 audio
    n = min(len(y), len(sine_raw))
    vocals_trimmed = y[:n].astype(np.float32)

    paths = {
        "f0_raw.wav": sine_raw,
        "f0_filtered.wav": sine_filtered,
        "f0_raw_mix.wav": mix(sine_raw[:n], vocals_trimmed),
        "f0_filtered_mix.wav": mix(sine_filtered[:n], vocals_trimmed),
    }

    for name, audio in paths.items():
        out_path = out_dir / name
        sf.write(str(out_path), audio, _SR)
        logger.info("Written: %s (%.1f s)", out_path, len(audio) / _SR)

    print(f"\nOutputs in {out_dir}:")
    for name in paths:
        print(f"  {name}")
    print(f"\nFilter: {args.window} frames = {args.window * 10} ms")
    print(f"Voiced frames: raw={voiced}, "
          f"filtered={int(np.sum(f0_filtered > 0))}")


if __name__ == "__main__":
    main()
