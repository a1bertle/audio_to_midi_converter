#!/usr/bin/env python3
"""
yt_to_stems.py — Download audio from a YouTube URL and run htdemucs_4s separation.

Usage:
    python3 scripts/yt_to_stems.py <youtube_url> [--device mps|cpu|cuda]

Output:
    output/<safe_title>/  — contains demucs stems (drums, bass, vocals, other)

Requires: yt-dlp, demucs (installed in .venv)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output"
MODEL = "htdemucs"  # htdemucs_4s (stock 4-stem)


def slugify(title: str) -> str:
    """Convert a title to a safe directory name."""
    slug = re.sub(r"[^\w\s\-]", "", title)
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug[:80]


def download_wav(url: str, dest_dir: Path) -> Path:
    """Download audio from YouTube as WAV into dest_dir. Returns the WAV path."""
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", str(dest_dir / "%(title)s.%(ext)s"),
        "--print", "after_move:filepath",
        url,
    ]
    print(f"[yt-dlp] downloading: {url}", flush=True)
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError("yt-dlp download failed")

    # Find the downloaded WAV
    wavs = list(dest_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV found in {dest_dir} after download")
    return wavs[0]


def get_title(url: str) -> str:
    """Fetch the video title without downloading audio."""
    result = subprocess.run(
        ["yt-dlp", "--print", "title", "--no-download", url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return "unknown"
    return result.stdout.strip()


def run_demucs(wav_path: Path, out_dir: Path, device: str) -> None:
    """Run htdemucs_4s separation on wav_path, outputting to out_dir."""
    cmd = [
        sys.executable, "-m", "demucs",
        "--name", MODEL,
        "--device", device,
        "--out", str(out_dir),
        str(wav_path),
    ]
    print(f"[demucs] model={MODEL} device={device}", flush=True)
    print(f"[demucs] input={wav_path.name}", flush=True)
    print(f"[demucs] output={out_dir}", flush=True)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("demucs separation failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download YouTube audio and run htdemucs_4s stem separation."
    )
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cpu", "cuda"],
        help="Compute device for demucs (default: mps)",
    )
    args = parser.parse_args()

    # Fetch title to name the output directory
    print("[info] fetching video title...", flush=True)
    title = get_title(args.url)
    slug = slugify(title)
    print(f"[info] title: {title}", flush=True)

    out_dir = OUTPUT_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] output directory: {out_dir}", flush=True)

    # Download into a temp directory, then separate directly from there
    with tempfile.TemporaryDirectory(prefix="yt_to_stems_") as tmp:
        tmp_path = Path(tmp)
        wav_path = download_wav(args.url, tmp_path)
        print(f"[info] downloaded: {wav_path.name} ({wav_path.stat().st_size / 1e6:.1f} MB)", flush=True)

        run_demucs(wav_path, out_dir, args.device)

    # demucs creates: out_dir/<model>/<track_name>/*.wav
    stems_dirs = list(out_dir.rglob("*.wav"))
    if stems_dirs:
        stems_dir = stems_dirs[0].parent
        print(f"\n[done] stems saved to: {stems_dir}")
        for stem in sorted(stems_dir.iterdir()):
            print(f"       {stem.name}")
    else:
        print(f"\n[done] output: {out_dir}")


if __name__ == "__main__":
    main()
