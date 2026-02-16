"""Audio extraction wrappers around ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from audio2midi.exceptions import ExtractionError, MissingDependencyError


def ensure_ffmpeg_installed() -> None:
    """Ensure ffmpeg binary is available."""
    if shutil.which("ffmpeg") is None:
        raise MissingDependencyError(
            "ffmpeg is not installed or not available on PATH."
        )


def build_ffmpeg_command(
    input_media_path: Path,
    output_wav_path: Path,
    sample_rate: int = 44100,
    channels: int = 1,
    sample_fmt: str = "flt",
    audio_codec: str = "pcm_f32le",
) -> list[str]:
    """Build deterministic ffmpeg command for WAV extraction."""
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_media_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        audio_codec,
        "-sample_fmt",
        sample_fmt,
        str(output_wav_path),
    ]


def extract_to_wav(
    input_media_path: Path,
    output_wav_path: Path,
    sample_rate: int = 44100,
    channels: int = 1,
    sample_fmt: str = "flt",
    audio_codec: str = "pcm_f32le",
) -> Path:
    """Extract and convert media to WAV format."""
    ensure_ffmpeg_installed()
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_ffmpeg_command(
        input_media_path=input_media_path,
        output_wav_path=output_wav_path,
        sample_rate=sample_rate,
        channels=channels,
        sample_fmt=sample_fmt,
        audio_codec=audio_codec,
    )
    proc = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ExtractionError(
            "ffmpeg extraction failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stderr: {proc.stderr.strip()}"
        )
    if not output_wav_path.exists():
        raise ExtractionError(
            "WAV output file missing after extraction: "
            f"{output_wav_path}"
        )
    return output_wav_path
