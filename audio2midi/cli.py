"""Command-line entrypoint for audio2midi."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

_USER_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "user.settings.json"


def _load_user_settings() -> dict:
    if _USER_SETTINGS_PATH.exists():
        try:
            return json.loads(_USER_SETTINGS_PATH.read_text())
        except Exception:
            pass
    return {}

import subprocess
import tempfile

import numpy as np

from audio2midi.downloader import YouTubeDownloader
from audio2midi.exceptions import Audio2MidiError
from audio2midi.extractor import extract_to_wav
from audio2midi.midi_writer import write_midi_file
from audio2midi.models import Instrument
from audio2midi.postprocess import postprocess_transcription
from audio2midi.preprocess import preprocess_wav_file
from audio2midi.transcribers.base import create_transcriber
from audio2midi.transcribers.rmvpe import _parse_beat_times


LOGGER = logging.getLogger("audio2midi")


def _run_bpm_detect(
    audio_path: Path,
    bpm_detect_bin: Path,
    click_track_path: Path,
) -> float | None:
    """Run bpm_detect on the raw mix and write click track to click_track_path.

    Transcodes WAV→MP3 first since bpm_detect only accepts MP3/MP4/M4A.
    Returns detected BPM or None on failure.
    """
    input_path = audio_path
    tmp_mp3 = None

    if audio_path.suffix.lower() == ".wav":
        tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_mp3.close()
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path), "-q:a", "2", tmp_mp3.name],
                capture_output=True, check=True, timeout=120,
            )
            input_path = Path(tmp_mp3.name)
        except Exception as exc:
            LOGGER.warning("ffmpeg WAV→MP3 conversion failed: %s; skipping bpm_detect", exc)
            return None

    try:
        result = subprocess.run(
            [str(bpm_detect_bin), "-v", str(input_path), "-o", str(click_track_path)],
            capture_output=True, text=True, timeout=120,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Detected BPM:"):
                bpm = float(line.split(":")[1].strip())
                LOGGER.info("Detected BPM: %.2f", bpm)
                return bpm
        LOGGER.warning("bpm_detect ran but produced no BPM line")
    except Exception as exc:
        LOGGER.warning("bpm_detect failed: %s", exc)
    finally:
        if tmp_mp3 is not None:
            Path(tmp_mp3.name).unlink(missing_ok=True)
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio2midi",
        description=(
            "Transcribe vocal performances from YouTube into MIDI using RMVPE pitch tracking. "
            "Also supports piano and guitar. Runs completely on-device after download."
        ),
    )
    parser.add_argument("--youtube-url", required=True, help="YouTube video URL.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output MIDI path. Defaults to video title with spaces replaced by underscores.",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help=(
            "Working directory for downloads and intermediates. "
            "Defaults to <project_root>/outputs/<video_name>/."
        ),
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        default=True,
        help="Keep extracted and preprocessed WAV files (default: True).",
    )
    parser.add_argument(
        "--allow-playlist",
        action="store_true",
        help="Allow playlist URLs (disabled by default).",
    )
    parser.add_argument("--retries", type=int, default=3, help="Download retry count.")
    parser.add_argument(
        "--instrument",
        default="vocals",
        choices=["piano", "guitar", "vocals"],
        help="Target instrument (default: vocals). Guitar uses basic-pitch; vocals uses rmvpe.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["pti", "piano-transcription-inference", "basic-pitch", "rmvpe"],
        help="Transcription backend. Default depends on instrument.",
    )
    parser.add_argument(
        "--bpm-detect-bin",
        default=None,
        help="Path to bpm_detect binary (vocals only). Falls back to PATH lookup, then 120 BPM.",
    )
    parser.add_argument(
        "--bpm-override",
        type=float,
        default=None,
        help="Override BPM for vocals transcription instead of auto-detecting.",
    )
    parser.add_argument(
        "--click-track",
        default=None,
        help=(
            "Path to write a click-track WAV overlay (vocals only). "
            "Requires --bpm-detect-bin. Output is the original audio mixed with "
            "a metronome click at the detected tempo."
        ),
    )
    parser.add_argument(
        "--f0-filter-frames",
        type=int,
        default=7,
        help=(
            "Median filter window in frames (10 ms each) applied to the raw F0 "
            "contour before semitone snapping (vocals only). Default 7 (70 ms), "
            "grounded in measured vibrato period of ~195 ms. Set to 1 to disable."
        ),
    )
    parser.add_argument(
        "--snap-to-beats",
        action="store_true",
        help=(
            "Snap note boundaries to the nearest 16th-note subdivision on the "
            "detected beat grid (vocals only). Requires --bpm-detect-bin. "
            "Improves rhythmic feel at the cost of exact onset timing."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Backend device setting, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--pti-checkpoint-path",
        default=None,
        help="Optional local path to piano-transcription-inference .pth checkpoint.",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Enable light denoise in preprocessing.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.05,
        help="Minimum note duration in seconds.",
    )
    parser.add_argument(
        "--note-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for note retention.",
    )
    parser.add_argument(
        "--quantize-grid-seconds",
        type=float,
        default=None,
        help="Optional quantization grid in seconds (disabled by default).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity.",
    )
    return parser


def _sanitize_filename(title: str) -> str:
    """Convert a video title to a safe filename with underscores."""
    name = title.strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s]+", "_", name)
    return name or "output"


def run_pipeline(args: argparse.Namespace) -> Path:
    """Run the end-to-end audio2midi pipeline."""
    user_settings = _load_user_settings()

    if args.bpm_detect_bin is None and "bpm_detect_bin" in user_settings:
        args.bpm_detect_bin = user_settings["bpm_detect_bin"]

    _project_root = Path(__file__).resolve().parents[1]

    # Use a temporary workdir for the initial download so we can derive the video name.
    tmp_workdir = (
        Path(args.workdir).expanduser().resolve()
        if args.workdir
        else _project_root / ".cache" / "audio2midi_tmp"
    )
    downloader = YouTubeDownloader(
        workdir=tmp_workdir,
        retries=args.retries,
        allow_playlist=args.allow_playlist,
    )
    download_result = downloader.download(args.youtube_url)

    video_name = _sanitize_filename(download_result.title or download_result.video_id)

    if args.workdir:
        workdir = Path(args.workdir).expanduser().resolve()
    else:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        workdir = _project_root / "outputs" / f"{timestamp}_{video_name}"

    workdir.mkdir(parents=True, exist_ok=True)
    extracted_dir = workdir / "extracted"
    preprocessed_dir = workdir / "preprocessed"

    # Move downloaded files into the final workdir if they came from the tmp location.
    if not args.workdir and not download_result.media_path.is_relative_to(workdir):
        import shutil
        from audio2midi.models import DownloadResult as _DR
        tmp_downloads = tmp_workdir / "downloads"
        dst_downloads = workdir / "downloads"
        if tmp_downloads.exists():
            shutil.copytree(str(tmp_downloads), str(dst_downloads), dirs_exist_ok=True)
            shutil.rmtree(str(tmp_downloads), ignore_errors=True)
        dest = dst_downloads / download_result.media_path.name
        download_result = _DR(
            video_id=download_result.video_id,
            title=download_result.title,
            media_path=dest,
            webpage_url=download_result.webpage_url,
            duration_seconds=download_result.duration_seconds,
        )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = workdir / f"{video_name}.mid"

    raw_wav_path = extracted_dir / f"{download_result.video_id}.wav"
    processed_wav_path = preprocessed_dir / f"{download_result.video_id}.wav"

    LOGGER.info("Extracting WAV from media...")
    extract_to_wav(download_result.media_path, raw_wav_path)

    LOGGER.info("Preprocessing audio...")
    preprocess_wav_file(
        input_wav_path=raw_wav_path,
        output_wav_path=processed_wav_path,
        denoise=args.denoise,
    )

    # Run bpm_detect on the raw mix before any stem separation.
    beat_times: np.ndarray = np.array([], dtype=float)
    bpm_detect_bin = (
        Path(args.bpm_detect_bin).expanduser().resolve() if args.bpm_detect_bin else None
    )
    if bpm_detect_bin is not None and args.instrument == "vocals":
        click_out = (
            Path(args.click_track).expanduser().resolve()
            if args.click_track
            else Path(tempfile.mktemp(suffix="_click.wav"))
        )
        detected_bpm = _run_bpm_detect(processed_wav_path, bpm_detect_bin, click_out)
        if detected_bpm is not None and click_out.exists():
            beat_times = _parse_beat_times(click_out)
            if args.click_track:
                LOGGER.info("Click track written: %s", click_out)
            elif click_out.exists():
                click_out.unlink(missing_ok=True)
        if args.bpm_override is None and detected_bpm is not None:
            args.bpm_override = detected_bpm
    elif args.bpm_detect_bin and args.instrument != "vocals":
        LOGGER.warning("--bpm-detect-bin is only used for vocals transcription")

    instrument = Instrument(args.instrument)
    LOGGER.info("Running transcription for instrument: %s", instrument.value)
    stems_dir = workdir / "stems"
    transcriber = create_transcriber(
        args.backend,
        instrument=instrument,
        device=args.device,
        pti_checkpoint_path=(
            Path(args.pti_checkpoint_path).expanduser().resolve()
            if args.pti_checkpoint_path
            else None
        ),
        bpm_override=args.bpm_override,
        beat_times=beat_times,
        snap_to_beats=getattr(args, "snap_to_beats", False),
        f0_filter_frames=args.f0_filter_frames,
        stems_dir=stems_dir,
    )
    transcription = transcriber.transcribe(processed_wav_path)

    LOGGER.info("Post-processing events...")
    cleaned = postprocess_transcription(
        transcription,
        min_duration_seconds=args.min_duration,
        note_on_threshold=args.note_threshold,
        quantize_grid_seconds=args.quantize_grid_seconds,
    )

    LOGGER.info("Writing MIDI output...")
    write_midi_file(cleaned, output_path, bpm=cleaned.bpm or 120.0)

    if not args.keep_intermediate:
        raw_wav_path.unlink(missing_ok=True)
        processed_wav_path.unlink(missing_ok=True)

    return output_path


def main(argv: list[str] | None = None) -> int:
    """CLI main function."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )
    try:
        output = run_pipeline(args)
    except Audio2MidiError as exc:
        LOGGER.error("%s", exc)
        if exc.__cause__ is not None:
            LOGGER.error("Cause: %s", exc.__cause__)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected failure.
        LOGGER.exception("Unexpected failure: %s", exc)
        return 1
    LOGGER.info("MIDI written: %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
