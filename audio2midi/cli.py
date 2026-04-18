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

from audio2midi.downloader import YouTubeDownloader
from audio2midi.exceptions import Audio2MidiError
from audio2midi.extractor import extract_to_wav
from audio2midi.midi_writer import write_midi_file
from audio2midi.models import Instrument
from audio2midi.postprocess import postprocess_transcription
from audio2midi.preprocess import preprocess_wav_file
from audio2midi.transcribers.base import create_transcriber


LOGGER = logging.getLogger("audio2midi")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio2midi",
        description=(
            "Convert YouTube audio performances into MIDI files. "
            "Supports piano (default) and guitar instruments. "
            "Runs completely on-device after download."
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
        default=".cache/audio2midi",
        help="Working directory for downloads and intermediates.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep extracted and preprocessed WAV files.",
    )
    parser.add_argument(
        "--allow-playlist",
        action="store_true",
        help="Allow playlist URLs (disabled by default).",
    )
    parser.add_argument("--retries", type=int, default=3, help="Download retry count.")
    parser.add_argument(
        "--instrument",
        default="piano",
        choices=["piano", "guitar", "vocals"],
        help="Target instrument (default: piano). Guitar uses basic-pitch; vocals uses rmvpe.",
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

    workdir = Path(args.workdir).expanduser().resolve()
    extracted_dir = workdir / "extracted"
    preprocessed_dir = workdir / "preprocessed"

    downloader = YouTubeDownloader(
        workdir=workdir,
        retries=args.retries,
        allow_playlist=args.allow_playlist,
    )
    download_result = downloader.download(args.youtube_url)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        filename = _sanitize_filename(download_result.title or download_result.video_id)
        output_path = Path(f"{filename}.mid").resolve()
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

    instrument = Instrument(args.instrument)
    LOGGER.info("Running transcription for instrument: %s", instrument.value)
    transcriber = create_transcriber(
        args.backend,
        instrument=instrument,
        device=args.device,
        pti_checkpoint_path=(
            Path(args.pti_checkpoint_path).expanduser().resolve()
            if args.pti_checkpoint_path
            else None
        ),
        bpm_detect_bin=(
            Path(args.bpm_detect_bin).expanduser().resolve()
            if args.bpm_detect_bin
            else None
        ),
        bpm_override=args.bpm_override,
        click_track_path=(
            Path(args.click_track).expanduser().resolve()
            if args.click_track
            else None
        ),
    )
    transcription = transcriber.transcribe(processed_wav_path)

    if args.click_track:
        click_path = Path(args.click_track).expanduser().resolve()
        if click_path.exists():
            LOGGER.info("Click track written: %s", click_path)
        else:
            LOGGER.warning("Click track was requested but not produced at %s", click_path)

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
