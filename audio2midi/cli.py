"""Command-line entrypoint for audio2midi."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from audio2midi.downloader import YouTubeDownloader
from audio2midi.exceptions import Audio2MidiError
from audio2midi.extractor import extract_to_wav
from audio2midi.midi_writer import write_midi_file
from audio2midi.postprocess import postprocess_transcription
from audio2midi.preprocess import preprocess_wav_file
from audio2midi.transcribers.base import create_transcriber


LOGGER = logging.getLogger("audio2midi")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio2midi",
        description=(
            "Convert noisy YouTube piano performances into MIDI files. "
            "This command runs completely on-device after download."
        ),
    )
    parser.add_argument("--youtube-url", required=True, help="YouTube video URL.")
    parser.add_argument("--output", required=True, help="Output MIDI path.")
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
        "--backend",
        default="pti",
        choices=["pti", "piano-transcription-inference", "basic-pitch"],
        help="Transcription backend.",
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


def run_pipeline(args: argparse.Namespace) -> Path:
    """Run the end-to-end audio2midi pipeline."""
    output_path = Path(args.output).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()
    extracted_dir = workdir / "extracted"
    preprocessed_dir = workdir / "preprocessed"

    downloader = YouTubeDownloader(
        workdir=workdir,
        retries=args.retries,
        allow_playlist=args.allow_playlist,
    )
    download_result = downloader.download(args.youtube_url)
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

    LOGGER.info("Running transcription backend: %s", args.backend)
    transcriber = create_transcriber(
        args.backend,
        device=args.device,
        pti_checkpoint_path=(
            Path(args.pti_checkpoint_path).expanduser().resolve()
            if args.pti_checkpoint_path
            else None
        ),
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
    write_midi_file(cleaned, output_path)

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
