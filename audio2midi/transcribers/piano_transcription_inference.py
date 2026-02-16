"""Adapter for piano-transcription-inference backend."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

from audio2midi.exceptions import MissingDependencyError, TranscriptionError
from audio2midi.models import TranscriptionResult
from audio2midi.transcribers.base import BaseTranscriber


DEFAULT_CHECKPOINT_NAME = "note_F1=0.9677_pedal_F1=0.9186.pth"
DEFAULT_CHECKPOINT_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
MIN_CHECKPOINT_BYTES = 160_000_000
DOWNLOAD_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0
DOWNLOAD_TIMEOUT_SECONDS = 60
CHECKPOINT_USER_AGENT = "audio2midi/0.1 (+https://github.com/)"


class PianoTranscriptionInferenceTranscriber(BaseTranscriber):
    """Transcriber using piano-transcription-inference package."""

    def __init__(
        self,
        device: str = "cpu",
        checkpoint_path: Path | None = None,
    ) -> None:
        self.device = device
        self.checkpoint_path = checkpoint_path

    def _resolve_checkpoint_path(self) -> Path:
        configured = (
            self.checkpoint_path
            or os.getenv("AUDIO2MIDI_PTI_CHECKPOINT_PATH")
        )
        if configured:
            return Path(configured).expanduser().resolve()
        return (
            Path.home()
            / "piano_transcription_inference_data"
            / DEFAULT_CHECKPOINT_NAME
        ).resolve()

    @staticmethod
    def _resolve_checkpoint_url() -> str:
        return os.getenv("AUDIO2MIDI_PTI_CHECKPOINT_URL", DEFAULT_CHECKPOINT_URL)

    @staticmethod
    def _validate_checkpoint_file(checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise TranscriptionError(
                f"Checkpoint file missing: {checkpoint_path}"
            )
        if checkpoint_path.stat().st_size < MIN_CHECKPOINT_BYTES:
            raise TranscriptionError(
                "Downloaded checkpoint is too small or corrupted: "
                f"{checkpoint_path}"
            )

    @staticmethod
    def _download_with_urllib(url: str, output_path: Path) -> None:
        request = Request(url, headers={"User-Agent": CHECKPOINT_USER_AGENT})
        with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            with output_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)

    @staticmethod
    def _download_with_curl(url: str, output_path: Path) -> None:
        if shutil.which("curl") is None:
            raise TranscriptionError(
                "curl is not installed for fallback download."
            )
        command = [
            "curl",
            "-fL",
            "--retry",
            str(DOWNLOAD_RETRIES),
            "--retry-delay",
            str(int(RETRY_DELAY_SECONDS)),
            "--connect-timeout",
            "20",
            "--output",
            str(output_path),
            url,
        ]
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise TranscriptionError(
                "curl fallback checkpoint download failed. "
                f"stderr: {proc.stderr.strip()}"
            )

    def _ensure_checkpoint(self, checkpoint_path: Path) -> None:
        if (
            checkpoint_path.exists()
            and checkpoint_path.stat().st_size >= MIN_CHECKPOINT_BYTES
        ):
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        url = self._resolve_checkpoint_url()
        temp_path = checkpoint_path.with_suffix(".part")
        errors: list[str] = []

        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            temp_path.unlink(missing_ok=True)
            try:
                self._download_with_urllib(url, temp_path)
                temp_path.replace(checkpoint_path)
                self._validate_checkpoint_file(checkpoint_path)
                return
            except Exception as exc:  # pragma: no cover - network/runtime dependent.
                errors.append(f"urllib attempt {attempt} failed: {exc}")
                if attempt < DOWNLOAD_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)

        temp_path.unlink(missing_ok=True)
        try:
            self._download_with_curl(url, temp_path)
            temp_path.replace(checkpoint_path)
            self._validate_checkpoint_file(checkpoint_path)
            return
        except Exception as exc:  # pragma: no cover - network/runtime dependent.
            errors.append(f"curl fallback failed: {exc}")

        checkpoint_path.unlink(missing_ok=True)
        temp_path.unlink(missing_ok=True)
        joined_errors = "; ".join(errors)
        raise TranscriptionError(
            "Failed to download piano-transcription-inference checkpoint. "
            "Provide --pti-checkpoint-path or set "
            "AUDIO2MIDI_PTI_CHECKPOINT_PATH. "
            f"Download URL: {url}. Errors: {joined_errors}"
        )

    @staticmethod
    def _ensure_matplotlib_cache_dir() -> None:
        """Set a writable matplotlib cache directory if one is not configured."""
        if os.getenv("MPLCONFIGDIR"):
            return
        mpl_dir = Path.home() / ".cache" / "audio2midi" / "matplotlib"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    def transcribe(self, wav_path: Path) -> TranscriptionResult:
        """Run transcription and convert generated MIDI into event objects."""
        self._ensure_matplotlib_cache_dir()
        try:
            from audio2midi.midi_writer import read_midi_as_transcription
            import librosa  # type: ignore
            from piano_transcription_inference import (  # type: ignore
                PianoTranscription,
                sample_rate,
            )
        except ImportError as exc:
            raise MissingDependencyError(
                "Required dependency missing for piano-transcription-inference. "
                "Install with: pip install piano-transcription-inference mido"
            ) from exc

        checkpoint_path = self._resolve_checkpoint_path()
        self._ensure_checkpoint(checkpoint_path)
        try:
            transcriptor = PianoTranscription(
                device=self.device,
                checkpoint_path=str(checkpoint_path),
            )
        except Exception as exc:  # pragma: no cover - backend-specific failure.
            raise TranscriptionError(
                "Failed to initialize piano-transcription-inference backend."
            ) from exc
        try:
            # Use modern librosa loader instead of upstream helper to avoid
            # older librosa API assumptions in piano-transcription-inference.
            audio, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
                midi_path = Path(handle.name)
            transcriptor.transcribe(audio, str(midi_path))
            result = read_midi_as_transcription(midi_path)
        except Exception as exc:  # pragma: no cover - backend-specific failure.
            raise TranscriptionError(
                f"piano-transcription-inference failed: {exc}"
            ) from exc
        finally:
            if "midi_path" in locals() and midi_path.exists():
                midi_path.unlink(missing_ok=True)
        return result
