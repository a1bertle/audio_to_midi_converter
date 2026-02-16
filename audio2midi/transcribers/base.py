"""Transcriber interface and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from audio2midi.exceptions import InvalidInputError
from audio2midi.models import TranscriptionResult


class BaseTranscriber(ABC):
    """Abstract transcription interface."""

    @abstractmethod
    def transcribe(self, wav_path: Path) -> TranscriptionResult:
        """Run inference and return note/pedal events."""


def create_transcriber(
    backend: str,
    device: str = "cpu",
    pti_checkpoint_path: Path | None = None,
) -> BaseTranscriber:
    """Create a transcriber backend by name."""
    normalized = backend.strip().lower()
    if normalized in {"pti", "piano-transcription-inference"}:
        from audio2midi.transcribers.piano_transcription_inference import (
            PianoTranscriptionInferenceTranscriber,
        )

        return PianoTranscriptionInferenceTranscriber(
            device=device,
            checkpoint_path=pti_checkpoint_path,
        )
    if normalized in {"basic-pitch", "basic_pitch"}:
        from audio2midi.transcribers.basic_pitch import BasicPitchTranscriber

        return BasicPitchTranscriber()
    raise InvalidInputError(
        f"Unsupported backend '{backend}'. Expected one of: "
        "pti, piano-transcription-inference, basic-pitch"
    )
