"""Transcriber interface and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from audio2midi.exceptions import InvalidInputError
from audio2midi.models import Instrument, TranscriptionResult

# Valid backends for each instrument.
_VALID_BACKENDS: dict[Instrument, set[str]] = {
    Instrument.PIANO: {"pti", "piano-transcription-inference", "basic-pitch", "basic_pitch"},
    Instrument.GUITAR: {"basic-pitch", "basic_pitch"},
    Instrument.VOCALS: {"rmvpe"},
}

# Default backend when none is specified.
_DEFAULT_BACKEND: dict[Instrument, str] = {
    Instrument.PIANO: "pti",
    Instrument.GUITAR: "basic-pitch",
    Instrument.VOCALS: "rmvpe",
}


class BaseTranscriber(ABC):
    """Abstract transcription interface."""

    @abstractmethod
    def transcribe(self, wav_path: Path) -> TranscriptionResult:
        """Run inference and return note/pedal events."""


def create_transcriber(
    backend: str | None,
    instrument: Instrument = Instrument.PIANO,
    device: str = "cpu",
    pti_checkpoint_path: Path | None = None,
    rmvpe_checkpoint_path: Path | None = None,
    bpm_override: float | None = None,
    beat_times=None,
    snap_to_beats: bool = False,
    f0_filter_frames: int = 7,
) -> BaseTranscriber:
    """Create a transcriber backend by name."""
    normalized = (backend or _DEFAULT_BACKEND[instrument]).strip().lower()

    valid = _VALID_BACKENDS[instrument]
    if normalized not in valid:
        raise InvalidInputError(
            f"Backend '{normalized}' is not supported for instrument '{instrument.value}'. "
            f"Valid backends: {', '.join(sorted(valid))}"
        )

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

        return BasicPitchTranscriber(instrument=instrument)
    if normalized == "rmvpe":
        from audio2midi.transcribers.rmvpe import RmvpeTranscriber

        return RmvpeTranscriber(
            checkpoint_path=rmvpe_checkpoint_path,
            device=device,
            bpm_override=bpm_override,
            beat_times=beat_times,
            snap_to_beats=snap_to_beats,
            f0_filter_frames=f0_filter_frames,
        )
    raise InvalidInputError(
        f"Unsupported backend '{backend}'."
    )
