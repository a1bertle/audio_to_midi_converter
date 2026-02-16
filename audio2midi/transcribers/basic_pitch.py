"""Adapter for basic-pitch backend."""

from __future__ import annotations

from pathlib import Path

from audio2midi.exceptions import MissingDependencyError, TranscriptionError
from audio2midi.models import NoteEvent, TranscriptionResult
from audio2midi.transcribers.base import BaseTranscriber


class BasicPitchTranscriber(BaseTranscriber):
    """Transcriber using Spotify Basic Pitch."""

    def transcribe(self, wav_path: Path) -> TranscriptionResult:
        """Run basic-pitch prediction and map to note events."""
        try:
            from basic_pitch.inference import predict  # type: ignore
        except ImportError as exc:
            raise MissingDependencyError(
                "basic-pitch is not installed. Install with: pip install basic-pitch"
            ) from exc
        try:
            _model_output, _midi_data, note_events = predict(str(wav_path))
        except Exception as exc:  # pragma: no cover - backend-specific failure.
            raise TranscriptionError("basic-pitch transcription failed.") from exc
        notes: list[NoteEvent] = []
        for event in note_events:
            # The tuple shape is start_time, end_time, pitch, confidence.
            start, end, pitch, confidence = event
            velocity = int(max(1, min(127, round(float(confidence) * 127))))
            notes.append(
                NoteEvent(
                    pitch=int(pitch),
                    start=float(start),
                    end=float(end),
                    velocity=velocity,
                    confidence=float(confidence),
                )
            )
        return TranscriptionResult(notes=notes, pedals=[])
