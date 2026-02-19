"""Adapter for basic-pitch backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audio2midi.exceptions import MissingDependencyError, TranscriptionError
from audio2midi.models import Instrument, NoteEvent, TranscriptionResult
from audio2midi.transcribers.base import BaseTranscriber

# Keyword arguments passed to basic_pitch.inference.predict() per instrument.
_INSTRUMENT_PREDICT_KWARGS: dict[Instrument, dict[str, Any]] = {
    Instrument.PIANO: {},
    Instrument.GUITAR: {
        "minimum_frequency": 82.0,
        "maximum_frequency": 1318.0,
        "multiple_pitch_bends": True,
        "frame_threshold": 0.15,
        "onset_threshold": 0.3,
        "minimum_note_length": 130,
    },
}


class BasicPitchTranscriber(BaseTranscriber):
    """Transcriber using Spotify Basic Pitch."""

    def __init__(self, instrument: Instrument = Instrument.PIANO) -> None:
        self._instrument = instrument
        self._predict_kwargs = _INSTRUMENT_PREDICT_KWARGS.get(instrument, {})

    def transcribe(self, wav_path: Path) -> TranscriptionResult:
        """Run basic-pitch prediction and map to note events."""
        try:
            from basic_pitch.inference import predict  # type: ignore
        except ImportError as exc:
            raise MissingDependencyError(
                "basic-pitch is not installed. Install with: pip install basic-pitch"
            ) from exc
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH  # type: ignore

            # Use the ONNX model to avoid TF SavedModel compatibility issues.
            onnx_path = Path(str(ICASSP_2022_MODEL_PATH) + ".onnx")
            model_path = str(onnx_path) if onnx_path.exists() else ICASSP_2022_MODEL_PATH

            _model_output, _midi_data, note_events = predict(
                str(wav_path), model_path, **self._predict_kwargs
            )
        except Exception as exc:  # pragma: no cover - backend-specific failure.
            raise TranscriptionError("basic-pitch transcription failed.") from exc
        notes: list[NoteEvent] = []
        for event in note_events:
            # The tuple shape is start_time, end_time, pitch, confidence.
            start, end, pitch, confidence = event[:4]
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
        return TranscriptionResult(
            notes=notes, pedals=[], instrument=self._instrument
        )
