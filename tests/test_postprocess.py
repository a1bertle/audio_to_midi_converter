"""Tests for transcription post-processing."""

from audio2midi.models import NoteEvent, PedalEvent, TranscriptionResult
from audio2midi.postprocess import postprocess_transcription


def test_postprocess_filters_short_and_low_confidence_notes() -> None:
    result = TranscriptionResult(
        notes=[
            NoteEvent(pitch=60, start=0.0, end=0.03, velocity=80, confidence=0.9),
            NoteEvent(pitch=62, start=0.0, end=0.40, velocity=80, confidence=0.4),
            NoteEvent(pitch=64, start=0.0, end=0.40, velocity=80, confidence=0.95),
        ],
        pedals=[],
    )
    cleaned = postprocess_transcription(
        result,
        min_duration_seconds=0.05,
        note_on_threshold=0.5,
    )
    assert len(cleaned.notes) == 1
    assert cleaned.notes[0].pitch == 64


def test_postprocess_extends_note_with_pedal() -> None:
    result = TranscriptionResult(
        notes=[NoteEvent(pitch=60, start=0.0, end=1.0, velocity=80, confidence=0.9)],
        pedals=[PedalEvent(start=0.5, end=1.5, value=127)],
    )
    cleaned = postprocess_transcription(
        result,
        min_duration_seconds=0.05,
        note_on_threshold=0.5,
    )
    assert len(cleaned.notes) == 1
    assert cleaned.notes[0].end == 1.5


def test_postprocess_quantizes_to_grid() -> None:
    result = TranscriptionResult(
        notes=[NoteEvent(pitch=67, start=0.12, end=0.41, velocity=90, confidence=0.9)],
        pedals=[],
    )
    cleaned = postprocess_transcription(
        result,
        min_duration_seconds=0.05,
        note_on_threshold=0.5,
        quantize_grid_seconds=0.1,
    )
    assert cleaned.notes[0].start == 0.1
    assert cleaned.notes[0].end == 0.4
