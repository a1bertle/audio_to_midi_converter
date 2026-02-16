"""Post-processing for transcription event cleanup."""

from __future__ import annotations

from dataclasses import replace

from audio2midi.models import NoteEvent, PedalEvent, TranscriptionResult


def _is_note_valid(
    note: NoteEvent,
    min_duration_seconds: float,
    note_on_threshold: float,
) -> bool:
    duration = note.end - note.start
    if duration < min_duration_seconds:
        return False
    if note.confidence is not None and note.confidence < note_on_threshold:
        return False
    return note.pitch >= 0


def _extend_end_with_pedals(note: NoteEvent, pedals: list[PedalEvent]) -> float:
    end = note.end
    for pedal in pedals:
        if pedal.value <= 0:
            continue
        if pedal.start <= end <= pedal.end and note.start <= pedal.end:
            end = max(end, pedal.end)
    return end


def _quantize_time(value: float, grid_seconds: float | None) -> float:
    if grid_seconds is None or grid_seconds <= 0:
        return value
    return round(value / grid_seconds) * grid_seconds


def postprocess_transcription(
    result: TranscriptionResult,
    min_duration_seconds: float = 0.05,
    note_on_threshold: float = 0.5,
    quantize_grid_seconds: float | None = None,
) -> TranscriptionResult:
    """Filter noisy notes, apply pedal extension, and optional quantization."""
    pedals = sorted(result.pedals, key=lambda pedal: (pedal.start, pedal.end))
    notes = sorted(result.notes, key=lambda note: (note.start, note.pitch))
    cleaned_notes: list[NoteEvent] = []
    for note in notes:
        if not _is_note_valid(
            note,
            min_duration_seconds=min_duration_seconds,
            note_on_threshold=note_on_threshold,
        ):
            continue
        adjusted_end = _extend_end_with_pedals(note, pedals)
        start = _quantize_time(note.start, quantize_grid_seconds)
        end = _quantize_time(adjusted_end, quantize_grid_seconds)
        if end <= start:
            end = start + min_duration_seconds
        cleaned_notes.append(replace(note, start=start, end=end))
    return TranscriptionResult(notes=cleaned_notes, pedals=pedals)
