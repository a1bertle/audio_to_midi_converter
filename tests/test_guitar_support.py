"""Tests for guitar instrument support."""

from pathlib import Path

from mido import MidiFile

from audio2midi.midi_writer import write_midi_file
from audio2midi.models import Instrument, NoteEvent, TranscriptionResult
from audio2midi.transcribers.basic_pitch import (
    BasicPitchTranscriber,
    _INSTRUMENT_PREDICT_KWARGS,
)


# -- BasicPitchTranscriber guitar preset tests --


def test_guitar_transcriber_has_predict_kwargs() -> None:
    t = BasicPitchTranscriber(instrument=Instrument.GUITAR)
    assert t._predict_kwargs == _INSTRUMENT_PREDICT_KWARGS[Instrument.GUITAR]
    assert t._predict_kwargs["minimum_frequency"] == 82.0
    assert t._predict_kwargs["maximum_frequency"] == 1318.0
    assert t._predict_kwargs["multiple_pitch_bends"] is True


def test_piano_transcriber_has_empty_predict_kwargs() -> None:
    t = BasicPitchTranscriber(instrument=Instrument.PIANO)
    assert t._predict_kwargs == {}


def test_default_transcriber_is_piano() -> None:
    t = BasicPitchTranscriber()
    assert t._instrument == Instrument.PIANO
    assert t._predict_kwargs == {}


# -- MIDI writer Program Change tests --


def test_midi_writer_guitar_program_change(tmp_path: Path) -> None:
    result = TranscriptionResult(
        notes=[NoteEvent(pitch=52, start=0.0, end=0.5, velocity=80)],
        pedals=[],
        instrument=Instrument.GUITAR,
    )
    out = tmp_path / "guitar.mid"
    write_midi_file(result, out)

    midi = MidiFile(str(out))
    messages = list(midi.tracks[0])
    program_changes = [m for m in messages if m.type == "program_change"]
    assert len(program_changes) == 1
    assert program_changes[0].program == 25  # Acoustic Guitar (Steel)


def test_midi_writer_piano_program_change(tmp_path: Path) -> None:
    result = TranscriptionResult(
        notes=[NoteEvent(pitch=60, start=0.0, end=0.5, velocity=80)],
        pedals=[],
        instrument=Instrument.PIANO,
    )
    out = tmp_path / "piano.mid"
    write_midi_file(result, out)

    midi = MidiFile(str(out))
    messages = list(midi.tracks[0])
    program_changes = [m for m in messages if m.type == "program_change"]
    assert len(program_changes) == 1
    assert program_changes[0].program == 0  # Acoustic Grand Piano


def test_midi_writer_default_is_piano(tmp_path: Path) -> None:
    result = TranscriptionResult(
        notes=[NoteEvent(pitch=60, start=0.0, end=0.5, velocity=80)],
        pedals=[],
    )
    out = tmp_path / "default.mid"
    write_midi_file(result, out)

    midi = MidiFile(str(out))
    messages = list(midi.tracks[0])
    program_changes = [m for m in messages if m.type == "program_change"]
    assert len(program_changes) == 1
    assert program_changes[0].program == 0  # Default is piano
