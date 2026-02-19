"""Tests for transcriber factory behavior."""

import pytest

from audio2midi.exceptions import InvalidInputError
from audio2midi.models import Instrument
from audio2midi.transcribers.base import create_transcriber


def test_create_transcriber_invalid_backend_raises() -> None:
    with pytest.raises(InvalidInputError):
        create_transcriber("invalid-backend")


def test_create_transcriber_piano_defaults_to_pti() -> None:
    t = create_transcriber(None, instrument=Instrument.PIANO)
    from audio2midi.transcribers.piano_transcription_inference import (
        PianoTranscriptionInferenceTranscriber,
    )

    assert isinstance(t, PianoTranscriptionInferenceTranscriber)


def test_create_transcriber_guitar_defaults_to_basic_pitch() -> None:
    t = create_transcriber(None, instrument=Instrument.GUITAR)
    from audio2midi.transcribers.basic_pitch import BasicPitchTranscriber

    assert isinstance(t, BasicPitchTranscriber)
    assert t._instrument == Instrument.GUITAR


def test_create_transcriber_guitar_rejects_pti() -> None:
    with pytest.raises(InvalidInputError, match="not supported for instrument 'guitar'"):
        create_transcriber("pti", instrument=Instrument.GUITAR)


def test_create_transcriber_guitar_accepts_basic_pitch() -> None:
    t = create_transcriber("basic-pitch", instrument=Instrument.GUITAR)
    from audio2midi.transcribers.basic_pitch import BasicPitchTranscriber

    assert isinstance(t, BasicPitchTranscriber)
    assert t._instrument == Instrument.GUITAR


def test_create_transcriber_piano_accepts_basic_pitch() -> None:
    t = create_transcriber("basic-pitch", instrument=Instrument.PIANO)
    from audio2midi.transcribers.basic_pitch import BasicPitchTranscriber

    assert isinstance(t, BasicPitchTranscriber)
    assert t._instrument == Instrument.PIANO
