"""Tests for transcriber factory behavior."""

import pytest

from audio2midi.exceptions import InvalidInputError
from audio2midi.transcribers.base import create_transcriber


def test_create_transcriber_invalid_backend_raises() -> None:
    with pytest.raises(InvalidInputError):
        create_transcriber("invalid-backend")
