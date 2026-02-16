"""Project-specific exceptions."""


class Audio2MidiError(Exception):
    """Base exception for audio2midi pipeline failures."""


class MissingDependencyError(Audio2MidiError):
    """Raised when a required external binary or package is missing."""


class InvalidInputError(Audio2MidiError):
    """Raised when user input is invalid."""


class DownloadError(Audio2MidiError):
    """Raised when media download fails."""


class ExtractionError(Audio2MidiError):
    """Raised when audio extraction fails."""


class TranscriptionError(Audio2MidiError):
    """Raised when transcription backend fails."""
