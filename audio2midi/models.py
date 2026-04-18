"""Shared data models for the audio2midi pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Instrument(str, Enum):
    """Supported target instruments."""

    PIANO = "piano"
    GUITAR = "guitar"
    VOCALS = "vocals"


@dataclass(slots=True)
class NoteEvent:
    """Represents a single note event."""

    pitch: int
    start: float
    end: float
    velocity: int = 80
    confidence: float | None = None


@dataclass(slots=True)
class PedalEvent:
    """Represents a sustain pedal event window."""

    start: float
    end: float
    value: int = 127


@dataclass(slots=True)
class TranscriptionResult:
    """Container for note and pedal events."""

    notes: list[NoteEvent] = field(default_factory=list)
    pedals: list[PedalEvent] = field(default_factory=list)
    instrument: Instrument = Instrument.PIANO
    bpm: float | None = None


@dataclass(slots=True)
class DownloadResult:
    """Metadata and path for a downloaded source."""

    video_id: str
    media_path: Path
    title: str | None = None
    webpage_url: str | None = None
    duration_seconds: float | None = None
