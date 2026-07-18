"""Regression tests for RMVPE note segmentation.

Run standalone with: ``python -m pytest tests/test_rmvpe.py``.
"""

import numpy as np
import pytest

from audio2midi.transcribers import rmvpe


def test_onset_split_gate_uses_two_beat_half_note(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sub-half-note hold must not be split at an interior onset."""
    monkeypatch.setattr(rmvpe, "_merge_same_pitch", lambda raw, *_args: raw)
    monkeypatch.setattr(rmvpe, "_gap_fill", lambda raw, *_args: raw)

    f0_hz = np.concatenate((np.full(75, 440.0), np.zeros(1)))
    notes = rmvpe._pitch_track_to_notes(
        f0_hz=f0_hz,
        onset_times=np.array([0.35]),
        min_note_duration_s=0.05,
        bpm=120.0,
        f0_filter_frames=1,
    )

    assert len(notes) == 1
    assert notes[0].start == pytest.approx(0.0)
    assert notes[0].end == pytest.approx(0.75)


def test_gap_fill_ceiling_uses_two_beat_half_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The F0 gap-fill ceiling must be two beats at the selected tempo."""
    observed: list[float] = []

    def capture_gap(
        raw: list[tuple[float, float, int]],
        _f0_hz: np.ndarray,
        max_gap_s: float,
    ) -> list[tuple[float, float, int]]:
        observed.append(max_gap_s)
        return raw

    monkeypatch.setattr(rmvpe, "_gap_fill", capture_gap)
    f0_hz = np.concatenate((np.full(10, 440.0), np.zeros(1)))

    rmvpe._pitch_track_to_notes(
        f0_hz=f0_hz,
        onset_times=np.array([]),
        min_note_duration_s=0.05,
        bpm=120.0,
        f0_filter_frames=1,
    )

    assert observed == [pytest.approx(1.0)]


def test_short_internal_unvoiced_gaps_only_preserves_enclosed_runs() -> None:
    voiced = np.array([
        False, True, True, False, False, True, False, False, False, True, False,
    ])

    preserved = rmvpe._short_internal_unvoiced_gaps(voiced, max_gap_frames=2)

    assert np.flatnonzero(preserved).tolist() == [3, 4]


def test_pitch_track_flushes_final_voiced_note() -> None:
    notes = rmvpe._pitch_track_to_notes(
        f0_hz=np.full(10, 440.0),
        onset_times=np.array([]),
        min_note_duration_s=0.05,
        bpm=120.0,
        f0_filter_frames=1,
    )

    assert len(notes) == 1
    assert notes[0].end == pytest.approx(0.1)


def test_clip_notes_to_voiced_spans_splits_internal_gap() -> None:
    raw = [(0.0, 0.3, 69)]
    f0_hz = np.concatenate((
        np.full(10, 440.0),
        np.zeros(10),
        np.full(10, 440.0),
    ))

    clipped = rmvpe._clip_notes_to_voiced_spans(raw, f0_hz, 0.05)

    assert clipped == [(0.0, 0.1, 69), (0.2, 0.3, 69)]
