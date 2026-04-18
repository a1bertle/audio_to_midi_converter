# Implementation Plan: F0 Median Filter

**Date:** 2026-04-18
**Based on:** `notes.md` — window sweep measurements

## Goal

Apply a median filter to the raw RMVPE F0 contour before semitone snapping in
`_pitch_track_to_notes`, reducing unison repeats at the source rather than in
post-processing.

## Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Default window | 7 frames (70 ms) | Best unison/note-count tradeoff [measured] |
| Application scope | Per contiguous voiced segment | Avoid cross-silence pitch leakage |
| Fallback | window=1 (no-op) when `f0_filter_frames` ≤ 1 | Backward compatible |

## Changes

### 1. `audio2midi/transcribers/rmvpe.py`

**Add `_apply_f0_median_filter(f0_hz, window_frames)` helper** near top of module:

```python
def _apply_f0_median_filter(f0_hz: np.ndarray, window_frames: int) -> np.ndarray:
    """Apply median filter per voiced segment to suppress vibrato/pitch wobble.

    Operates on the continuous Hz contour before semitone snapping so that
    short oscillations (vibrato, portamento) are smoothed without affecting
    genuine pitch steps.
    """
    if window_frames <= 1:
        return f0_hz
    from scipy.ndimage import median_filter as _mf
    result = f0_hz.copy()
    voiced = f0_hz > 0
    # Find contiguous voiced segments and filter each independently.
    changes = np.diff(voiced.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for s, e in zip(starts, ends):
        if e - s >= window_frames:
            result[s:e] = _mf(f0_hz[s:e].astype(float), size=window_frames)
    return result
```

**Add `f0_filter_frames` parameter to `_pitch_track_to_notes`:**

```python
def _pitch_track_to_notes(
    f0_hz: np.ndarray,
    onset_times: np.ndarray,
    min_note_duration_s: float,
    bpm: float,
    beat_times: np.ndarray | None = None,
    snap_to_beats: bool = False,
    f0_filter_frames: int = 7,          # ← new
) -> list[NoteEvent]:
```

**Apply filter as first step in `_pitch_track_to_notes`**, before the voiced/semitone snap block:

```python
    # Smooth F0 contour to suppress vibrato before semitone quantization.
    f0_hz = _apply_f0_median_filter(f0_hz, f0_filter_frames)
```

**Add `f0_filter_frames` to `RmvpeTranscriber.__init__` and thread to `_pitch_track_to_notes`:**

```python
def __init__(
    self,
    ...
    f0_filter_frames: int = 7,
) -> None:
    ...
    self._f0_filter_frames = f0_filter_frames
```

In `transcribe()`:
```python
notes = _pitch_track_to_notes(
    f0, onset_times, min_note_s, bpm,
    beat_times=self._beat_times,
    snap_to_beats=self._snap_to_beats,
    f0_filter_frames=self._f0_filter_frames,
)
```

### 2. `audio2midi/transcribers/base.py`

Add `f0_filter_frames: int = 7` to `create_transcriber` and thread to `RmvpeTranscriber`.

### 3. `audio2midi/cli.py`

Add `--f0-filter-frames` CLI arg:

```python
parser.add_argument(
    "--f0-filter-frames",
    type=int,
    default=7,
    help=(
        "Median filter window in frames (10 ms each) applied to the raw F0 "
        "contour before semitone snapping (vocals only). "
        "Default 7 (70 ms). Set to 1 to disable."
    ),
)
```

Pass to `create_transcriber`:
```python
f0_filter_frames=args.f0_filter_frames,
```

## File/Line References

| File | Location | Change |
|------|----------|--------|
| `audio2midi/transcribers/rmvpe.py` | after `_parse_beat_times` | add `_apply_f0_median_filter` |
| `audio2midi/transcribers/rmvpe.py` | `_pitch_track_to_notes` signature | add `f0_filter_frames=7` |
| `audio2midi/transcribers/rmvpe.py` | first line of `_pitch_track_to_notes` body | apply filter |
| `audio2midi/transcribers/rmvpe.py` | `RmvpeTranscriber.__init__` | add `f0_filter_frames=7` |
| `audio2midi/transcribers/rmvpe.py` | `transcribe()` call to `_pitch_track_to_notes` | pass `f0_filter_frames` |
| `audio2midi/transcribers/base.py` | `create_transcriber` signature + body | add + thread `f0_filter_frames` |
| `audio2midi/cli.py` | `_build_parser` | add `--f0-filter-frames` |
| `audio2midi/cli.py` | `run_pipeline` | pass to `create_transcriber` |

## Validation Steps

See `validate.sh`. Checks:
1. Unison repeat count < 25 (current baseline: 38 raw, 25 after post-processing)
2. Note count within 10% of baseline (target: 180–220 for this fixture)
3. No crash when `--f0-filter-frames 1` (filter disabled)
4. No crash when `--f0-filter-frames 31` (large window)
