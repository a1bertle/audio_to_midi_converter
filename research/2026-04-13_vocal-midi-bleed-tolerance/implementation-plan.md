# Implementation Plan: Vocals Instrument Path in audio2midi

> **Status (2026-04-17): BLOCKED**
> Follow-up testing (htdemucs_ft + subjective MIDI quality assessment) confirmed
> the separation stage is the bottleneck. The basic-pitch threshold parameters
> documented here are correct, but wiring them up without a better separation
> front-end will not produce usable MIDI. This plan should be revisited after
> the separation problem is solved. See `notes.md` findings F7–F10.

**Date:** 2026-04-13
**Session:** `research/2026-04-13_vocal-midi-bleed-tolerance/`
**Derived from:** `notes.md` findings — basic-pitch tolerates htdemucs vocals/other
bleed; 99.8% in-range notes, mean confidence 0.592 [measured].

---

## Objective

Add `Instrument.VOCALS` as a supported instrument in the transcription pipeline,
backed by `BasicPitchTranscriber` with vocals-optimised threshold parameters.

---

## Files to Change

| File | Change |
|------|--------|
| `audio2midi/models.py` | Add `VOCALS = "vocals"` to `Instrument` enum |
| `audio2midi/transcribers/base.py` | Add `Instrument.VOCALS` to `_VALID_BACKENDS` and `_DEFAULT_BACKEND` |
| `audio2midi/transcribers/basic_pitch.py` | Add `Instrument.VOCALS` entry in `_INSTRUMENT_PREDICT_KWARGS` |

No new files needed. No new dependencies.

---

## Step-by-step

### Step 1 — Add `VOCALS` to `Instrument` enum
**File:** `audio2midi/models.py`
- Add `VOCALS = "vocals"` alongside `PIANO` and `GUITAR`.

### Step 2 — Register in `_VALID_BACKENDS` and `_DEFAULT_BACKEND`
**File:** `audio2midi/transcribers/base.py`
```python
_VALID_BACKENDS = {
    Instrument.PIANO:  {"pti", "piano-transcription-inference", "basic-pitch", "basic_pitch"},
    Instrument.GUITAR: {"basic-pitch", "basic_pitch"},
    Instrument.VOCALS: {"basic-pitch", "basic_pitch"},   # ← add
}
_DEFAULT_BACKEND = {
    Instrument.PIANO:  "pti",
    Instrument.GUITAR: "basic-pitch",
    Instrument.VOCALS: "basic-pitch",                    # ← add
}
```

### Step 3 — Add vocals kwargs to `BasicPitchTranscriber`
**File:** `audio2midi/transcribers/basic_pitch.py`
```python
_INSTRUMENT_PREDICT_KWARGS = {
    Instrument.PIANO: {},
    Instrument.GUITAR: {
        "minimum_frequency": 82.0,
        "maximum_frequency": 1318.0,
        "multiple_pitch_bends": True,
        "frame_threshold": 0.15,
        "onset_threshold": 0.3,
        "minimum_note_length": 130,
    },
    Instrument.VOCALS: {
        "minimum_frequency": 80.0,    # ~E2 — below lowest expected vocal
        "maximum_frequency": 1400.0,  # ~F6 — above highest expected vocal
        "multiple_pitch_bends": False, # monophonic vocal assumption
        "frame_threshold": 0.3,        # measured: 3.8% low-conf notes at this threshold
        "onset_threshold": 0.4,        # slightly above guitar default to suppress bleed onsets
        "minimum_note_length": 100,    # ms — suppresses short bleed transients
    },
}
```

**Parameter provenance:**
| Parameter | Value | Provenance |
|-----------|-------|------------|
| minimum_frequency | 80.0 Hz | [assumed — below E2 vocal floor] |
| maximum_frequency | 1400.0 Hz | [assumed — above F6 vocal ceiling] |
| frame_threshold | 0.3 | [measured — 99.8% in-range notes at this setting] |
| onset_threshold | 0.4 | [assumed — suppress bleed onsets vs guitar default 0.3] |
| minimum_note_length | 100 ms | [assumed — filter sub-phrase bleed transients] |
| multiple_pitch_bends | False | [assumed — monophonic vocal line] |

---

## Validation Steps

After implementation:

1. Run existing test suite — should pass unchanged:
   ```
   source .venv/bin/activate && python3 -m pytest tests/ -q
   ```

2. Smoke-test `create_transcriber` with `Instrument.VOCALS`:
   ```python
   from audio2midi.transcribers.base import create_transcriber
   from audio2midi.models import Instrument
   t = create_transcriber(None, instrument=Instrument.VOCALS)
   assert t is not None
   ```

3. Re-run session validate script to confirm transcription output unchanged:
   ```
   ./research/2026-04-13_vocal-midi-bleed-tolerance/validate.sh
   ```
   Expected: note_count ~650, in_range_fraction ≥ 0.99, confidence_mean ≥ 0.55.

---

## Out of Scope for This Plan

- Demucs stem separation integration (separate concern)
- Vocals-specific post-processing (pitch smoothing, vibrato removal)
- Ground-truth evaluation (no reference MIDI available)
- CREPE or MT3 integration (not in project venv)
