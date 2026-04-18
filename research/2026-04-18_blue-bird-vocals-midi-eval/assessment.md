# Assessment: Naruto_Shippuden_Opening_3_Blue_Bird.mid

**Date:** 2026-04-18
**Input:** `output/Naruto_Shippuden_Opening_3_Blue_Bird.mid`
**Source:** YouTube https://www.youtube.com/watch?v=2upuBiEiXDk
**Assessment Goal:** Evaluate MIDI transcription quality for vocal melody extraction
**Ground Truth / Reference:** none
**Script:** `research/2026-04-18_blue-bird-vocals-midi-eval/evaluate.py`
**Command:** `python research/2026-04-18_blue-bird-vocals-midi-eval/evaluate.py Naruto_Shippuden_Opening_3_Blue_Bird.mid`

## Input Properties

| Property | Value | Provenance |
|----------|-------|------------|
| Duration | 88.450 s | [measured] |
| File size | 2,273 bytes | [measured] |
| Track count | 1 | [measured] |
| Note count | 252 | [measured] |
| Note density | 2.849 notes/s | [measured] |
| Tempo (embedded) | 120.00 BPM | [measured] |
| True song tempo | ~120–128 BPM | [assumed] — Blue Bird is commonly cited at ~124 BPM; pipeline used default 120 BPM due to missing bpm_detect |

## Measurements

### Polyphony
| Metric | Value | Provenance |
|--------|-------|------------|
| Simultaneous note events | 0 | [measured] |

A value of 0 confirms fully monophonic output — correct for a single vocal line.

### Pitch Distribution
| Metric | Value | Provenance |
|--------|-------|------------|
| Pitch minimum (MIDI) | 57 (A3) | [measured] |
| Pitch maximum (MIDI) | 82 (A#5) | [measured] |
| Pitch mean (MIDI) | 67.31 (G4) | [measured] |
| Pitch std dev (semitones) | 3.73 | [measured] |
| Unique pitches | 18 | [measured] |
| Notes outside vocal range C3–C6 | 0 (0.0%) | [measured] |
| Notes outside soprano range C4–A5 | 7 (2.8%) | [measured] |

All 252 notes fall within a plausible vocal range (A3–A#5). The 7 notes slightly above A5 (soprano ceiling) are minor outliers — consistent with a high-register J-pop vocalist.

### Note Duration Statistics
| Metric | Value | Provenance |
|--------|-------|------------|
| Min duration | 0.0698 s (≈ 32nd note @ 107 BPM) | [measured] |
| Max duration | 0.9698 s (≈ 2 beats @ 124 BPM) | [measured] |
| Mean duration | 0.1853 s | [measured] |
| Median duration | 0.1500 s | [measured] |
| Std deviation | 0.1280 s | [measured] |

The duration range is musically plausible. The short minimum (70 ms) is near the pipeline's stated 32nd-note floor at the default BPM.

### Quantization Regularity
| Metric | Value | Provenance |
|--------|-------|------------|
| Grid interval (32nd note @ 120 BPM) | 62.50 ms | [back-calc] 60/120/8 × 1000 |
| Mean onset deviation | 15.65 ms | [measured] |
| Max onset deviation | 30.21 ms | [measured] |
| Std onset deviation | 9.43 ms | [measured] |
| Onsets off-grid > 10 ms | 71.0% | [measured] |

71% of note onsets deviate more than 10 ms from the nearest 32nd-note grid position at 120 BPM. The maximum deviation is 30.21 ms — nearly half a grid cell. This is consistent with either (a) the pipeline's default BPM of 120 mismatching the true song tempo, or (b) RMVPE producing raw continuous-pitch detections that are then snapped to an incorrect grid.

### Melodic Intervals
| Metric | Value | Provenance |
|--------|-------|------------|
| Mean interval | 2.18 semitones | [measured] |
| Max interval | 18 semitones (1 octave + minor 6th) | [measured] |
| Leaps > 1 octave | 1 (0.4%) | [measured] |
| Unison repeats | 75 (29.8% of transitions) | [measured] |

The 2.18 semitone mean interval is reasonable for a stepwise vocal melody. The single large leap (18 semitones) is likely a pitch tracking error at a phrase boundary. 75 unison repeats (30% of note transitions) are higher than expected for a melodic vocal line — suggesting many continuous pitch segments are being segmented into repeated-same-pitch notes rather than held notes.

## Findings

| # | Severity | Finding | Measured Value | Threshold / Expected |
|---|----------|---------|----------------|----------------------|
| 1 | **Critical** | 71% of note onsets deviate >10 ms from 32nd-note grid due to incorrect BPM assumption | 71.0% off-grid; mean deviation 15.65 ms | < 20% off-grid for usable quantization |
| 2 | **Major** | 30% of note-to-note transitions are unison repeats, indicating oversegmentation of held notes | 75 unison repeats / 251 transitions (29.9%) | Expected < 10% for a natural vocal melody |
| 3 | **Major** | Only 18 unique pitches across 252 notes over 88 s — low melodic resolution | 18 unique pitches | Blue Bird has ~25–30 distinct pitches in the main melody [assumed] |
| 4 | **Minor** | 1 melodic leap of 18 semitones at a phrase boundary (likely pitch-tracking artifact) | 1 event (0.4%) | 0 leaps > 12 semitones expected |
| 5 | **Informational** | All notes within plausible vocal range A3–A#5 | 0 out-of-range notes | — |
| 6 | **Informational** | Fully monophonic output (0 polyphony events) — correct for vocal | 0 simultaneous events | — |

## Problem Statement

The pipeline transcribed 252 notes over 88 s with correct monophonic structure and plausible pitch range (A3–A#5), but quantization is severely degraded: 71% of onsets deviate >10 ms from the 32nd-note grid because the default 120 BPM assumption does not match the song's actual tempo (est. ~124 BPM). Additionally, 30% of note transitions are same-pitch repeats, indicating that long held notes are being oversegmented into short repeated fragments rather than encoded as single sustained notes, reducing melodic coherence.

## Suggested Success Criteria

| Finding | Metric | Target |
|---------|--------|--------|
| BPM mismatch / quantization | % onsets off 32nd-note grid > 10 ms | < 20% |
| Oversegmentation of held notes | % unison note transitions | < 10% |
| Melodic pitch vocabulary | Unique pitches over full transcription | ≥ 25 |
| Spurious large leaps | Leaps > 12 semitones | 0 |

## Recommended Next Step

Invoke **project-planner** with the problem statement above. Key inputs:
1. BPM auto-detection is broken (bpm_detect not on PATH) — default 120 BPM causes systematic quantization error.
2. Note segmentation logic oversegments sustained pitches into unison repeats — likely a minimum-duration threshold or onset detection issue in `postprocess.py` / `transcribers/`.
