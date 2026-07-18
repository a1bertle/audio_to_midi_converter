# Assessment: Adounravel_歌いました.mid

**Date:** 2026-04-26
**Input:** `outputs/2026-04-26_12-34-26_Adounravel_歌いました/Adounravel_歌いました.mid`
**Assessment Goal:** Vocal MIDI transcription quality — pitch accuracy, coverage, and note structure
**Ground Truth / Reference:** `outputs/2026-04-26_12-34-26_Adounravel_歌いました/stems/htdemucs/eZuDklxsaRg/vocals.wav` (Demucs vocal stem); `outputs/2026-04-26_12-34-26_Adounravel_歌いました/extracted/eZuDklxsaRg.wav` (original mix)
**Evaluation Script:** `scripts/eval_vocal_midi.py` (canonical location)
**Command:**
```
python scripts/eval_vocal_midi.py \
  "outputs/2026-04-26_12-34-26_Adounravel_歌いました/Adounravel_歌いました.mid" \
  "outputs/2026-04-26_12-34-26_Adounravel_歌いました/stems/htdemucs/eZuDklxsaRg/vocals.wav" \
  "outputs/2026-04-26_12-34-26_Adounravel_歌いました/extracted/eZuDklxsaRg.wav"
```

---

## Input Properties

| Property | Value | Provenance |
|----------|-------|------------|
| MIDI duration | 229.61 s | [measured] |
| Vocals audio duration | 240.37 s | [measured] |
| Mix audio duration | 240.37 s | [measured] |
| Duration mismatch (MIDI vs audio) | 10.76 s | [measured] |
| MIDI instruments | 1 | [measured] |
| Total MIDI notes | 520 | [measured] |
| MIDI pitch range | A2–A5 (MIDI 45–81), 36 semitones | [measured] |
| MIDI pitch median | 70 (A#4) | [measured] |
| Tempo | 135.29 BPM | [measured] |
| Vocals sample rate | 44100 Hz | [measured] |
| Vocals peak | −4.64 dBFS | [measured] |
| Mix peak | +4.35 dBFS (clipped) | [measured] |

---

## Measurements

### Vocal Activity (RMS threshold −40 dBFS)

| Metric | Value | Provenance |
|--------|-------|------------|
| Total analysis frames | 20,705 | [measured] |
| Voiced frames (RMS) | 13,637 | [measured] |
| Voiced ratio (RMS) | 65.9% | [measured] |
| RMS mean | −28.97 dBFS | [measured] |

### pYIN F0 Extraction (vocals stem, hop=512)

| Metric | Value | Provenance |
|--------|-------|------------|
| pYIN voiced frames | 10,218 | [measured] |
| pYIN voiced ratio | 49.4% | [measured] |
| F0 mean | 533.9 Hz (MIDI 70.6) | [measured] |
| F0 median | 466.2 Hz (MIDI 70.0) | [measured] |
| F0 MIDI range | 36.0–95.6, 59.6 semitones | [measured] |

> Note: RMS marks 65.9% of frames as active but pYIN only voices 49.4% — a 16.5 pp gap [back-calc: 65.9 − 49.4] attributable to breaths, soft onsets, and vibrato tails falling below pYIN confidence threshold.

### MIDI Note Statistics

| Metric | Value | Provenance |
|--------|-------|------------|
| Note density | 2.27 notes/s | [back-calc: 520 / 229.61] |
| Mean note duration | 182 ms | [measured] |
| Median note duration | 150 ms | [measured] |
| Std note duration | 127 ms | [measured] |
| Min note duration | 59 ms | [measured] |
| Max note duration | 910 ms | [measured] |
| Velocity mean | 80 (constant) | [measured] |
| Velocity std | 0.0 | [measured] |

### Pitch Coverage (pYIN F0 vs MIDI note roll, frame-level)

| Metric | Value | Provenance |
|--------|-------|------------|
| pYIN voiced frames (trimmed to MIDI length) | 9,912 | [measured] |
| MIDI active frames | 8,136 | [measured] |
| Co-active frames (both voiced and MIDI-on) | 7,056 | [measured] |
| Coverage ratio (co-active / pYIN voiced) | 71.2% | [measured] |
| Hallucination ratio (MIDI-on / pYIN unvoiced) | 13.3% | [measured] |
| Mean pitch error (co-active frames) | 0.62 semitones | [measured] |
| Median pitch error | 0.20 semitones | [measured] |
| 90th-percentile pitch error | 0.70 semitones | [measured] |
| Frames within ±0.5 semitones | 80.3% | [measured] |
| Frames within ±1.0 semitone | 94.7% | [measured] |

### Pitch Histogram Overlap

| Metric | Value | Provenance |
|--------|-------|------------|
| Histogram intersection (MIDI vs pYIN F0) | 0.810 | [measured] |

### Inter-Onset Interval (IOI) Regularity

| Metric | Value | Provenance |
|--------|-------|------------|
| IOI mean | 438 ms | [measured] |
| IOI std | 1,245 ms | [measured] |
| IOI coefficient of variation | 2.84 | [measured] |
| IOI min | 60 ms | [measured] |
| IOI max | 21,080 ms | [measured] |

---

## Findings

| # | Severity | Finding | Measured Value | Threshold / Expected |
|---|----------|---------|----------------|----------------------|
| 1 | Major | 28.8% of pYIN-voiced frames have no active MIDI note (coverage gap) | Coverage ratio 71.2% | ≥ 90% for usable melody MIDI [assumed] |
| 2 | Major | Notes are highly fragmented — median 150 ms, 2.27 notes/s, IOI CV 2.84 | Median duration 150 ms; IOI CV 2.84 | Phrase-level notes typically 300–800 ms; IOI CV < 1.0 [assumed] |
| 3 | Minor | 13.3% of MIDI-active frames fire on pYIN-unvoiced passages (hallucination) | Hallucination ratio 13.3% | < 5% [assumed] |
| 4 | Minor | MIDI ends 10.76 s before audio — final phrase likely truncated | Duration mismatch 10.76 s | 0 s |
| 5 | Informational | Pitch accuracy on covered frames is good: 80.3% within ±0.5 st, 94.7% within ±1.0 st | Mean error 0.62 st | — |
| 6 | Informational | Velocity is constant at 80 across all 520 notes — no dynamics encoded | Velocity std 0.0 | — |
| 7 | Informational | Mix WAV is clipped (+4.35 dBFS peak) — not a MIDI quality issue but may affect upstream processing | Peak +4.35 dBFS | ≤ 0 dBFS |

---

## Problem Statement

The vocal MIDI transcription covers only 71.2% of pYIN-voiced frames and produces highly fragmented notes (median 150 ms, 2.27 notes/s, IOI CV 2.84) that do not reflect phrase-level vocal structure; approximately 13.3% of MIDI notes fire on passages where pYIN detects no voicing, indicating bleed or onset over-segmentation. Pitch accuracy on covered frames is acceptable (80.3% within ±0.5 semitones, 94.7% within ±1.0 semitone), so the primary defects are incomplete coverage and note fragmentation rather than tuning error.

---

## Suggested Success Criteria

| Finding | Metric | Target |
|---------|--------|--------|
| Coverage gap | Coverage ratio (co-active / pYIN voiced) | ≥ 90% |
| Note fragmentation | Median note duration | ≥ 300 ms |
| Note fragmentation | IOI coefficient of variation | ≤ 1.0 |
| Hallucination | Hallucination ratio | ≤ 5% |
| Duration truncation | Duration mismatch | ≤ 1.0 s |
| Pitch accuracy | Frames within ±0.5 semitones | ≥ 85% |

---

## Recommended Next Step

Invoke **project-planner** with the problem statement above. Key constraints: pitch accuracy is already within target on covered frames, so improvements should focus on (1) reducing note fragmentation via note merging / minimum duration gating, and (2) increasing coverage by lowering onset detection thresholds or extending note boundaries to fill voiced gaps.
