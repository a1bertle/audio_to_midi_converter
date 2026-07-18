# Assessment: Adounravel_歌いました (MBR pipeline)

**Date:** 2026-04-27
**Session:** `research/2026-04-27_adounravel-mbr-vocal-midi-eval/`
**Input workdir:** `outputs/2026-04-27_22-03-59_Adounravel_歌いました/`
**Assessment Goal:** Vocal MIDI transcription quality after replacing HTDemucs with
Mel-Band RoFormer (MelBandRoformerSYHFTV3Epsilon.ckpt) — pitch accuracy, coverage,
hallucination, and note structure.
**Ground Truth / Reference:** pYIN F0 contour extracted from MBR vocals stem
(same stem used as RMVPE input); original mix `extracted/eZuDklxsaRg.wav`
**Evaluation Script:** `research/2026-04-27_adounravel-mbr-vocal-midi-eval/evaluate.py`
**Command:**
```
python research/2026-04-27_adounravel-mbr-vocal-midi-eval/evaluate.py \
  --workdir outputs/2026-04-27_22-03-59_Adounravel_歌いました
```

---

## Input Properties

| Property | Value | Provenance |
|----------|-------|------------|
| MIDI file | `Adounravel_歌いました.mid` | [measured] |
| MIDI duration | 229.53 s | [measured] |
| Mix duration | 240.37 s | [measured] |
| Duration mismatch (MIDI vs audio) | 10.84 s | [measured] |
| MIDI instruments | 1 | [measured] |
| Total MIDI notes | 512 | [measured] |
| MIDI pitch range | E3–A5 (MIDI 52–81), 29 semitones | [measured] |
| MIDI pitch median | 70 (A#4) | [measured] |
| Tempo | 135.29 BPM | [measured] |
| Stem separator | MelBandRoformerSYHFTV3Epsilon.ckpt (audio-separator 0.44.1) | [source-code] |
| Vocals stem sample rate | 44100 Hz | [measured] |
| Vocals stem peak | −5.09 dBFS | [measured] |
| Mix peak | +4.35 dBFS (clipped) | [measured] |

---

## Measurements

### Vocals Stem Audio Properties

| Metric | Value | Provenance |
|--------|-------|------------|
| Duration | 240.373 s | [measured] |
| Peak | −5.09 dBFS | [measured] |
| RMS | −21.79 dBFS | [measured] |
| Spectral centroid | 3675.9 Hz | [measured] |
| Mix spectral centroid | 3684.9 Hz | [measured] |
| Centroid delta (stem vs mix) | −9.0 Hz | [back-calc: 3675.9 − 3684.9] |
| Silence fraction | 31.5% | [measured] |

> Note: centroid 3675.9 Hz vs mix 3684.9 Hz — near-identical, confirming MBR
> captures the full vocal spectral body rather than leaking fundamentals to
> "other" (HTDemucs baseline centroid was 4427 Hz vs mix ~2041 Hz [measured —
> `research/2026-04-13_escene-vocals-eval`]).

### Vocal Activity

| Metric | Value | Provenance |
|--------|-------|------------|
| Total frames | 20,705 | [measured] |
| Voiced frames (RMS −40 dBFS) | 13,720 | [measured] |
| Voiced ratio (RMS) | 66.3% | [measured] |
| pYIN voiced frames | 10,811 | [measured] |
| pYIN voiced ratio | 52.2% | [measured] |
| RMS–pYIN gap | 14.1 pp | [back-calc: 66.3 − 52.2] |

### pYIN F0 Extraction

| Metric | Value | Provenance |
|--------|-------|------------|
| F0 mean | 552.9 Hz (MIDI 71.2) | [measured] |
| F0 median | 468.9 Hz (MIDI 70.1) | [measured] |
| F0 MIDI range | 40.5–95.6, 55.1 semitones | [measured] |

### MIDI Note Statistics

| Metric | Value | Provenance |
|--------|-------|------------|
| Note density | 2.23 notes/s | [back-calc: 512 / 229.53] |
| Mean note duration | 189 ms | [measured] |
| Median note duration | 150 ms | [measured] |
| Std note duration | 136 ms | [measured] |
| Min note duration | 59 ms | [measured] |
| Max note duration | 890 ms | [measured] |
| Velocity mean | 80 (constant) | [measured] |
| Velocity std | 0.0 | [measured] |

### Pitch Coverage (pYIN F0 vs MIDI note roll, frame-level)

| Metric | Value | Provenance |
|--------|-------|------------|
| pYIN voiced frames (trimmed to MIDI length) | 10,762 | [measured] |
| MIDI active frames | 8,351 | [measured] |
| Co-active frames | 7,415 | [measured] |
| Coverage ratio | 68.9% | [measured] |
| Hallucination ratio | 11.2% | [measured] |
| Mean pitch error (co-active) | 0.61 semitones | [measured] |
| Median pitch error | 0.20 semitones | [measured] |
| 90th-percentile pitch error | 0.70 semitones | [measured] |
| Frames within ±0.5 semitones | 80.2% | [measured] |
| Frames within ±1.0 semitone | 93.7% | [measured] |

### Pitch Histogram Overlap

| Metric | Value | Provenance |
|--------|-------|------------|
| Histogram intersection (MIDI vs pYIN F0) | 0.782 | [measured] |

### Inter-Onset Interval (IOI) Regularity

| Metric | Value | Provenance |
|--------|-------|------------|
| IOI mean | 444 ms | [measured] |
| IOI std | 962 ms | [measured] |
| IOI coefficient of variation | 2.16 | [measured] |
| IOI min | 59 ms | [measured] |
| IOI max | 14,381 ms | [measured] |

---

## Comparison vs HTDemucs Baseline

Baseline values from `research/2026-04-26_adounravel-vocal-midi-eval/assessment.md` [measured].

| Metric | HTDemucs baseline | MBR (this run) | Δ | Target |
|--------|------------------|----------------|---|--------|
| Hallucination ratio | 13.3% | **11.2%** | −2.1 pp | ≤ 5% |
| Coverage ratio | 71.2% | **68.9%** | −2.3 pp | ≥ 90% |
| Pitch accuracy (±0.5 st) | 80.3% | **80.2%** | −0.1 pp | ≥ 85% |
| Pitch accuracy (±1.0 st) | 94.7% | **93.7%** | −1.0 pp | — |
| Median note duration | 150 ms | **150 ms** | 0 ms | ≥ 300 ms |
| IOI CV | 2.84 | **2.16** | −0.68 | ≤ 1.0 |
| Histogram intersection | 0.810 | **0.782** | −0.028 | — |
| Vocals centroid | 4427 Hz | **3675.9 Hz** | −751 Hz | ≈ mix |
| Mix centroid | ~2041 Hz | **3684.9 Hz** | — | — |

All delta values [back-calc from measured columns].

---

## Findings

| # | Severity | Finding | Measured Value | Threshold / Expected |
|---|----------|---------|----------------|----------------------|
| 1 | **Major** | 31.1% of pYIN-voiced frames have no active MIDI note (coverage gap) | Coverage ratio 68.9% | ≥ 90% [assumed] |
| 2 | **Major** | Notes are highly fragmented — median 150 ms, 2.23 notes/s, IOI CV 2.16 | Median 150 ms; IOI CV 2.16 | Median ≥ 300 ms; IOI CV ≤ 1.0 [assumed] |
| 3 | **Major** | 11.2% of MIDI-active frames fire on pYIN-unvoiced passages | Hallucination ratio 11.2% | ≤ 5% [assumed] |
| 4 | Minor | Pitch accuracy on covered frames is below target on ±0.5 st metric | 80.2% within ±0.5 st | ≥ 85% [assumed] |
| 5 | Minor | MIDI ends 10.84 s before audio — final phrase likely truncated | Duration mismatch 10.84 s | ≤ 1.0 s [assumed] |
| 6 | Minor | IOI CV improved (2.84 → 2.16) but still well above target | IOI CV 2.16 | ≤ 1.0 [assumed] |
| 7 | Informational | Vocals spectral centroid (3675.9 Hz) now matches mix (3684.9 Hz) — MBR bleed fix confirmed | Δ centroid −9 Hz vs mix | ≈ 0 Hz [assumed] |
| 8 | Informational | Velocity is constant at 80 across all 512 notes — no dynamics encoded | Velocity std 0.0 | — |
| 9 | Informational | Mix WAV is clipped (+4.35 dBFS peak) | Peak +4.35 dBFS | ≤ 0 dBFS |

---

## Problem Statement

The vocal MIDI transcription covers only 68.9% of pYIN-voiced frames and produces
highly fragmented notes (median 150 ms, IOI CV 2.16) that do not reflect
phrase-level vocal structure. 11.2% of MIDI-active frames fire on pYIN-unvoiced
passages (hallucination). Pitch accuracy on covered frames is near target (80.2%
within ±0.5 semitones, 93.7% within ±1.0 semitone), so the primary defects remain
incomplete coverage, note fragmentation, and excess hallucination — unchanged from
the HTDemucs baseline despite the improved stem quality.

---

## Suggested Success Criteria

| Finding | Metric | Target |
|---------|--------|--------|
| Coverage gap | Coverage ratio | ≥ 90% |
| Note fragmentation | Median note duration | ≥ 300 ms |
| Note fragmentation | IOI CV | ≤ 1.0 |
| Hallucination | Hallucination ratio | ≤ 5% |
| Pitch accuracy | Frames within ±0.5 semitones | ≥ 85% |
| Duration truncation | Duration mismatch | ≤ 1.0 s |

---

## Recommended Next Step

The coverage, fragmentation, and hallucination defects are present at nearly the
same level as the HTDemucs baseline, despite the MBR stem having correct spectral
balance (centroid 3676 Hz ≈ mix 3685 Hz). This confirms the defects originate in
the **RMVPE → note segmentation** layer, not in stem separation quality. Invoke
**project-planner** with this problem statement, focusing on: (1) reducing note
fragmentation via minimum duration gating and note merging, (2) increasing coverage
by lowering onset detection thresholds or extending note boundaries into voiced gaps,
and (3) suppressing hallucination via a voiced/unvoiced confidence gate on RMVPE output.
