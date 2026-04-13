# Research Notes: Demucs htdemucs_6s Stem Quality Evaluation

**Date:** 2026-04-13
**Session:** `research/2026-04-13_demucs-quality-eval/`
**Goal:** Measure blind stem separation quality (leakage, bleed, silence) for
Demucs `htdemucs_6s` on a real-world rock track, using `tests/evaluation/evaluate_stems.py`.

---

## Environment

| Field | Value | Provenance |
|-------|-------|------------|
| Model | htdemucs_6s | [source-code — demucs CLI `--name htdemucs_6s`] |
| Demucs version | 4.0.1 | [measured — `importlib.metadata.version('demucs')`] |
| Device used | MPS (Apple M1) | [measured — benchmark session `research/2026-04-12_demucs-benchmark/`] |
| Evaluator script | `tests/evaluation/evaluate_stems.py` | [source-code] |
| Evaluator command | `python3 tests/evaluation/evaluate_stems.py --mix "inputs/Foals - My Number (Official Audio).mp3" --stems-dir "output/2026-04-13_foals-my-number-stems/htdemucs_6s/Foals - My Number (Official Audio)" --json-out "output/2026-04-13_foals-my-number-stems/evaluation.json"` | [source-code] |
| JSON output | `output/2026-04-13_foals-my-number-stems/evaluation.json` | [measured] |

---

## Input

| Field | Value | Provenance |
|-------|-------|------------|
| File | `inputs/Foals - My Number (Official Audio).mp3` | [measured] |
| Duration | 242.58 s (4.04 min) | [measured — librosa load] |
| Genre | Rock (full band: vocals, guitar, bass, drums) | [assumed — subjective characterisation] |
| Ground truth stems | None available | — |

---

## Reconstruction Metrics

Measures how faithfully the 6 stems sum back to the original mix.

| Metric | Value | Provenance |
|--------|-------|------------|
| Mix energy | 57.07 dB | [measured — `evaluate_stems.py`] |
| Stem-sum energy | 56.56 dB | [measured — `evaluate_stems.py`] |
| Residual energy | 36.18 dB | [measured — `evaluate_stems.py`] |
| Leakage ratio | **−20.90 dB** | [measured — `evaluate_stems.py`; formula: residual_energy_db − mix_energy_db] |

A leakage ratio of −20.90 dB means the reconstruction residual carries
~1% of the mix energy — good overall reconstruction fidelity.

---

## Per-Stem Quality Metrics

### Key: metric definitions
- **RMS (dBFS)** — loudness of the stem
- **SRR (dB)** — stem-to-residual ratio; higher = cleaner isolation
- **Centroid (Hz)** — mean spectral centroid; tonal character sanity check
- **Silence (%)** — fraction of frames below −60 dBFS; high = over-suppressed or absent
- **Cross-leakage (dB)** — spectral similarity to each other stem; closer to −∞ = less bleed

### Results table

| Stem | RMS (dBFS) | SRR (dB) | Centroid (Hz) | Silence (%) | Worst cross-leakage | vs |
|------|-----------|---------|--------------|------------|--------------------|----|
| drums | −17.51 | **+16.61** | 4707.7 | 7.4% | −6.40 dB | bass |
| bass | −20.89 | +13.23 | 1738.6 | 40.9% | −6.40 dB | drums |
| vocals | −22.12 | +12.00 | 2296.9 | 14.2% | −4.48 dB | other |
| guitar | −23.54 | +10.57 | 1449.9 | 2.0% | **−3.52 dB** | other |
| other | −32.29 | +1.83 | 2135.6 | 34.3% | **−3.52 dB** | guitar |
| piano | −69.17 | **−35.05** | 4051.0 | **99.8%** | — | — |

All values [measured — `evaluate_stems.py` output from `evaluation.json`].

### Full cross-leakage matrix (dB)

Lower (more negative) = less spectral bleed between stems.

|  | vocals | drums | bass | guitar | piano | other |
|--|--------|-------|------|--------|-------|-------|
| **vocals** | — | −17.87 | −19.32 | −4.65 | −13.93 | −4.48 |
| **drums** | −17.87 | — | −6.40 | −14.70 | −22.74 | −19.08 |
| **bass** | −19.32 | −6.40 | — | −12.16 | −23.86 | −19.08 |
| **guitar** | −4.65 | −14.70 | −12.16 | — | −13.05 | −3.52 |
| **piano** | −13.93 | −22.74 | −23.86 | −13.05 | — | −13.52 |
| **other** | −4.48 | −19.08 | −19.08 | −3.52 | −13.52 | — |

All values [measured — `evaluate_stems.py`].

---

## Findings

| # | Severity | Finding | Measured value | Threshold / expected |
|---|----------|---------|---------------|----------------------|
| 1 | **Critical** | `piano` stem is effectively silent — model found no piano content | SRR −35.05 dB, silence 99.8% | Expected: SRR > 0 dB if piano present |
| 2 | **Major** | Heavy spectral bleed between `guitar` and `other` | Cross-leakage −3.52 dB (≈67% shared spectral energy) | Target: < −15 dB for usable isolation |
| 3 | **Major** | Heavy spectral bleed between `vocals` and `other` | Cross-leakage −4.48 dB | Target: < −15 dB |
| 4 | **Major** | Moderate spectral bleed between `vocals` and `guitar` | Cross-leakage −4.65 dB | Target: < −15 dB |
| 5 | **Major** | `other` stem has very low SRR (+1.83 dB) — barely above residual noise | SRR +1.83 dB | Target: > +10 dB |
| 6 | Minor | `bass` silence fraction is high (40.9%) — model over-suppresses during quiet sections | 40.9% silent frames | Informational |
| 7 | Minor | `drums`/`bass` cross-leakage is −6.40 dB — moderate bleed between low-frequency stems | −6.40 dB | Informational |

---

## Problem Statement

Demucs `htdemucs_6s` produces acceptable reconstruction fidelity (−20.9 dB
leakage ratio) but exhibits significant per-stem spectral bleed for the
instrument classes most relevant to this pipeline. The `guitar` stem bleeds
heavily into `other` (−3.52 dB cross-leakage), `vocals` bleeds into `other`
and `guitar` (−4.48 to −4.65 dB), and the `other` SRR is only +1.83 dB —
confirming the subjective report of background instruments audible in
targeted stems. The `piano` stem is empty on this track (correct, but means
htdemucs_6s piano isolation is untested on a track with actual piano). These
bleed levels will degrade downstream transcription quality for guitar and
other harmonic content.

---

## Recommended Next Step

Invoke **project-planner** with: "Evaluate alternative separation models or
post-processing approaches (e.g. htdemucs 4-stem + residual subtraction,
Open-Unmix, or Demucs fine-tuned variants) targeting guitar/other
cross-leakage < −15 dB and guitar SRR > +15 dB on the Foals benchmark track.
Current htdemucs_6s baseline: guitar SRR +10.57 dB, guitar/other
cross-leakage −3.52 dB."
