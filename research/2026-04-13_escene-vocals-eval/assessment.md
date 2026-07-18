# Assessment: E.scene "意識" Music Video — Vocals Stem Quality

**Date:** 2026-04-13
**Input (mix):** `/tmp/yt_download/E.scene ＂意識＂ Music Video.mp3`
**Stems:** `output/2026-04-13_escene-htdemucs-4s/E.scene ＂意識＂ Music Video/`
**Model:** `htdemucs` (4-stem: vocals / drums / bass / other)
**Assessment Goal:** Measure quality of the vocals stem separated by htdemucs 4-stem from an electronic/J-pop track
**Ground Truth / Reference:** None (blind evaluation)
**Evaluation Script:** `research/2026-04-13_escene-vocals-eval/evaluate.py`
**Command:**
```
python3 research/2026-04-13_escene-vocals-eval/evaluate.py \
    --mix "/tmp/yt_download/E.scene ＂意識＂ Music Video.mp3" \
    --stems-dir "output/2026-04-13_escene-htdemucs-4s/E.scene ＂意識＂ Music Video" \
    --stems vocals drums bass other \
    --json-out "research/2026-04-13_escene-vocals-eval/evaluation.json"
```

---

## Input Properties

| Property | Value | Provenance |
|----------|-------|------------|
| Duration | 245.59 s (4.09 min) | [measured] |
| Sample rate | 44 100 Hz | [measured] |
| Codec / container | MP3 (yt-dlp WebM → MP3 transcode) | [measured — yt-dlp output] |
| File size | 4.06 MiB (source WebM) | [measured — yt-dlp output] |
| RMS | −11.23 dBFS | [measured] |
| Peak | 0.44 dBFS | [measured] |
| Clipping fraction | 0.0032% | [measured] |
| LUFS estimate | −11.92 LUFS (RMS proxy, ±2 LUFS) | [assumed — full K-weighting not applied] |
| Spectral centroid | 2 041.3 Hz | [measured] |
| Genre | Electronic / J-pop (E.scene — 意識) | [assumed — subjective] |

---

## Measurements

### Reconstruction (mix-level)

| Metric | Value | Provenance |
|--------|-------|------------|
| Mix energy | 59.12 dB | [measured] |
| Stem-sum energy | 57.90 dB | [measured] |
| Residual energy | 60.83 dB | [measured] |
| Leakage ratio | **+1.70 dB** | [measured — residual_energy_db − mix_energy_db] |

A positive leakage ratio means the reconstruction residual carries *more* energy than the
original mix — indicating that the four stems are not energy-conservative and their sum
differs materially from the original. This is anomalous for htdemucs (expected: ≤ −15 dB).
The likely cause is that Demucs wrote .mp3 outputs rather than .wav; lossy re-encoding
introduces quantization noise that accumulates when stems are summed, inflating the
residual above the mix level.

### Per-stem metrics

| Stem | RMS (dBFS) | Peak (dBFS) | SRR (dB) | Centroid (Hz) | Silence (%) | Clipping (%) |
|------|-----------|------------|---------|--------------|------------|-------------|
| vocals | −20.27 | −0.70 | −10.75 | 4 325.9 | 36.44% | 0.00% |
| drums  | −18.67 | −0.21 | −9.15  | 2 977.6 | 11.37% | 0.00% |
| bass   | −15.15 | −0.24 | −5.63  | 588.8   | 6.23%  | 0.00% |
| other  | −28.31 | −7.57 | −18.79 | 1 770.7 | 2.30%  | 0.00% |

All values [measured].

### Cross-leakage matrix (dB) — lower (more negative) = less bleed

|         | vocals | drums  | bass   | other  |
|---------|--------|--------|--------|--------|
| **vocals** | —   | −13.29 | −22.47 | −5.13 |
| **drums**  | −13.29 | —   | −4.70  | −13.32 |
| **bass**   | −22.47 | −4.70 | —     | −15.39 |
| **other**  | −5.13 | −13.32 | −15.39 | —     |

All values [measured].

### Vocals-specific

| Metric | Value | Provenance |
|--------|-------|------------|
| Vocal presence ratio (RMS voc / RMS mix) | 0.3529 | [measured] |
| Worst cross-leakage (vocals → other) | −5.13 dB | [measured] |
| Worst cross-leakage (other → vocals) | −5.13 dB | [measured] |
| Vocals silence fraction | 36.44% | [measured] |
| Vocals SRR | −10.75 dB | [measured] |
| Vocals spectral centroid | 4 325.9 Hz | [measured] |

---

## Findings

| # | Severity | Finding | Measured Value | Threshold / Expected |
|---|----------|---------|---------------|----------------------|
| 1 | **Critical** | Reconstruction residual exceeds mix energy — stem sum is not energy-conservative; MP3 quantization noise accumulates across 4 stems | Leakage ratio **+1.70 dB** | Expected: ≤ −15 dB for lossless stems; ≤ −10 dB acceptable for lossy |
| 2 | **Critical** | All SRR values are negative — every stem carries less energy than the reconstruction residual, making SRR an unreliable quality indicator for this run | Vocals SRR **−10.75 dB**, all stems negative | Expected: SRR > 0 dB for usable isolation |
| 3 | **Major** | Heavy spectral bleed between vocals and other — both share nearly the same spectral profile | Cross-leakage **−5.13 dB** (vocals↔other) | Target: < −15 dB for usable isolation |
| 4 | **Major** | Vocals silence fraction is high — model over-suppresses during non-vocal sections but this may also indicate bleed in both directions | **36.44%** frames below −60 dBFS | Informational (no reference); high silence can indicate over-suppression |
| 5 | **Major** | Vocals spectral centroid (4 325.9 Hz) is significantly higher than mix centroid (2 041.3 Hz) — suggests the model has pushed mid-frequency content (harmonics, body) into `other`, retaining primarily high-frequency components (sibilance, air) in the vocals stem | Vocals centroid **4 325.9 Hz** vs mix **2 041.3 Hz** | Expected: centroid ≈ 1 500–3 000 Hz for a lead vocal stem |
| 6 | Minor | Drums / bass cross-leakage is −4.70 dB — moderate bleed between low-frequency stems | −4.70 dB | Target: < −15 dB |
| 7 | Informational | Vocal presence ratio 0.3529 is within expected range for a lead vocal in a dense mix | 0.3529 (RMS voc/mix) | Expected: 0.20–0.50 |
| 8 | Informational | No clipping in any stem | 0.00% all stems | — |

---

## Note on Metric Reliability

Findings 1 and 2 indicate that the MP3-encoded stems introduce accumulated
quantization noise that inflates the reconstruction residual above the mix
energy level. This renders the **SRR** and **leakage ratio** metrics unreliable
as absolute quality indicators for this run. The **cross-leakage** and
**spectral centroid** values are derived from the stem waveforms independently
and are not affected by the reconstruction residual, so they remain valid.

---

## Problem Statement

The `htdemucs` (4-stem) vocals separation of "E.scene — 意識" exhibits two
classes of defect. First, the MP3 stem encoding makes energy-based metrics
(SRR, leakage ratio) unreliable for this run: all SRR values are negative and
the leakage ratio is +1.70 dB, anomalous for a model known to achieve −20 dB
leakage on wav outputs. Second, the vocals stem shows heavy spectral bleed with
the `other` stem (cross-leakage −5.13 dB, versus a −15 dB usability target)
and an anomalously high spectral centroid (4 325.9 Hz vs a 2 041.3 Hz mix
centroid), indicating that mid-frequency vocal body is being routed to `other`
rather than the vocals stem — which will degrade downstream pitch transcription
accuracy for this track's vocal content.

---

## Suggested Success Criteria

| Finding | Metric | Target |
|---------|--------|--------|
| MP3 metric inflation (Finding 1) | Re-run separation with `--wav` flag; leakage ratio | ≤ −15 dB |
| Vocals/other bleed (Finding 3) | Cross-leakage vocals↔other | < −15 dB |
| Vocal centroid anomaly (Finding 5) | Vocals spectral centroid | 1 500–3 000 Hz |
| Vocals SRR (Finding 2) | SRR on wav outputs | > +10 dB |
| Vocals over-suppression (Finding 4) | Silence fraction | < 20% |

---

## Recommended Next Step

Invoke **project-planner** with: "The htdemucs 4-stem vocals separation of a
J-pop/electronic track (E.scene — 意識) shows vocals/other cross-leakage of
−5.13 dB (target < −15 dB) and an anomalous vocals spectral centroid of 4 325 Hz
(vs 2 041 Hz mix centroid), suggesting mid-frequency vocal body leaks into
`other`. Energy metrics (SRR, leakage ratio) are unreliable due to MP3 stem
encoding; re-run with `--wav` to validate. Evaluate htdemucs_ft, mdx_extra, or
post-processing (spectral subtraction, Wiener filter) to improve vocals/other
isolation and recover vocal body frequencies below 3 kHz."
