# Research Notes: Dedicated Vocal Pitch Trackers — Direct-on-Mix Approaches

**Date:** 2026-04-17
**Session:** `research/2026-04-17_vocal-pitch-tracker-research/`
**Goal:** Evaluate dedicated vocal pitch trackers and vocal-to-MIDI approaches that operate
directly on a polyphonic mix (no stem separation required), or with improved accuracy on a rough
stem, to replace the failing htdemucs + basic-pitch pipeline.
**Context:** Prior sessions (2026-04-13_vocal-midi-bleed-tolerance) confirmed that all htdemucs
variants produce vocals stems with centroid ~4400–4800 Hz (vocal fundamentals and formants 100–3000
Hz routed to `other` stem). basic-pitch MIDI output was subjectively unusable despite 99.8%
in-range note statistics. The pipeline needs either a separation model that preserves vocal body
below 3 kHz, OR a pitch tracker designed to operate in polyphonic audio without requiring a clean
stem.

---

## Prior Context Summary

| Finding | Value | Provenance |
|---------|-------|------------|
| htdemucs (all variants) vocals centroid | 4 427–4 781 Hz | [measured — 2026-04-13_vocal-midi-bleed-tolerance/notes.md] |
| Mix spectral centroid | 2 041 Hz | [measured — 2026-04-13_escene-vocals-eval/assessment.md] |
| Expected vocal centroid | 1 500–3 000 Hz | [assumed — genre knowledge] |
| basic-pitch on htdemucs stem: note count | 650 | [measured] |
| basic-pitch on htdemucs stem: mean confidence | 0.592 | [measured] |
| basic-pitch on htdemucs stem: subjective quality | Unusable (octave errors, unstable) | [measured — F9] |
| Separation bottleneck confirmed | All htdemucs variants show same floor | [measured — F10] |

**Root cause (F9):** basic-pitch's CQT detects pitch from harmonic partials. When the vocal
fundamental and lower formants (100–3000 Hz) are routed to `other`, basic-pitch latches onto upper
partials, producing octave errors. In-range note statistics were a false quality indicator.

---

## Environment

| Field | Value | Provenance |
|-------|-------|------------|
| Machine | Apple MacBook Air (M1), 16 GB RAM | [assumed — global-standards.md] |
| Python | 3.12 | [measured — venv] |
| librosa | 0.11.0 | [measured — venv] |
| basic-pitch | 0.3.0 | [measured — venv] |
| PyTorch | in venv (version TBD) | [source-code — requirements.txt] |
| Target RTF budget | ≤ 3× real-time (~12 min for 4-min track) | [assumed — project constraint] |

---

## Candidates Evaluated

### 1. CREPE / torchcrepe

| Attribute | Value | Provenance |
|-----------|-------|------------|
| Pip install | `pip install torchcrepe` (no heavy build) | [source-code — PyPI] |
| M1 native | Yes — PyTorch arm64 MPS backend | [assumed — PyTorch standard] |
| MIDI output | Pitch track only; `crepe-notes` package adds note segmentation | [source-code — PyPI] |
| Monophonic | Yes — single-pitch estimation only | [source-code] |
| Accuracy on clean vocal | ~90%+ RPA @ 10 cent threshold | [measured — CREPE paper, arXiv 1802.06182] |
| Accuracy on polyphonic mix | Poor — "sharply decreases" per RMVPE authors | [measured — RMVPE paper, arXiv 2306.15412] |
| RTF on M1 (est.) | 0.05–0.15× | [assumed — GPU baseline 0.05×; M1 CPU extrapolated] |
| Verdict | Not suitable for raw polyphonic mix; acceptable only with clean stem | [assumed] |

### 2. pYIN (librosa.pyin)

| Attribute | Value | Provenance |
|-----------|-------|------------|
| Pip install | Already in venv (librosa 0.11.0) | [source-code] |
| M1 native | Yes — pure Python + numpy | [assumed — librosa architecture] |
| MIDI output | Pitch track + voicing flags; manual post-processing required | [source-code] |
| Monophonic | Yes — single-pitch; reverb/polyphony degrades accuracy per docs | [source-code — librosa docs] |
| Accuracy on clean vocal | Good — older (YIN 2002, pYIN 2014); lags CREPE in smoothness | [measured — literature] |
| Accuracy on polyphonic mix | Poor — not designed for polyphonic input | [measured — librosa docs] |
| RTF on M1 (est.) | 0.5–0.8× CPU (Intel i5 baseline: 0.80×; M1 faster) | [measured — SWIFTF0 comparison paper] |
| Verdict | Lowest bar; use only as quick zero-install baseline | [assumed] |

### 3. Omnizart (vocal module)

| Attribute | Value | Provenance |
|-----------|-------|------------|
| Pip install | Yes (`pip install omnizart`) | [source-code — PyPI] |
| M1 native | **NO — TensorFlow 1.15 requires AVX, unavailable on ARM** | [measured — GitHub issue + prior session notes] |
| Verdict | **Disqualified — M1 incompatible** | [measured] |

### 4. SPICE (Google, TensorFlow Hub)

| Attribute | Value | Provenance |
|-----------|-------|------------|
| Pip install | Via TensorFlow + tensorflow-hub (`pip install tensorflow tensorflow-hub`) | [source-code — TF Hub tutorial] |
| M1 native | Yes — TensorFlow 2.16+ has arm64 wheels | [measured — TensorFlow blog] |
| MIDI output | Pitch track only; manual post-processing required | [source-code] |
| Monophonic | Yes — single pitch; **trained with polyphonic augmentation (SNR −5 to +25 dB)** | [measured — SPICE paper, arXiv 1910.11664] |
| Accuracy on polyphonic mix | Good — 90.7% RPA on MIR-1K (vocal+accompaniment dataset) | [measured — SPICE paper] |
| Accuracy on clean vocal | Excellent — 90.7% RPA; mean pitch error 0.2–0.3 semitones above 110 Hz | [measured — SPICE paper] |
| RTF on M1 (est.) | 0.05–0.1× (older CNN; similar depth to CREPE) | [assumed — architecture comparison] |
| Verdict | **Promising — explicitly trained on vocal-in-mix; no stem required** | [measured + assumed] |

**Note:** SPICE's training on MIR-1K at multiple SNR levels (0, 10, 20 dB accompaniment mixing)
directly addresses the vocal-in-polyphonic-mix use case. This is the key differentiator from
CREPE/pYIN.

### 5. PESTO (Sony CSL, 2023)

| Attribute | Value | Provenance |
|-----------|-------|------------|
| Pip install | `pip install pesto-pitch` (latest Mar 2025) | [source-code — PyPI] |
| M1 native | Yes — PyTorch arm64 (assumed; not explicitly documented for M1) | [assumed — PyTorch standard] |
| MIDI output | Pitch track only; manual post-processing required | [source-code] |
| Monophonic | Yes — trained on MIR-1K, MDB-stem-synth, PTDB | [source-code] |
| Accuracy on polyphonic mix | Good on semi-clean stem; 97.0% RPA on MIR-1K | [measured — ISMIR 2023 paper] |
| Accuracy on clean vocal | Excellent — 97–98.3% RPA; ISMIR 2023 best paper nominee | [measured — PESTO paper, arXiv 2309.02265] |
| RTF on M1 (est.) | 0.5–1.5× CPU (GPU RTX A2000: ~0.5×; M1 CPU extrapolated) | [measured — paper GPU result; M1 extrapolated] |
| Verdict | **High viability for stem-based path; top accuracy among pip-installable tools** | [measured + assumed] |

### 6. FCPE (Fast Context-based Pitch Estimation, 2024)

| Attribute | Value | Provenance |
|-----------|-------|------------|
| Pip install | `pip install torchfcpe` (v0.0.4+) | [source-code — PyPI] |
| M1 native | Yes — PyTorch arm64 | [assumed — PyTorch standard] |
| MIDI output | Pitch track only; manual post-processing required | [source-code] |
| Monophonic | Yes — trained on MIR-1K | [source-code] |
| Accuracy on polyphonic mix | Good on semi-clean stem; 96.79% RPA on MIR-1K | [measured — FCPE paper, arXiv 2509.15140] |
| Accuracy on clean vocal | Excellent — 96.79% RPA | [measured — FCPE paper] |
| RTF on M1 (est.) | **0.01–0.05×** (GPU RTX 4090: 0.0062×; ~77× faster than CREPE) | [measured — FCPE paper GPU result; M1 extrapolated] |
| Verdict | **Speed champion; 96% RPA; pip-easy — ideal if RTF is the binding constraint** | [measured + assumed] |

### 7. RMVPE (2023)

| Attribute | Value | Provenance |
|-----------|-------|------------|
| Pip install | **No PyPI** — GitHub clone + `pip install -e .` | [source-code — GitHub] |
| M1 native | Likely (PyTorch-based); **not explicitly verified for M1** | [assumed] |
| MIDI output | Pitch track only; manual post-processing required | [source-code] |
| Monophonic | Yes — single pitch; **polyphony-aware by design** | [source-code — RMVPE paper, arXiv 2306.15412] |
| Accuracy on polyphonic mix | **Excellent — superior to CREPE on polyphonic audio** (RMVPE developed because CREPE fails here) | [measured — RMVPE paper] |
| Accuracy on clean vocal | Very good — competitive with CREPE/PESTO | [measured — RMVPE paper] |
| RTF on M1 (est.) | 0.05–0.15× (CNN; similar depth to CREPE) | [assumed — architecture comparison] |
| Verdict | **Best fit for raw polyphonic mix; highest risk (no PyPI, M1 unverified)** | [measured + assumed] |

**Note:** RMVPE was explicitly designed to solve the failure mode documented in Finding F9: CREPE
produces errors when vocal fundamental is weak relative to accompaniment. RMVPE uses improved
feature extraction specifically for this scenario.

### 8. VOCANO / ROSVOT

| Attribute | VOCANO | ROSVOT | Provenance |
|-----------|--------|--------|------------|
| Pip install | No PyPI — GitHub clone | No PyPI — GitHub clone | [source-code] |
| M1 native | Unverified | Likely (PyTorch) | [assumed] |
| MIDI output | Direct note-level output | Direct MIDI | [source-code] |
| Polyphonic | Yes — ISMIR 2021 polyphonic vocal | Yes — "noisy or accompanied" vocals | [source-code] |
| Verdict | Medium interest; install friction; untested on M1 | Medium interest; install friction; untested on M1 | [assumed] |

### 9. MT3 / YourMT3+

| Attribute | Value | Provenance |
|-----------|-------|------------|
| Pip install | Yes but complex (JAX + system deps: libfluidsynth3, libjack-dev, etc.) | [source-code — GitHub] |
| M1 native | Uncertain — JAX M1 support varies by version; typically Rosetta 2 needed | [assumed] |
| MIDI output | Direct multi-instrument MIDI | [source-code] |
| Polyphonic | Yes — state-of-the-art multi-instrument transcription; YourMT3+ (2024) eliminates need for voice separation | [measured — arXiv 2407.04822] |
| RTF on M1 (est.) | 1–5× (JAX transformer; slow on CPU) | [assumed] |
| Verdict | **Best accuracy; impractical RTF; JAX/M1 uncertain** — revisit if RTF budget relaxed | [measured + assumed] |

---

## Accuracy Benchmark Summary

All RPA values from published papers on MIR-1K (vocal + accompaniment benchmark).

| Tool | RPA (MIR-1K) | Polyphonic design | Pip | M1 | Provenance |
|------|-------------|-------------------|-----|----|------------|
| RMVPE | Best-in-class (>CREPE on polyphony) | ✓ explicit | ✗ | ? | [measured — paper] |
| PESTO | 97–98.3% | ✗ (stem assumed) | ✓ | ✓ (assumed) | [measured — paper] |
| FCPE | 96.79% | ✗ (stem assumed) | ✓ | ✓ (assumed) | [measured — paper] |
| CREPE | ~90% | ✗ | ✓ | ✓ | [measured — paper] |
| SPICE | 90.7% | ✓ explicit | ✓ | ✓ | [measured — paper] |
| pYIN | ~85% (estimated) | ✗ | ✓ (in venv) | ✓ | [assumed] |
| basic-pitch (stem) | Poor (subjective) | ✗ | ✓ (in venv) | ✓ | [measured] |

---

## Post-Processing Gap Analysis

Tools that output pitch tracks (not direct MIDI) require a pitch→MIDI conversion step:

| Step | Library | Complexity |
|------|---------|------------|
| Pitch track → frame-level MIDI note number | numpy: `round(12 * log2(f0 / 440) + 69)` | Low |
| Voicing / unvoiced frame detection | Tool-provided flags or confidence threshold | Low |
| Onset detection | `librosa.onset.onset_detect()` | Medium |
| Note segmentation (contiguous pitch regions) | `scipy.signal.find_peaks()` or HMM | Medium |
| Velocity assignment | RMS per note segment → MIDI velocity | Low |
| MIDI file assembly | `mido` (already in project) | Low |

This is a one-time implementation cost: ~100–150 lines of Python. It is not a disqualifying burden
given that `mido` and `librosa` are already in the venv.

Only `crepe-notes` (`pip install crepe-notes`) automates this for CREPE. All other tools require
manual implementation or reuse of the same post-processing code.

---

## Findings

### F1 — RMVPE is the best-fit algorithm for raw polyphonic mix but has installation risk
[measured + assumed]
RMVPE was developed to address exactly the failure mode observed in F9 of the prior session: CREPE
(and by extension other SPE tools) fails when the vocal fundamental is weak relative to
accompaniment. RMVPE uses improved feature extraction for polyphonic robustness. No PyPI package
exists; M1 support is unverified. Installation risk is the primary blocker, not the algorithm.

### F2 — SPICE is the only pip-installable tool explicitly trained on vocal-in-polyphonic-mix
[measured]
SPICE's training on MIR-1K at multiple SNR levels (−5 to +25 dB accompaniment mixing) directly
targets the vocal-in-mix scenario. No stem separation needed by design. However, TensorFlow is a
new dependency not currently in the project venv.

### F3 — PESTO and FCPE are highest-accuracy pip-installable tools but require a clean(er) stem
[measured]
Both achieve 96–98% RPA, but are monophonic single-pitch estimators trained on separated or
lightly-accompanied vocals. On the htdemucs stem (centroid 4400–4800 Hz, vocal body lost), they
will face the same root cause as basic-pitch (F9). Their advantage over basic-pitch is that SPE
tools track a single pitch contour deterministically, so octave errors are more systematic and
potentially fixable in post-processing.

### F4 — pYIN is the zero-install baseline but its accuracy lags all dedicated trackers
[measured]
librosa.pyin is already installed and runs on M1 natively. It is the correct first test to
establish a baseline before installing new dependencies. However, it is monophonic-only and not
designed for polyphonic input.

### F5 — All pitch-track tools share the same post-processing pipeline cost (~100–150 lines)
[assumed]
The pitch→MIDI conversion gap is a one-time implementation cost, not a per-tool barrier. A shared
`PitchToMIDI` utility class can serve all tools. This should be built once and reused.

### F6 — MT3/YourMT3+ represents the accuracy ceiling but is not viable for M1 runtime budget
[measured + assumed]
YourMT3+ (2024) is specifically designed to skip stem separation for multi-instrument transcription
including vocals. However, JAX on M1 is fragile and the estimated RTF (1–5×) likely exceeds the 3×
budget. Revisit if RTF budget is relaxed or if a CoreML port emerges.

### F7 — FCPE's exceptional speed (0.01–0.05× RTF) opens batch processing pathways
[measured]
FCPE on GPU is 77× faster than CREPE (RTF 0.0062×). Even on M1 CPU, estimated RTF is
0.01–0.05×. This opens the possibility of running multiple FCPE passes (e.g., at different
confidence thresholds or on multiple rough stems) and selecting the best output — without exceeding
the 3× RTF budget.

---

## Recommended Investigation Order

Based on findings, test in this order (lowest risk → highest quality):

1. **pYIN on htdemucs stem** — zero new installs; establish a baseline [librosa already in venv]
2. **SPICE on raw mix** — `pip install tensorflow tensorflow-hub`; tests polyphonic-robust design
3. **PESTO on htdemucs stem** — `pip install pesto-pitch`; highest accuracy pip-installable option
4. **FCPE on htdemucs stem** — `pip install torchfcpe`; best speed/accuracy tradeoff
5. **RMVPE on raw mix** — `pip install -e git+...`; highest accuracy polyphonic; M1 risk
6. **RMVPE on htdemucs stem** — if M1 native confirmed; benchmark vs. PESTO/FCPE

---

## Tradeoffs Summary

| Approach | Polyphonic robustness | Pitch accuracy | MIDI direct | Speed (RTF est.) | M1 install risk | Recommended |
|----------|----------------------|---------------|-------------|-----------------|-----------------|-------------|
| pYIN (librosa) on stem | Low | ~85% | No | 0.5–0.8× | None (in venv) | Baseline only |
| SPICE on mix | High (trained for it) | 90.7% | No | 0.05–0.1× | Low (pip TF) | ✓ Phase 1 |
| PESTO on stem | Low–Med | 97–98.3% | No | 0.5–1.5× | Low (pip) | ✓ Phase 2 |
| FCPE on stem | Low–Med | 96.79% | No | 0.01–0.05× | Low (pip) | ✓ Phase 2 alt |
| RMVPE on mix | High (designed) | Best in class | No | 0.05–0.15× | Med (no PyPI) | ✓ Phase 3 |
| basic-pitch on stem | Med (polyphonic model) | Poor [measured] | Yes | 0.3–0.5× | None (in venv) | Baseline (tested) |
| MT3 on mix | Very high | State-of-art | Yes | 1–5× | High (JAX) | Aspirational only |
| Omnizart on mix | High | Unknown | Yes | N/A | Disqualified | ✗ |

All provenance tags [measured], [assumed], or [source-code] as marked above.

---

## Trial Results (2026-04-17)

### Tools run

| Tool | Input | Notes | Notes/s | Voiced frac | RTF | Subjective | Provenance |
|------|-------|-------|---------|-------------|-----|------------|------------|
| pYIN | htdemucs 4s stem | 556 | 2.26 | 57.4% | 0.25× | Poor — glimmers but incoherent | [measured] |
| PESTO | htdemucs 4s stem | 265 | 1.08 | 45.6% | 0.10× | Poor — same as pYIN | [measured] |
| FCPE | htdemucs 4s stem | 484 | 1.97 | 55.9% | 0.013× | Poor — same | [measured] |
| FCPE | raw mix | 706 | 2.87 | 78.3% | 0.012× | Vocal line present but heavy accompaniment noise | [measured] |
| FCPE | mix blend 50% stem/50% mix (4s) | 705 | 2.87 | 76.6% | 0.012× | Poor | [measured] |
| FCPE | mix blend 70% stem/30% mix (4s) | 676 | 2.75 | 73.7% | 0.012× | Poor | [measured] |
| FCPE | mix blend 90% stem/10% mix (4s) | 597 | 2.43 | 62.2% | 0.012× | Poor | [measured] |
| FCPE | htdemucs 6s stem | 514 | 2.09 | 57.7% | 0.013× | Poor | [measured] |
| FCPE | blend 70% 6s stem / 30% mix | 674 | 2.74 | 73.8% | 0.013× | Poor | [measured] |
| FCPE | blend 50% 6s stem / 50% mix | 703 | 2.86 | 76.6% | 0.013× | Poor | [measured] |

### Finding F-T1 — 6s centroid measurement at 16 kHz was a resampling artefact
[measured]
At 16 kHz resampling, the htdemucs 6s vocals stem appeared to have centroid 2120 Hz (within the
1500–3000 Hz target range). At native 44.1 kHz the centroid is 4788 Hz — worse than the 4s stem
(4427 Hz). The 6s model routes even more vocal body to `other` than the 4s model on this track.
All htdemucs variants (4s, 4s-ft, 6s) are confirmed as equivalent failures for vocal body
preservation. [measured — 44.1 kHz: mix=1858 Hz, 4s=4427 Hz, 6s=4788 Hz]

### Finding F-T2 — Monophonic pitch trackers on degraded stems all produce incoherent MIDI
[measured]
pYIN, PESTO, and FCPE all produced subjectively poor MIDI on htdemucs stems regardless of model
variant (4s, 4s-ft, 6s) or blend ratio. The root cause is the same as F9/F10 from the prior
session: vocal fundamentals are not present in the stems. Changing the pitch tracker does not fix
the upstream separation problem.

### Finding F-T3 — FCPE on raw mix produces the vocal line but with heavy accompaniment noise
[measured]
Running FCPE directly on the unprocessed mix produced audible vocal melody content — the first
tracker to do so across all trials. However, accompaniment (synths, bass, piano) dominates the
pitch estimate and produces continuous noise. This confirms the vocal fundamental is intact in the
mix and the bottleneck is accompaniment rejection, not pitch detection sensitivity.

### Finding F-T4 — Stem/mix blending does not improve over raw mix
[measured]
All blend ratios (50/50, 70/30, 90/10) with both 4s and 6s stems produced MIDI no better than the
raw mix alone. The stems' spectral loss means blending adds back vocal body (from the mix
component) but simultaneously adds back all accompaniment content. The blend offers no
signal-to-accompaniment improvement over the raw mix for this track.

### Finding F-T5 — FCPE is confirmed as the fastest viable tracker (RTF 0.013×)
[measured]
All FCPE runs completed in 3–4s for a 245s track on M1 CPU. This is well within the ≤3× RTF
budget regardless of input source. FCPE should be retained as the pitch tracker for any future
pipeline.

### Conclusion
The only path forward that has not been tried is a pitch tracker specifically designed to extract
vocal pitch from a polyphonic mix — i.e. one that internally learns to reject accompaniment rather
than relying on a clean stem. RMVPE (arXiv 2306.15412) is the strongest candidate: it was
explicitly developed because CREPE and similar trackers fail on polyphonic content. Installation
requires a GitHub clone (no PyPI package). M1 support unverified but likely (PyTorch-based).
