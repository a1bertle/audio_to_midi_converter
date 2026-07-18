# Research Notes: Vocal-to-MIDI Algorithms — Theory and Core Algorithms

**Date:** 2026-04-18
**Session:** `research/2026-04-18_vocal-to-midi-theory/`
**Goal:** Survey the theoretical foundations and algorithmic landscape of vocal-to-MIDI
transcription, independent of hardware. Focus on understanding the problem decomposition,
algorithm families, mathematical underpinnings, and current state of the art.
**Context:** Prior sessions have benchmarked FCPE, PESTO, RMVPE, and pYIN at the empirical level
(2026-04-17_vocal-pitch-tracker-research). This session records the theoretical layer — the
concepts and signal processing theory that explain *why* algorithms succeed or fail — to ground
future implementation decisions.

---

## 1. Problem Decomposition

Vocal-to-MIDI transcription decomposes into five sequential sub-problems. Each has its own
theoretical literature and failure modes.

```
Audio Waveform
     │
     ▼
[1] Source Separation (optional)
     │   separate vocals from accompaniment
     ▼
[2] F0 / Pitch Estimation
     │   extract fundamental frequency (F0) frame-by-frame
     ▼
[3] Voicing Detection
     │   classify each frame: voiced (singing) vs. unvoiced (silence/noise)
     ▼
[4] Note Segmentation
     │   convert pitch contour → note events (onset, offset, pitch)
     ▼
[5] MIDI Assembly & Expression Encoding
         quantize pitch to semitones, assign velocity, encode bends
```

Each stage has independent failure modes. A pipeline fails if *any* stage fails. Most
vocal-to-MIDI pipelines in prior sessions failed at Stage 2 (F0) due to a broken Stage 1
(separation).

[assumed — synthesis of literature survey; structure is standard in MIR review papers]

---

## 2. Stage 1 — Source Separation

### 2.1 Why It Is Needed (and When It Can Be Skipped)

Standard pitch estimators (CREPE, pYIN, etc.) are monophonic: they assume a single dominant
pitch per frame. In a polyphonic mix, accompaniment harmonics interfere with vocal harmonics,
causing the estimator to latch onto the wrong candidate. Separation isolates the vocal signal
before pitch estimation.

**Separation bottleneck confirmed in prior sessions:**
- All htdemucs variants produce vocal stems with spectral centroid 4427–4788 Hz [measured —
  2026-04-13_vocal-midi-bleed-tolerance/notes.md]
- Vocal fundamental range is ~100–900 Hz (soprano up to ~1100 Hz) [assumed — acoustic knowledge]
- htdemucs routes vocal body (fundamentals + lower formants) to the `other` stem, destroying
  pitch information before Stage 2 [measured — F9, 2026-04-13]

**When separation can be skipped:**
Algorithms that model vocal-in-accompaniment explicitly (RMVPE, SPICE) operate on the mix
directly. They internalize an accompaniment rejection mechanism as part of their architecture.

### 2.2 Neural Separation: U-Net / Transformer Models

Current state-of-the-art (Demucs htdemucs family, MEL-RoFormer) uses deep encoder-decoder
networks trained on musically diverse datasets. The core architecture:

- **Encoder:** convolutional or transformer blocks that produce multi-scale latent representations
  of the spectrogram or waveform
- **Skip connections:** U-Net style; pass high-resolution features from encoder to decoder
- **Decoder:** mirrors encoder; outputs stem waveform or mask applied to original mixture
- **Loss:** combination of spectral and time-domain L1/L2 losses (e.g., SI-SNR)

**Key limitation for vocals:** Networks trained to maximize signal-to-noise ratio for
*percussive/harmonic instrument* classes may reassign low-frequency vocal energy to the `other`
stem because it co-occurs with bass/piano in frequency. The network's training objective does not
directly optimize for pitch estimation quality downstream.

[assumed — synthesis from Demucs/Hybrid-Demucs papers and session findings; no architecture
paper directly cited in session measurements]

---

## 3. Stage 2 — Fundamental Frequency (F0) Estimation

This is the most theoretically rich sub-problem and the focus of most vocal-to-MIDI research.

### 3.1 Classical Signal Processing Methods

#### 3.1.1 Autocorrelation (ACF)

The earliest and most basic approach. For a periodic signal with period T, the autocorrelation
function R(τ) peaks at τ = T. F0 = 1/T.

```
R(τ) = Σ x(t) · x(t + τ)
```

**Failure modes:**
- Sub-harmonic errors: the function also peaks at τ = 2T, 3T, etc., causing octave errors
- Phase sensitivity: does not handle inharmonicity (vocal formants, noise) well
- No voicing decision built in

[source-code — de Cheveigné & Kawahara 2002, JASA 111.4:1917–1930]

#### 3.1.2 YIN (2002)

YIN replaces autocorrelation with the **squared difference function** (SDF):

```
d(τ) = Σ [x(t) - x(t + τ)]²
```

Expanded, this relates to autocorrelation: `d(τ) = R(0) + R(0) - 2R(τ)`. YIN then applies
**cumulative mean normalization** (CMNDF):

```
d'(τ) = 1,                         τ = 0
d'(τ) = d(τ) / [(1/τ) Σ d(j)],    τ > 0
```

This suppresses the systematic bias toward long lags, preventing the sub-harmonic errors of raw
ACF. A threshold is applied to the CMNDF to find the first minimum below the threshold.

**Key properties:**
- No upper frequency bound (suitable for high-pitched soprano voices)
- Error rate ~3× lower than best competing methods at 2002 benchmark [measured — YIN paper]
- Purely signal-processing: fast, no training data needed
- Monophonic only; no accompaniment rejection

[source-code — de Cheveigné & Kawahara 2002 (arXiv / JASA)]

#### 3.1.3 pYIN (2014)

Extends YIN with a **probabilistic framework** (de Mareschal, Mauch & Dixon, ICASSP 2014):

- YIN produces multiple F0 candidates per frame (all threshold-crossing minima in CMNDF)
- pYIN assigns **probability distributions** over these candidates using a mixture model:
  - "Voiced + pitch = p" probability for each candidate
  - "Unvoiced" probability for the frame
- A **Viterbi decoder** runs over the candidate lattice, finding the globally optimal pitch
  trajectory given a smoothness prior (large pitch jumps penalized)

The Viterbi pass provides joint voicing detection and pitch smoothing — a key advantage over raw
YIN. This is why pYIN produces coherent pitch tracks while YIN alone is noisy.

**Mathematical model:**
```
State lattice: frames × (unvoiced + N_pitch_candidates)
Transition cost: penalizes pitch jumps > 1 semitone per frame
Emission cost: negative log probability from pYIN mixture model
Viterbi: finds min-cost path through lattice
```

[source-code — Mauch & Dixon, ICASSP 2014; librosa.pyin implementation]

**Performance:** ~85% RPA on vocal benchmarks [assumed — estimated from literature comparison
with CREPE; no direct MIR-1K citation found for pYIN in search results]

---

### 3.2 Neural Network Methods

#### 3.2.1 CREPE (2018) — CNN on Time-Domain Waveform

**Core idea:** Train a deep CNN to classify raw waveform frames into 360 pitch bins (50 Hz to
2006 Hz, log-spaced at 20 cents/bin). No hand-crafted features.

**Architecture:**
- Input: 1024-sample window of raw 16 kHz audio (~64 ms)
- 6 convolutional layers with increasing dilation → 2048-dimensional latent space
- Dense layer → 360-dimensional output with sigmoid activations
- Each output unit = probability that the pitch is in that bin
- Final F0 = weighted mean of bin center frequencies (probability-weighted)

**Why CNNs work here:** The convolution layers learn to detect harmonic patterns (ratios 1:2:3:4)
in the waveform directly, without needing a Fourier transform. The learned filters implicitly
represent the autocorrelation structure but in a task-optimal basis.

**Performance:** ~90% RPA on clean vocal; **"sharply decreases" on polyphonic content** [measured
— RMVPE paper arXiv 2306.15412 citing CREPE failure mode]

[source-code — Kim et al., ICASSP 2018, arXiv 1802.06182]

#### 3.2.2 PESTO (2023) — Self-Supervised Transposition Equivariance

**Core innovation:** Train without labeled pitch data using a **transposition-equivariance
objective**: if audio A is transposed up by k semitones to produce audio B, the pitch estimate of
B should be exactly k semitones higher than A.

**Architecture:**
- Input: Constant-Q Transform (CQT) / Variable-Q Transform (VQT) of audio (time-frequency)
- **Siamese network**: two identical encoder branches process two transposed versions of the same
  audio
- **Toeplitz matrices** (learnable): enforce shift-equivariance in the CQT frequency axis —
  because transposing audio = shifting the CQT log-frequency axis, a Toeplitz-structured weight
  matrix preserves this symmetry
- Output: pitch distribution over ~360 cents bins; Siamese loss = equivariance error

**Why self-supervised works:** CQT already maps pitch transposition to a log-frequency shift.
Training the network to be invariant to the *absolute position* while equivariant to *relative
shift* forces it to encode pitch.

**Performance:** 97–98.3% RPA on MIR-1K; Best Paper nominee ISMIR 2023 [measured — PESTO paper
arXiv 2309.02265]; <30k parameters; 12× faster than real-time on consumer CPU [measured — PESTO
paper]

[source-code — Riou et al., ISMIR 2023, arXiv 2309.02265]

#### 3.2.3 RMVPE (2023) — U-Net + GRU for Polyphonic Robustness

**Core innovation:** Replace CREPE's classification of raw waveform with a **U-Net encoder-decoder
on mel-spectrograms**, specifically designed to reject accompaniment.

**Architecture:**
- Input: log mel-spectrogram (16 kHz, 256 mel bins, hop=320 samples=20 ms, window=2048 samples,
  range 30–8000 Hz)
- **U-Net encoder:** multiple convolutional blocks, each halving the time-frequency resolution;
  skip connections to decoder
- **GRU layers:** between encoder and decoder, capturing long-range temporal context (important
  for sustained vocal notes)
- **Decoder:** mirrors encoder; reconstructs full-resolution pitch probability map
- **Output:** voiced probability + pitch probability distribution (360 bins, 20 cents/bin) per
  frame

**Why U-Net helps polyphonic scenarios:** The multi-scale encoder captures both local harmonic
patterns (fine scale) and global melodic structure (coarse scale). The skip connections allow the
decoder to localize the vocal pitch against a background of accompaniment partials that appear at
different scales.

**Loss function:** Weighted cross-entropy, with higher weights for voiced frames (addressing
class imbalance — unvoiced frames dominate most audio).

**Training:** Polyphonic vocal datasets with accompaniment at various SNR levels. RMVPE's training
specifically includes polyphonic mixtures.

**Performance:** Best-in-class on polyphonic content; outperforms CREPE significantly on
accompaniment-heavy audio [measured — RMVPE paper arXiv 2306.15412]; degrades only 0.83% RPA
vs. HARMOF0 on MDB-stem-synth [measured — RMVPE paper]

[source-code — Wei et al., arXiv 2306.15412]

#### 3.2.4 FCPE (2025) — Conformer + Depthwise Conv (Speed-Optimized)

**Core innovation:** Replace U-Net with a **Lynx-Net** (lightweight Conformer variant) to achieve
extreme speed while maintaining ~97% accuracy.

**Architecture:**
- Input: log mel-spectrogram → shallow 1D conv embedding block
- **Optional harmonic embedding:** learnable projection that explicitly encodes harmonic frequency
  ratios (f, 2f, 3f, 4f) into the input representation
- **Lynx-Net backbone:** stack of Conformer-inspired blocks using **depthwise separable Conv1D**
  — reduces parameter count and FLOPs relative to standard Conv2D while preserving temporal
  context modeling
- Output: pitch probability distribution per frame
- **RTF:** 0.0062 on RTX 4090; ~5.3× faster than RMVPE; ~77× faster than CREPE [measured —
  FCPE paper arXiv 2509.15140]

**Why depthwise separable convolutions:** Standard Conv1D with kernel k and C channels costs
O(k·C²) per layer; depthwise separable factorizes this to O(k·C + C²), a ~k× reduction. For
k=3–7 (common in pitch models), this is a 3–7× compute reduction per layer.

**Performance:** 96.79% RPA on MIR-1K [measured — FCPE paper]

[source-code — arXiv 2509.15140]

---

### 3.3 Comparison: Algorithm Families

| Property | Classical (YIN/pYIN) | CREPE-family (CNN) | RMVPE (U-Net) | PESTO (SSL) | FCPE (Conformer) |
|---|---|---|---|---|---|
| Input domain | Time-domain (diff. function) | Time-domain waveform | Mel-spectrogram | CQT/VQT | Mel-spectrogram |
| Polyphonic robustness | None | Poor | High (by design) | Low–Med | Low–Med |
| Training data needed | None | Large labeled | Large labeled | Self-supervised | Large labeled |
| Parameters | 0 | ~1M | ~12M [assumed] | <30k | <1M [assumed] |
| RPA (clean vocal) | ~85% | ~90% | ~97% | 97–98.3% | 96.79% |
| RPA (polyphonic mix) | Poor | "Sharply decreases" | Best-in-class | Not evaluated | Not evaluated |
| Speed (RTF relative) | 0.5–0.8× CPU | 1× (CREPE baseline) | 5.3× slower than FCPE | 12× faster than RT | Fastest (0.0062× GPU) |

Provenance: [measured — respective papers]; [assumed] where noted.

---

## 4. Stage 3 — Voicing Detection

Voicing detection classifies each frame as **voiced** (the singer is sustaining a pitch) vs.
**unvoiced** (silence, consonant, breath noise). This is critical: feeding unvoiced frames to
Stage 4 generates spurious note events.

### 4.1 Threshold-Based (Classical)

- pYIN: outputs a probability p(voiced|frame) from the mixture model; threshold at p > 0.5
- CREPE: no built-in voicing; confidence = max of output distribution; threshold chosen manually
- Energy gating: simple RMS threshold pre-filters obvious silence

### 4.2 Neural Voicing Detection

- RMVPE and FCPE output a separate voiced probability alongside the pitch distribution
- Trained on annotated voiced/unvoiced labels; learns to distinguish singing from breath noise,
  plosives, and vibrato troughs (which can briefly look like pitch instability)

**Critical failure mode:** Vibrato causes the instantaneous F0 to oscillate ±50–100 cents
around the note center at ~5–7 Hz. A poorly calibrated voicing detector may drop frames at
vibrato troughs, fragmenting one long note into many short ones.

[assumed — synthesis from vocal transcription literature and prior session observations]

---

## 5. Stage 4 — Note Segmentation

### 5.1 The Core Challenge

The output of Stage 2–3 is a **pitch contour**: a sequence of (time, F0, voiced_flag) triples.
Converting this to discrete MIDI notes requires identifying:

1. **Onset:** when a note begins
2. **Offset:** when a note ends
3. **Pitch:** the canonical semitone value (not the instantaneous F0, which varies due to vibrato,
   portamento, ornaments)

This is not trivial because:
- Vibrato makes the pitch oscillate continuously; the note center must be extracted
- Portamento produces a smooth glide between notes with no clear onset in pitch
- Vowels transition between pitches without stopping; the boundary is perceptual, not acoustic
- Consonants and breath create low-energy gaps that may or may not represent note offsets

### 5.2 Hidden Markov Model (HMM) Approaches

The standard classical approach models each note as a 3-state left-to-right HMM:

```
State 1 (attack): F0 rising to target; short duration
State 2 (sustain): F0 near target with vibrato; longer duration
State 3 (release): F0 falling or fading; short duration
```

Viterbi decoding over the pitch contour finds the globally optimal note sequence:

```
Observation: (F0_frame, energy_frame) per frame
States: silence | note_1_attack | note_1_sustain | note_1_release | note_2_attack | ...
Transition: penalizes unrealistic note durations and pitch jumps
Emission: Gaussian likelihood around target pitch per state
```

The HMM provides a principled way to handle vibrato (sustain state absorbs oscillation) and
portamento (modeled as attack→release transition without sustain).

[source-code — Nakamura et al., ISMIR 2016; TONY (Mauch et al.) implementing HMM on pYIN output]

### 5.3 Onset + Frame (Neural) Approaches

Basic-pitch (Spotify) and related systems use two parallel binary classifiers:

```
Onset predictor:  P(onset|frame) — high at note beginnings
Frame predictor:  P(active|frame) — high during note continuation
```

A note is created when onset predictor fires AND frame predictor is high in subsequent frames. A
note ends when frame predictor drops below threshold.

This avoids HMM complexity but requires labeled onset annotations for training.

**Comparison:**

| Method | Onset detection | Vibrato handling | Portamento handling | Requires labeled onsets |
|---|---|---|---|---|
| Threshold on dF0/dt | Simple diff | Poor | Misses | No |
| HMM / Viterbi | Probabilistic | Good (sustain state) | Moderate | No |
| Onset+Frame (neural) | Learned | Learned | Learned | Yes |
| Object detection (YOLO-based) | Bbox localization | Depends | Depends | Yes |

[source-code — VOCANO ISMIR 2021; Basic Pitch; Nakamura ISMIR 2016]

### 5.4 Modern End-to-End Approaches

ROSVOT (2024) and unified frameworks (ACL 2025) replace the cascade entirely:

- **Multi-scale encoder** processes audio at coarse (note-level) and fine (frame-level) granularity
- **Attention-based pitch decoder** predicts onset, offset, and pitch jointly
- Output: token sequence [onset_time, offset_time, pitch, note_value]
- **Autoregressive decoding:** next note token conditioned on all previous tokens

End-to-end avoids error propagation between stages but requires large annotated datasets and
heavier inference.

[source-code — arXiv 2405.09940; ACL 2025 findings paper]

---

## 6. Stage 5 — MIDI Assembly and Expression Encoding

### 6.1 Pitch Quantization

The raw F0 contour contains sub-semitone deviations. Two strategies:

1. **Hard quantization:** round every frame to nearest semitone. Loses all expression. Suitable
   for music notation use cases.
2. **Pitch-bend encoding:** preserve F0 deviation as MIDI pitch-bend events. Standard MIDI pitch
   bend = ±2 semitones (or ±12 with controller configuration). Resolution: 14-bit (16384 steps).

For vocal MIDI, hard quantization destroys vibrato, portamento, and expressive intonation. The
right choice depends on the target use case:
- **Score generation:** hard quantization + note-level pitch label
- **Vocal synthesis / SVC:** preserve pitch bend to enable re-synthesis

[assumed — synthesis from Basic Pitch docs and MIDI standard]

### 6.2 Velocity Assignment

Vocal MIDI velocity is typically derived from the **RMS energy** of the audio within the note
window, mapped to MIDI velocity range 0–127.

```
velocity = clip(int(127 * (rms - rms_min) / (rms_max - rms_min)), 1, 127)
```

A dynamic range compression step (log scale) is often applied because vocal dynamics span 40–60
dB but MIDI expects approximately linear perceived loudness.

[assumed — standard practice; no paper citation]

### 6.3 Vibrato Modeling

Vibrato is a sinusoidal modulation of F0 at 5–7 Hz with ±50–100 cents amplitude. For note
segmentation, the note center pitch = the mean or median F0 over the note's voiced frames. For
MIDI pitch-bend output, the full F0 contour is encoded.

Some systems detect vibrato explicitly and generate periodic pitch-bend curves; others pass the
raw F0 deviation as-is. The difference matters for synthesis but not for MIDI note identity.

[assumed — synthesis from vibrato research literature]

---

## 7. Key Theoretical Tradeoffs Summary

### 7.1 Why Monophonic Pitch Estimators Fail on Polyphonic Audio

A monophonic estimator maximizes `P(F0 | frame)` — the most likely single pitch given the frame
features. In a polyphonic mix, this distribution is multi-modal: there are peaks at the vocal F0,
the piano fundamental, the bass, etc. The estimator picks the globally dominant peak, which is
often not the vocal.

RMVPE's design addresses this by:
1. Training on labeled vocal pitch in accompaniment — the network learns to suppress non-vocal
   harmonics
2. Multi-scale U-Net — vocal harmonic structure differs from accompaniment in temporal persistence
   (sustained notes vs. attack-decay-sustain-release percussion/piano), and the U-Net's coarse
   scale captures this temporal difference

[measured — RMVPE paper; confirmed experimentally in 2026-04-17 session F-T3]

### 7.2 CQT vs. Mel-Spectrogram vs. Raw Waveform

| Representation | Log-pitch shift = | Harmonic structure | Data efficiency |
|---|---|---|---|
| Raw waveform | Complex phase rotation | Implicit (learned by CNN) | Low |
| Mel-spectrogram | Non-linear frequency warp | Partially explicit | Medium |
| CQT | Frequency shift | Explicit (constant ratio bins) | High (PESTO exploits this) |

CQT's property that pitch transposition = frequency-axis shift is the mathematical foundation of
PESTO's transposition-equivariance objective. This is why PESTO achieves high accuracy with <30k
parameters.

[source-code — PESTO paper arXiv 2309.02265]

### 7.3 The Pitch→Note Boundary Problem

The F0 contour is **continuous**; MIDI notes are **discrete events**. The gap between these is
the fundamental challenge of Stage 4. Three theoretical handles:

1. **Derivative-based:** dF0/dt spike → onset. Simple but fails for portamento (smooth F0
   transition has no spike).
2. **Energy-based:** rapid amplitude drop → offset. Reliable for consonants; misses legato
   singing.
3. **Probabilistic / sequential:** HMM or neural models that learn the perceptual note boundary
   jointly from F0, energy, and timing context. Requires either labeled data or a strong prior
   model.

The derivative approach is appropriate as a fast baseline. HMM or neural is required for
high-quality transcription of expressive singing.

[assumed — synthesis from MIR literature]

---

## 8. Algorithm Genealogy

```
1973  AMDF (average magnitude difference function) — first practical F0 estimator
1977  Harmonic Product Spectrum
1993  SWIPE (sawtooth waveform inspired pitch estimator)
2002  YIN — squared difference + cumulative normalization
2004  HPSS (harmonic-percussive source separation) — enables simpler pitch tracking
2014  pYIN — probabilistic YIN + Viterbi; HMM note segmentation (TONY)
2018  CREPE — first CNN direct waveform pitch estimator
2018  Basic Pitch precursors — CQT + onset+frame model
2021  VOCANO — note-level singing transcription in polyphonic music (ISMIR)
2022  Basic Pitch (Spotify) — CQT + onset+frame neural model
2023  PESTO — self-supervised transposition-equivariant CQT model
2023  RMVPE — U-Net + GRU; polyphonic-robust vocal pitch
2024  ROSVOT — multi-scale attention encoder, end-to-end note transcription
2025  FCPE — Conformer-based mel-spec model; fastest published (arXiv 2509.15140)
2025  MEL-RoFormer — joint separation + transcription
2025  Unified SVTS frameworks (ACL 2025) — end-to-end MIDI + phoneme + technique
```

[source-code — respective papers; dates from arXiv submission dates]

---

## 9. Open Problems (2025–2026)

1. **Polyphonic pitch estimation without separation:** RMVPE is the best available, but
   accompaniment rejection is still imperfect on bass-heavy or dense harmonic content.
2. **Note boundary quality in expressive singing:** Vibrato, ornaments, and portamento remain
   poorly handled by all non-end-to-end systems.
3. **Velocity and dynamics modeling:** Most systems assign velocity from RMS energy alone;
   perceptual loudness and vocal register (chest/head) are not modeled.
4. **Language-dependent transcription:** Consonant patterning differs across languages; most
   models trained on English/Mandarin generalize poorly to flamenco, Hindustani, etc.
5. **Real-time inference:** FCPE approaches real-time; end-to-end systems are still 1–5× slower
   than real-time for full tracks.

[assumed — synthesis from literature; no direct citation per item]

---

## 10. Prior Session Cross-Reference

| Session | Relevant Finding |
|---------|-----------------|
| 2026-03-29_ml_audio_to_midi_models | Survey of AMT tools; basic-pitch and PTI identified as primary backends |
| 2026-04-12_alt-models-eval | Demucs model comparison; htdemucs_6s evaluated |
| 2026-04-13_escene-vocals-eval | Mix spectral centroid 2041 Hz baseline established |
| 2026-04-13_vocal-midi-bleed-tolerance | All htdemucs variants confirmed to lose vocal body below 3 kHz |
| 2026-04-17_vocal-pitch-tracker-research | FCPE/PESTO/pYIN benchmarked; FCPE on raw mix yields vocal line with accompaniment noise; RMVPE identified as best next candidate |

---

## 11. Findings

### F1 — Vocal-to-MIDI is a five-stage cascade; stage failure propagates forward
[assumed — synthesis]
Any single stage failure (especially Stage 1 or 2) produces unusable output regardless of the
quality of downstream stages. Prior sessions' failures were caused by Stage 1 (separation) not
Stage 2 (pitch estimation).

### F2 — Classical algorithms (YIN/pYIN) have mathematically principled voicing and smoothing
[source-code — YIN/pYIN papers]
pYIN's Viterbi-decoded HMM over pitch candidates is theoretically sound and still competitive
for clean-vocal scenarios without the cost of GPU inference. Its failure mode is polyphonic input,
not its mathematical formulation.

### F3 — Neural pitch estimators learn harmonic structure in the input representation
[source-code — CREPE, RMVPE papers]
CNN-based estimators implicitly learn autocorrelation-like harmonic filters from data. The
architecture choice (waveform vs. mel vs. CQT) determines which pitch symmetries are explicitly
preserved vs. must be learned.

### F4 — PESTO's self-supervised objective exploits CQT's pitch-shift = frequency-shift property
[source-code — PESTO paper arXiv 2309.02265]
This is a mathematically elegant design: the equivariance objective trains for pitch without
labeled data, while Toeplitz matrices enforce shift-symmetry algebraically. The result is a
<30k parameter model matching labeled models in accuracy.

### F5 — RMVPE's U-Net architecture is the theoretically justified choice for polyphonic scenarios
[source-code — RMVPE paper arXiv 2306.15412]
Multi-scale skip connections allow the decoder to distinguish vocal F0 from accompaniment by
combining local harmonic patterns with global temporal structure. The GRU captures note-level
continuity that frame-level classifiers miss.

### F6 — Note segmentation is the hardest unsolved stage for expressive singing
[assumed — synthesis]
F0 estimation has reached 96–98% RPA on clean vocals. The remaining quality gap in vocal MIDI
transcription comes from Stage 4: converting a continuous F0 contour with vibrato, portamento,
and ornaments into discrete note events. This is where the most improvement is available.

### F7 — End-to-end models (ROSVOT, ACL 2025) address Stage 4 directly but require labeled data
[source-code — arXiv 2405.09940]
The theoretical advantage is joint optimization across all stages, eliminating error propagation.
The practical cost is dataset requirements and heavier inference.

### F8 — FCPE's depthwise separable convolution reduces compute ~k× vs. standard Conv
[source-code — FCPE paper arXiv 2509.15140]
The architectural choice is principled: k×C reduction in FLOPs per layer while maintaining
receptive field size. This explains FCPE's 77× speed advantage over CREPE without proportional
accuracy loss.
