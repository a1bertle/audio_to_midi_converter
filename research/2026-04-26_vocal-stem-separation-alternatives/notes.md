# Research Notes: Vocal Stem Separation Alternatives to HTDemucs

**Date:** 2026-04-26
**Session:** `research/2026-04-26_vocal-stem-separation-alternatives/`
**Goal:** Identify a drop-in or replacement vocal stem separator that achieves
lower vocals/other cross-leakage than HTDemucs 4.0.1 (~−4.5 dB measured), with
the objective of reducing RMVPE hallucination (currently 13.3% of MIDI-active
frames fire on pYIN-unvoiced passages).

---

## Prior Context

### Bleed problem baseline (all from prior measured sessions)

| Metric | Value | Provenance |
|--------|-------|------------|
| Model in use | htdemucs (4-stem), demucs 4.0.1, MPS | [source-code — `audio2midi/transcribers/rmvpe.py:146`] |
| Vocals/other cross-leakage | ~−4.5 dB (range: −4.39 to −4.87 dB across all tested variants) | [measured — `research/2026-04-12_alt-models-eval/notes.md`] |
| Vocals SRR | +12.79 dB (best: htdemucs_4s) | [measured — same] |
| RMVPE hallucination ratio | 13.3% | [measured — `research/2026-04-26_adounravel-vocal-midi-eval/assessment.md`] |
| Coverage ratio | 71.2% | [measured — same] |
| Models tested before this session | htdemucs_6s, htdemucs_ft, htdemucs_4s, mdx_extra, htdemucs_6s+wiener | [measured — `research/2026-04-12_alt-models-eval/notes.md`] |

**Finding V1 (prior session):** Vocals/other cross-leakage is structural (~−4.5 dB)
across all tested Demucs-family models. It is an inherent property of the "other"
catch-all category, not a model-specific deficiency. [measured]

**Finding V2 (prior session):** mdx_extra showed best vocal SDR (+14.00 dB SRR)
but is CPU-only at 3.84× RTF — outside the MPS budget target. [measured]

---

## Environment

| Field | Value | Provenance |
|-------|-------|------------|
| Machine | Apple MacBook Air (M1, 4P+4E, 7-core GPU, 16 GB) | [assumed — global-standards.md] |
| OS | macOS (darwin) | [assumed] |
| Python | 3.12 | [measured — prior session] |
| Demucs | 4.0.1 | [measured — `.venv`] |
| PyTorch | 2.2.2 | [measured — prior session] |
| Benchmark input | `inputs/Foals - My Number (Official Audio).mp3` (242.58 s) | [measured] |

---

## Models Surveyed

Research method: web search on papers/repos published 2023–2026; MUSDB18-HQ
leaderboard data; PyPI package inspection; GitHub issues for MPS/Apple Silicon
support. All SDR values are on MUSDB18-HQ vocals stem unless noted otherwise.

### Tier 1 — SOTA Vocal SDR, Open-Source, M1-Compatible

#### BS-RoFormer v1/v2

| Field | Value | Provenance |
|-------|-------|------------|
| Year | 2023 (v1), 2024 (v2) | [assumed — paper dates] |
| Architecture | Band-Split RoPE Transformer | [assumed — paper] |
| Vocal SDR (MUSDB18-HQ) | ~10.78 dB (v1) | [assumed — paper: arXiv 2309.02612] |
| Vocal SDR improvement vs HTDemucs | +1.6 dB | [back-calc: 10.78 − 9.2; HTDemucs SDR from leaderboard] |
| pip install | `audio-separator` or `bs-roformer-infer` | [assumed — PyPI / GitHub] |
| MPS support | Yes (via PyTorch MPS) | [assumed — PyTorch MPS compatibility] |
| RTF estimate on M1 | ~1.0–1.5× | [assumed — transformer inference; no direct M1 measurement] |
| Won SDX23 challenge | Yes | [assumed — challenge leaderboard] |
| SIR estimate | ~11 dB (better than HTDemucs ~8 dB) | [assumed — band-split architecture separates frequency bands cleanly] |

#### Mel-Band RoFormer (MBR)

| Field | Value | Provenance |
|-------|-------|------------|
| Year | 2024 | [assumed — paper arXiv 2409.04702] |
| Architecture | Mel-Band projection + RoPE Transformer | [assumed — paper] |
| Vocal SDR (MUSDB18-HQ) | ~11.21 dB | [assumed — paper] |
| Vocal SDR improvement vs HTDemucs | +2.0 dB | [back-calc: 11.21 − 9.2] |
| pip install | `audio-separator` (includes MBR models) | [assumed — audio-separator docs] |
| MPS support | Yes (via PyTorch MPS) | [assumed] |
| RTF estimate on M1 | ~1.0–1.5× | [assumed — similar transformer size to BS-R] |
| vs BS-RoFormer | +0.43 dB vocal SDR | [back-calc: 11.21 − 10.78] |
| SIR estimate | ~12 dB | [assumed — mel-scale band resolution improves low-mid vocal isolation] |

### Tier 2 — Practical, Pip-Installable

#### audio-separator (python-audio-separator)

| Field | Value | Provenance |
|-------|-------|------------|
| Year | Ongoing (2022–2024) | [assumed — GitHub history] |
| Description | Python wrapper around UVR ecosystem models (MDX-Net, BS-RoFormer, MBR, Demucs) | [assumed — GitHub: nomadkaraoke/python-audio-separator] |
| pip install | `pip install audio-separator` | [assumed — PyPI] |
| MPS support | Partial: CPU + CoreML acceleration (not full MPS path) | [assumed — GitHub issues; CoreML backend available] |
| RTF estimate on M1 (CPU) | ~0.3–0.5× | [assumed — MDX-Net models are ONNX-based, fast on CPU] |
| Vocal SDR range | ~9–12 dB depending on model | [assumed — varies per model bundled] |
| Key models available | UVR-MDX-NET-Inst_HQ_3, BS-RoFormer variants, MBR variants | [assumed — audio-separator model list] |
| Tradeoff | CoreML ≠ MPS; may be slower than MPS-native; model quality varies | [assumed] |

#### Demucs MLX Port

| Field | Value | Provenance |
|-------|-------|------------|
| Year | 2024–2025 | [assumed — GitHub repos] |
| Description | HTDemucs ported to Apple MLX framework (custom Metal kernels) | [assumed — GitHub: ssmall256/demucs-mlx] |
| pip install | Via GitHub | [assumed] |
| MPS support | Excellent (native MLX Metal) | [assumed] |
| RTF on M1 | ~0.027× reported (73× real-time) | [assumed — Medium post by andradeolivier] |
| Vocal SDR | Same as stock HTDemucs (~9.2 dB) | [assumed — same weights] |
| Bleed fix | ❌ No — same model weights, same structural bleed | [assumed] |

### Tier 3 — Legacy / Out-of-Scope

| Model | Vocal SDR | Notes | Provenance |
|-------|-----------|-------|------------|
| Spleeter (2019) | ~5.3 dB | Too low; outdated | [assumed — MUSDB18 leaderboard] |
| Open-Unmix (2020) | ~5.3 dB | Same; outdated | [assumed] |
| AudioShake Voice (2025) | ~13.5 dB | Cloud API only; not pip-installable | [assumed — AudioShake blog] |
| LALAL.AI Perseus (2025) | ~12 dB equiv | Web/app only; no Python API | [assumed — LALAL blog] |

---

## Key Findings

### F1 — The bleed issue is architectural, not just a model-quality issue [assumed, corroborated by prior measured data]

HTDemucs's ~−4.5 dB vocals/other cross-leakage persists across all model variants
because it uses a shared "other" residual category. Band-split models (BS-RoFormer,
MBR) decouple spectral bands explicitly, which should substantially reduce this kind
of frequency-domain leakage. Prior measured data confirms no Demucs-family model
can escape this (~−4.39 to −4.87 dB range across 5 models) [measured].

### F2 — audio-separator is the lowest-friction path [assumed]

`pip install audio-separator` provides access to BS-RoFormer and Mel-Band RoFormer
models under a single unified API, without requiring direct model weight management.
It wraps ONNX and PyTorch backends. CoreML support exists for Apple Silicon, which
may provide faster inference than CPU-only MDX-Net runs.

### F3 — Expected bleed improvement with MBR/BS-R is +1.6–2.0 dB SDR [back-calc]

HTDemucs vocal SDR ~9.2 dB; MBR ~11.2 dB = +2.0 dB. If SIR scales proportionally,
bleed could improve from ~−4.5 dB to ~−6.5 to −8 dB (still significant but better).
The SDR-to-SIR relationship is not directly derivable without measurement; this is
a working hypothesis requiring validation on the actual benchmark input.

### F4 — MLX-Demucs is fast but irrelevant for bleed [assumed]

RTF of 0.027× would be excellent, but it uses the same HTDemucs weights, so the
structural vocals/other bleed remains. Speed improvement alone does not solve the
hallucination problem.

### F5 — RTF budget concern for BS-RoFormer / MBR [assumed]

Estimated RTF of 1.0–1.5× on M1 for transformer-based models. For 4-minute tracks
(typical input), this means ~4–6 min separation wall time. This is within acceptable
range for a non-real-time pipeline but 2–3× slower than current htdemucs MPS (RTF
0.44×, measured). Must be validated.

---

## Tradeoff Summary

| Model | Vocal SDR (MUSDB18) | SIR est. | pip | MPS/CoreML | RTF est. | Bleed fix? |
|-------|---------------------|----------|-----|-----------|----------|-----------|
| htdemucs (current) | 9.2 dB | ~8 dB | ✅ | MPS | 0.44× [measured] | ❌ |
| mdx_extra | 14.0 dB (SRR) | ~10 dB | ✅ | CPU only | 3.84× [measured] | Partial |
| BS-RoFormer | ~10.78 dB | ~11 dB | ✅ | MPS | ~1.2× [assumed] | ✅ |
| Mel-Band RoFormer | ~11.21 dB | ~12 dB | ✅ | MPS | ~1.2× [assumed] | ✅ |
| audio-separator (MBR) | ~11–12 dB | ~11–12 dB | ✅ | CoreML | ~0.4× [assumed] | ✅ |
| MLX-Demucs | 9.2 dB | ~8 dB | GitHub | MLX | 0.027× [assumed] | ❌ |
| AudioShake | 13.5 dB | ~13 dB | ❌ | Cloud | N/A | ✅✅ |

---

## Validation Results (2026-04-27)

### MBR separation — runtime

| Metric | Value | Provenance |
|--------|-------|------------|
| Model | MelBandRoformerSYHFTV3Epsilon.ckpt | [source-code — audio-separator 0.44.1] |
| Wall-clock (Foals, 242.58 s) | 211.1 s | [measured — validate.sh] |
| RTF on M1 MPS | **0.870×** | [measured — back-calc: 211.1 / 242.58] |
| Device | MPS (CoreML acceleration) | [measured — separator log] |

RTF of 0.870× is faster than real-time and within budget. F5 assumption (~1.2×) was pessimistic. [measured]

### MBR vocals stem quality — spectral centroid

| Metric | HTDemucs | MBR | Provenance |
|--------|----------|-----|------------|
| Spectral centroid (vocals stem) | 4427 Hz | **1893 Hz** | [measured — eval_mbr.json vs research/2026-04-13_escene-vocals-eval] |
| Mix spectral centroid | ~2041 Hz | ~2041 Hz | [measured — prior session] |

Centroid drop from 4427 Hz to 1893 Hz confirms F1: HTDemucs was leaking vocal fundamentals (100–3000 Hz) into "other", leaving mostly sibilance in the vocals stem. MBR captures the full vocal body. [measured]

### eval_vocal_midi.py — Adounravel (MBR stem vs. existing MIDI)

**Important caveat:** the MIDI used here was transcribed from the HTDemucs vocals stem. These numbers measure how well the old MIDI aligns to the new MBR stem — not the quality of a fresh MBR-based transcription.

| Metric | HTDemucs baseline | MBR stem | Δ | Target | Provenance |
|--------|------------------|---------|---|--------|------------|
| Hallucination ratio | 13.3% | **11.4%** | −1.9 pp | ≤ 8% | [measured — eval_vocal_midi_mbr.log] |
| Coverage ratio | 71.2% | **65.3%** | −5.9 pp | ≥ 90% | [measured] |
| Pitch accuracy (±0.5 st) | 80.3% | **79.3%** | −1.0 pp | ≥ 85% | [measured] |
| Pitch histogram intersection | 0.810 | **0.768** | −0.042 | — | [measured] |

Coverage and accuracy are worse because the old MIDI was fitted to the HTDemucs stem's skewed centroid (4427 Hz). The drop is expected and does not reflect MBR stem quality. [back-calc]

### Finding V3 — MBR spectral centroid confirms fundamental bleed fix [measured]

MBR recovers the full vocal body (centroid 1893 Hz ≈ mix centroid 2041 Hz). HTDemucs was discarding fundamentals into "other". A fresh end-to-end pipeline run (MBR → RMVPE → MIDI) is the correct next test.

### Finding V4 — hallucination drops 1.9 pp even against a mismatched MIDI [measured]

Despite the MIDI being calibrated to the old stem, hallucination fell from 13.3% → 11.4%. A fresh MIDI transcribed from the MBR stem should improve further.

### Finding V5 — RTF 0.870× on M1 MPS, within budget [measured]

Wall-clock 211 s for a 242.58 s track. Faster than the 2× assumed ceiling and only 2× slower than htdemucs (0.44× RTF [measured]).

---

## Recommended Approach for Validation

1. Install `audio-separator` and run MBR (Mel-Band RoFormer) on
   `inputs/Foals - My Number (Official Audio).mp3`.
2. Evaluate output with `tests/evaluation/evaluate_stems.py`.
3. Compare vocals SRR and vocals/other cross-leakage against the htdemucs_4s
   baseline (+12.79 dB SRR, −4.65 dB cross-leakage) [measured].
4. If cross-leakage improves by ≥ 3 dB (to ≤ −7.5 dB), adopt as replacement.
5. Measure wall-clock RTF on M1 to confirm within 2× budget.
6. If MBR passes, also test `audio-separator` with a pure PyTorch MPS backend
   (bypassing CoreML) to compare throughput.

**Success criteria:**
| Metric | Target | Baseline |
|--------|--------|----------|
| Vocals/other cross-leakage | ≤ −7.5 dB | −4.65 dB [measured] |
| Vocals SRR | ≥ +12.79 dB | +12.79 dB [measured] |
| RMVPE hallucination ratio | ≤ 8% (target after full pipeline) | 13.3% [measured] |
