# Proposal: Stem Separation Quality Improvement — Guitar/Other Bleed

**Date:** 2026-04-12
**Status:** Closed — session redirected (see below)

---

## Problem Statement

`htdemucs_6s` (Demucs v4.0.1, MPS on Apple M1) produces acceptable overall
reconstruction fidelity (leakage ratio −20.90 dB) but exhibits severe
spectral bleed between the `guitar` and `other` stems: cross-leakage −3.52 dB
(≈67% shared spectral energy) and a guitar SRR of only +10.57 dB. The `other`
stem SRR is +1.83 dB — barely above the residual noise floor. This bleed
degrades downstream guitar transcription quality.

Baseline source: `research/2026-04-13_demucs-quality-eval/benchmark_results.md`
and `notes.md`.

---

## Success Criteria

- Guitar/other cross-leakage < −15 dB on the Foals benchmark track.
- Guitar SRR > +15 dB on the Foals benchmark track.
- Wall-clock separation time ≤ 3× real-time on M1 MPS (≤ ~12 min for a 4 min
  track).
- Peak memory ≤ 4 GB (within M1 16 GB unified memory budget).
- No new framework dependency that conflicts with existing `torch>=2.2` stack.

---

## Constraints

- Platform: Apple M1, 7-core GPU, 16 GB unified RAM, macOS (darwin).
- Existing Demucs v4.0.1, PyTorch 2.2.2, torchaudio 2.2.2 installed.
- GPL-3.0-compatible licence required.
- Benchmark input: `inputs/Foals - My Number (Official Audio).mp3` (242.58 s).
- Evaluator: `tests/evaluation/evaluate_stems.py` (existing script, unchanged).

---

## Non-Goals

- Retraining or fine-tuning any model.
- Improving vocals, bass, or drums separation (secondary stems for this pipeline).
- Formal MUSDB18 SDR evaluation (blind test on proprietary benchmark, not
  available on this machine).
- Real-time or low-latency operation.

---

## Candidate Approaches

| # | Approach | Summary |
|---|----------|---------|
| 1 | **htdemucs_4s + residual subtraction** | Run the 4-stem model (drums, bass, vocals, other), subtract known stems from mix to form a cleaner guitar/other residual; avoids the 6s model's guitar–other confusion |
| 2 | **Open-Unmix (umxl / umx)** | PyTorch 4-stem model (drums, bass, vocals, other); independent architecture from Demucs; `other` cross-leakage may differ; lighter weight |
| 3 | **htdemucs_6s + Wiener/spectral masking post-processing** | Keep current model, apply a per-frequency Wiener gain or hard mask on the guitar/other pair to sharpen boundaries using the model's own output as prior |
| 4 | **htdemucs_ft (fine-tuned variant)** | Demucs ships `htdemucs_ft` fine-tuned for SDR; may improve guitar/other separation without changing pipeline |

---

## Research Findings

*To be populated by researcher sub-sessions below.*

### Research Session 1 — htdemucs_ft quality eval
`research/2026-04-12_alt-models-eval/`

### Research Session 2 — Open-Unmix quality eval
`research/2026-04-12_alt-models-eval/`

### Research Session 3 — htdemucs_4s residual subtraction
`research/2026-04-12_alt-models-eval/`

### Research Session 4 — htdemucs_6s spectral masking post-processing
`research/2026-04-12_alt-models-eval/`

---

## Results Summary

*To be populated after research sessions complete.*

| Approach | Guitar SRR (dB) | Guitar/other XL (dB) | Other SRR (dB) | Wall-clock RTF | Peak mem (MB) | Notes |
|----------|----------------|----------------------|----------------|----------------|---------------|-------|
| htdemucs_6s (baseline) | +10.57 [measured] | −3.52 [measured] | +1.83 [measured] | 0.52× [measured] | 988 RSS / 3302 footprint [measured] | `research/2026-04-13_demucs-quality-eval/` |
| htdemucs_ft | | | | | | |
| htdemucs_4s + residual | | | | | | |
| Open-Unmix (umxl) | | | | | | |
| htdemucs_6s + spec. mask | | | | | | |

---

## Recommendation

*Pending research results.*

---

## Tradeoff Analysis

*Pending research results.*

---

## Risks and Open Questions

- **[assumed]** `htdemucs_ft` ships fine-tuned weights for the same 4-stem
  heads; the 6-stem variant (`htdemucs_6s_ft`) may not exist — need to verify
  what Demucs 4.0.1 exposes.
- **[assumed]** Residual subtraction (approach 1) may amplify reconstruction
  error from the 4-stem model rather than reducing bleed; quality depends on
  bass/drums/vocals SRR being high.
- **[assumed]** Open-Unmix `other` stem bundles guitar+piano+other; without a
  dedicated guitar head, downstream guitar transcription would still face
  polyphony.
- **[assumed]** Spectral masking (approach 3) is a zero-cost post-process but
  risks over-suppressing frequencies shared by guitar and vocals (e.g. upper
  harmonics 2–5 kHz).

---

## Visualizations

*To be added after top candidates identified.*

---

## Research Results (session 2026-04-12_alt-models-eval)

All four candidates were benchmarked. Summary:

| Approach | Guitar SRR (dB) | Guitar/other XL (dB) | Other SRR (dB) | RTF | Guitar stem? |
|---|---|---|---|---|---|
| htdemucs_6s (baseline) | +10.57 [measured] | −3.52 [measured] | +1.83 [measured] | 0.52× [measured] | Yes |
| htdemucs_ft | n/a | n/a | +11.04 [measured] | 1.74× [measured] | No |
| htdemucs 4s | n/a | n/a | +13.06 [measured] | 0.44× [measured] | No |
| mdx_extra | n/a | n/a | +14.17 [measured] | 3.84× [measured] ‡ | No |
| htdemucs_6s + Wiener mask | +7.59 [measured] | −4.61 [measured] | −2.43 [measured] | ~0.59× [measured] | Yes (degraded) |

‡ Exceeds ≤3× real-time budget (CPU-only, no MPS path).

Source: `research/2026-04-12_alt-models-eval/benchmark_results.md`

**None of the four approaches met the success criteria.**

## Session Redirect

After completing the benchmark, the downstream use case was clarified: MIDI
transcription has **least interest in the guitar stem**. The primary stems of
interest are **vocals** and **drums**.

The measured data covers vocals and drums SRR and cross-leakage across all runs
and is available in `research/2026-04-12_alt-models-eval/benchmark_results.md`
for a follow-on analysis. The guitar/other success criteria are retired.

## Vocals and Drums Re-analysis

Re-analysed from existing eval JSONs (no additional separations). Full tables
in `research/2026-04-12_alt-models-eval/notes.md`.

| Model | Vocals SRR (dB) | Drums SRR (dB) | RTF |
|---|---|---|---|
| htdemucs_6s (baseline) | +12.00 [measured] | +16.61 [measured] | 0.52× [measured] |
| htdemucs_ft | +10.83 [measured] | +14.95 [measured] | 1.74× [measured] |
| **htdemucs_4s** | **+12.79** [measured] | **+16.91** [measured] | **0.44×** [measured] |
| mdx_extra | +14.00 [measured] | +18.34 [measured] | 3.84× [measured] ‡ |
| htdemucs_6s+wiener | +9.25 [measured] | +13.86 [measured] | ~0.59× [measured] |

‡ Exceeds ≤3× real-time budget.

## Recommendation

**htdemucs_4s** — stock 4-stem model, MPS device.

Best within-budget model for both vocals (+12.79 dB SRR) and drums (+16.91 dB SRR).
Faster than the current 6s baseline (RTF 0.44× vs 0.52×). The 6s model offers
no advantage for vocals or drums transcription.

## Next Step

Invoke **planner** with: "Replace htdemucs_6s with htdemucs (4-stem) as the
default separation model in the stem splitter pipeline, targeting vocals and
drums transcription. Expose --model flag to allow override."
