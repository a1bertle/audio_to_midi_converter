# Research Notes: Alternative Stem Separation Model Evaluation

**Date:** 2026-04-12
**Session:** `research/2026-04-12_alt-models-eval/`
**Goal:** Evaluate four candidate separation approaches against the htdemucs_6s
baseline, targeting guitar/other cross-leakage < −15 dB and guitar SRR > +15 dB
on the Foals benchmark track.

---

## Baseline (prior session)

| Metric | Value | Source |
|--------|-------|--------|
| Model | htdemucs_6s | [source-code — demucs CLI] |
| Guitar SRR | +10.57 dB | [measured — `research/2026-04-13_demucs-quality-eval/benchmark_results.md`] |
| Guitar/other cross-leakage | −3.52 dB | [measured — same] |
| Other SRR | +1.83 dB | [measured — same] |
| Leakage ratio | −20.90 dB | [measured — same] |
| Wall-clock (95.09 s track, MPS) | 49.02 s (RTF 0.52×) | [measured — `research/2026-04-12_demucs-benchmark/benchmark_results.md`] |
| Peak RSS (MPS) | 988 MB | [measured — same] |
| Peak memory footprint (MPS) | 3,302 MB | [measured — same] |

---

## Environment

| Field | Value | Provenance |
|-------|-------|------------|
| Machine | Apple MacBook Air (MacBookAir10,1) | [assumed — global-standards.md] |
| Chip | Apple M1, 4P+4E cores, 7-core GPU | [assumed — global-standards.md] |
| RAM | 16 GB unified | [assumed — global-standards.md] |
| Python | 3.12 | [measured — venv] |
| PyTorch | 2.2.2 | [measured — prior session] |
| torchaudio | 2.2.2 | [measured — prior session] |
| Demucs | 4.0.1 | [measured — prior session] |
| librosa | 0.11.0 | [measured — `librosa.__version__`] |
| soundfile | 0.13.1 | [measured — `soundfile.__version__`] |
| Benchmark input | `inputs/Foals - My Number (Official Audio).mp3` | [measured] |
| Input duration | 242.58 s | [measured — prior session] |

---

## Model Availability Check

Pre-run verification (2026-04-12):

| Model | Available | Sources | Notes |
|-------|-----------|---------|-------|
| htdemucs_6s | Yes (cached) | drums, bass, other, vocals, guitar, piano | 6-stem; baseline model |
| htdemucs | Yes (downloaded) | drums, bass, other, vocals | 4-stem stock |
| htdemucs_ft | Yes (downloaded) | drums, bass, other, vocals | 4-stem fine-tuned; **no 6-stem ft variant** |
| htdemucs_6s_ft | Not available | — | Does not exist in Demucs 4.0.1 [measured — `get_model` error] |
| mdx_extra | Yes (downloaded) | drums, bass, other, vocals | MDX-Net, 4-stem; CPU-only (no MPS backend) |

All values [measured — `demucs.pretrained.get_model()` probe, 2026-04-12].

**Key finding:** `htdemucs_6s_ft` does not exist. Fine-tuned variants are
4-stem only. The guitar/piano separation in htdemucs_6s cannot be improved
via a drop-in fine-tuned model — only approaches that modify the 6s outputs
or use post-processing are viable for guitar/other separation.

---

## Candidate Approaches Run

### Approach 1 — htdemucs_ft (fine-tuned 4-stem, MPS)
- No guitar stem; `other` bundles guitar+piano+synths.
- Evaluated to see if 4-stem separation quality (other SRR, overall leakage)
  differs from the 6s baseline.
- Expected: better drums/bass/vocals SRR (fine-tuning targets SDR on MUSDB18),
  but `other` still contains guitar.

### Approach 2 — htdemucs (stock 4-stem, MPS)
- Same stem set as htdemucs_ft.
- Provides baseline for measuring fine-tuning benefit.

### Approach 3 — mdx_extra (MDX-Net 4-stem, CPU)
- Alternative neural architecture to Demucs; trained on internal dataset.
- No MPS support in Demucs 4.0.1 [measured — demucs source; MDX models use
  ONNX runtime which lacks MPS path].
- CPU-only: expected higher wall-clock time.

### Approach 4 — htdemucs_6s + Wiener spectral masking (post-processing)
- Reuses existing 6s separation (`output/2026-04-13_foals-my-number-stems/`).
- Applies per-frequency Wiener gain with alpha=2.0 to sharpen guitar/other
  boundary:
  ```
  M_guitar(f,t) = |S_guitar(f,t)|^2 / (|S_guitar(f,t)|^2 + |S_other(f,t)|^2 + ε)
  M_other(f,t)  = |S_other(f,t)|^2  / (|S_guitar(f,t)|^2 + |S_other(f,t)|^2 + ε)
  ```
- N_FFT=4096, HOP=1024.
- Wall cost: post-processing only (separation is pre-computed).
- Risk: mask sharpening cannot recover information lost by the original model;
  if guitar energy is uniformly present in `other`, the mask will only
  redistribute energy, not suppress it.

---

## Measurement Method

- Separation: `python3 -m demucs --name <model> --device <device> --out <dir> <input>`
- Wall-clock: `time.perf_counter()` wrapper in `run_all.py`
- Quality evaluation: `tests/evaluation/evaluate_stems.py` (unchanged from
  prior session)
- Metrics: Guitar SRR (dB), Guitar/other cross-leakage (dB), Other SRR (dB),
  Leakage ratio (dB)
- Script: `research/2026-04-12_alt-models-eval/run_all.py`

---

## Results

See `benchmark_results.md` for full structured entries and `all_results.json`
for raw JSON.

### Summary table

| Approach | Guitar SRR (dB) | Guitar/other XL (dB) | Other SRR (dB) | RTF | Guitar stem? |
|---|---|---|---|---|---|
| htdemucs_6s (baseline) | +10.57 | −3.52 | +1.83 | 0.52× | Yes |
| htdemucs_ft (run 1) | n/a | n/a | +11.04 | 1.74× | No |
| htdemucs 4s (run 2) | n/a | n/a | +13.06 | 0.44× | No |
| mdx_extra (run 3) | n/a | n/a | +14.17 | 3.84× ‡ | No |
| htdemucs_6s+wiener (run 4) | +7.59 | −4.61 | −2.43 | ~0.59× | Yes (degraded) |

‡ Exceeds ≤3× real-time budget.
All values [measured — respective `*_eval.json` files].

---

## Findings

### Finding 1 — htdemucs_6s_ft does not exist
[measured — `demucs.pretrained.get_model('htdemucs_6s_ft')` raises error in Demucs 4.0.1]
Fine-tuned variants are 4-stem only. There is no drop-in fine-tuned replacement for
htdemucs_6s that preserves the guitar stem.

### Finding 2 — 4-stem models improve `other` SRR but eliminate guitar stem
[measured — runs 1, 2, 3]
By not separating guitar from other, all three 4-stem models score dramatically
better `other` SRR (+11–14 dB vs +1.83 dB baseline). This is not a separation
improvement — it is the absence of the guitar/other separation problem. Guitar
content remains mixed into `other`, making downstream guitar transcription
infeasible without a further separation step.

### Finding 3 — mdx_extra exceeds real-time budget on CPU
[measured — run 3: wall 930.91 s, RTF 3.84× on 242.58 s track]
mdx_extra has no MPS path in Demucs 4.0.1 (MDX models use ONNX runtime without
Apple Silicon GPU support). CPU-only operation at 3.84× RTF exceeds the ≤3× budget
and makes this model impractical for this pipeline.

### Finding 4 — Wiener spectral masking degraded guitar quality
[measured — run 4]
Guitar SRR fell from +10.57 dB (baseline) to +7.59 dB. Guitar/other cross-leakage
worsened from −3.52 dB to −4.61 dB. `other` SRR collapsed to −2.43 dB.
Root cause: the Wiener mask redistributes energy between guitar and other using the
model's own spectral estimates as priors. Since the model assigns overlapping spectral
weight to both stems at every frequency, the mask cannot recover separation — it only
reshapes the energy distribution, amplifying reconstruction error.

### Finding 5 — vocals/other bleed is structural across all models
[measured — all runs: vocals/other XL −4.39 to −4.87 dB]
Every tested model, regardless of architecture, shows ~−4.5 dB vocals/other
cross-leakage. This is a property of the "other" category definition (vocals
bleed into a catch-all residual), not a model-specific deficiency.

### Conclusion
None of the four tested approaches meet both success criteria (guitar SRR > +15 dB
and guitar/other XL < −15 dB) on the Foals benchmark track. The guitar/other
separation problem is not solvable by:
- Switching to a fine-tuned 4-stem model (eliminates guitar stem)
- Switching to mdx_extra (no guitar stem; CPU-only; over budget)
- Post-filtering with Wiener masking (degrades quality)

### Session redirect (2026-04-12)
After reviewing findings, the downstream use case (MIDI transcription) has least
interest in the guitar stem. The more relevant stems for transcription are
**vocals** and **drums**. The research framing (guitar/other bleed as primary
success criterion) was misaligned with actual pipeline priorities.

The measured data already covers vocals and drums quality across all runs — see
the per-stem SRR and cross-leakage matrices in benchmark_results.md. A follow-on
research session should re-evaluate model selection against vocals SRR and
drums SRR as the primary success criteria. Session closed.
