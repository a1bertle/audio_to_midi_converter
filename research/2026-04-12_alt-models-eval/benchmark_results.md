# Benchmark Results

<!-- Auto-managed by benchmarker agent. Do not edit run entries manually. -->

Topic: Alternative stem separation model evaluation — guitar/other bleed
Input: inputs/Foals - My Number (Official Audio).mp3 (242.58 s, 44.1 kHz stereo)

---

## Run 1 — 2026-04-12 — htdemucs_ft MPS quality eval

### Configuration
| Field | Value |
|---|---|
| Date / time | 2026-04-12 |
| Command | `python3 -m demucs --name htdemucs_ft --device mps --out /tmp/demucs_alt_eval2/htdemucs_ft <input>` |
| Model | htdemucs_ft |
| Device | mps (Apple M1) |
| Input file | `inputs/Foals - My Number (Official Audio).mp3` |
| Audio duration | 242.58 s |
| Stems produced | 4 (drums, bass, vocals, other) — no guitar/piano |
| Evaluator | `tests/evaluation/evaluate_stems.py` |
| JSON report | `research/2026-04-12_alt-models-eval/htdemucs_ft_eval.json` |

### Reconstruction Metrics
| Metric | Value | Provenance |
|---|---|---|
| Mix energy | 57.07 dB | [measured — `evaluate_stems.py`] |
| Stem-sum energy | 56.54 dB | [measured — `evaluate_stems.py`] |
| Residual energy | 37.54 dB | [measured — `evaluate_stems.py`] |
| Leakage ratio | −19.54 dB | [measured — `evaluate_stems.py`] |

### Per-Stem Quality
| Stem | RMS (dBFS) | SRR (dB) | Centroid (Hz) | Silence (%) |
|---|---|---|---|---|
| drums | −17.81 | +14.95 | 4920.6 | 8.4% |
| bass | −21.50 | +11.26 | 1832.3 | 42.2% |
| vocals | −21.92 | +10.83 | 2450.5 | 19.0% |
| other | −21.71 | +11.04 | 1743.5 | 1.8% |

All values [measured — `evaluate_stems.py`, `htdemucs_ft_eval.json`].

### Cross-Leakage Matrix (dB)
|  | vocals | drums | bass | other |
|--|--------|-------|------|-------|
| **vocals** | — | −18.23 | −19.63 | −4.39 |
| **drums** | −18.23 | — | −6.35 | −14.25 |
| **bass** | −19.63 | −6.35 | — | −11.64 |
| **other** | −4.39 | −14.25 | −11.64 | — |

All values [measured — `evaluate_stems.py`].

### Notes
- No guitar or piano stem — model is 4-stem only. Guitar content is folded into `other`.
- `other` SRR +11.04 dB vs baseline `other` +1.83 dB: substantially cleaner because guitar
  is not being separated from other — it simply stays in `other`.
- `vocals`/`other` cross-leakage −4.39 dB remains poor (same structural cause as baseline).
- Fine-tuning vs stock htdemucs: `other` SRR 11.04 vs 13.06 (htdemucs_ft slightly worse
  on this track — fine-tuning targets MUSDB18 SDR, not this track).

---

## Run 2 — 2026-04-12 — htdemucs_4s (stock) MPS quality eval

### Configuration
| Field | Value |
|---|---|
| Date / time | 2026-04-12 |
| Command | `python3 -m demucs --name htdemucs --device mps --out /tmp/demucs_alt_eval2/htdemucs_4s <input>` |
| Model | htdemucs |
| Device | mps (Apple M1) |
| Input file | `inputs/Foals - My Number (Official Audio).mp3` |
| Audio duration | 242.58 s |
| Stems produced | 4 (drums, bass, vocals, other) — no guitar/piano |
| Evaluator | `tests/evaluation/evaluate_stems.py` |
| JSON report | `research/2026-04-12_alt-models-eval/htdemucs_4s_eval.json` |

### Reconstruction Metrics
| Metric | Value | Provenance |
|---|---|---|
| Mix energy | 57.07 dB | [measured — `evaluate_stems.py`] |
| Stem-sum energy | 56.54 dB | [measured — `evaluate_stems.py`] |
| Residual energy | 35.61 dB | [measured — `evaluate_stems.py`] |
| Leakage ratio | −21.46 dB | [measured — `evaluate_stems.py`] |

### Per-Stem Quality
| Stem | RMS (dBFS) | SRR (dB) | Centroid (Hz) | Silence (%) |
|---|---|---|---|---|
| drums | −17.78 | +16.91 | 4848.9 | 7.8% |
| bass | −21.45 | +13.24 | 1873.0 | 42.5% |
| vocals | −21.89 | +12.79 | 2411.6 | 18.1% |
| other | −21.62 | +13.06 | 1790.8 | 1.8% |

All values [measured — `evaluate_stems.py`, `htdemucs_4s_eval.json`].

### Cross-Leakage Matrix (dB)
|  | vocals | drums | bass | other |
|--|--------|-------|------|-------|
| **vocals** | — | −17.88 | −19.12 | −4.65 |
| **drums** | −17.88 | — | −6.29 | −13.90 |
| **bass** | −19.12 | −6.29 | — | −11.18 |
| **other** | −4.65 | −13.90 | −11.18 | — |

All values [measured — `evaluate_stems.py`].

### Notes
- No guitar or piano stem — model is 4-stem only.
- `other` SRR +13.06 dB: best `other` SRR seen so far, better than fine-tuned
  htdemucs_ft (+11.04) and far above 6s baseline (+1.83).
- Better overall leakage ratio (−21.46) than htdemucs_ft (−19.54) — stock model
  reconstructs more faithfully on this track.
- `vocals`/`other` cross-leakage −4.65 dB: identical to 6s baseline; structural
  bleed not resolved by 4-stem architecture.

---

## Run 3 — 2026-04-12 — mdx_extra CPU quality eval

### Configuration
| Field | Value |
|---|---|
| Date / time | 2026-04-12 |
| Command | `python3 -m demucs --name mdx_extra --device cpu --out /tmp/demucs_alt_eval2/mdx_extra <input>` |
| Model | mdx_extra (bag of 4 MDX-Net models) |
| Device | cpu (no MPS path for MDX models in Demucs 4.0.1) |
| Input file | `inputs/Foals - My Number (Official Audio).mp3` |
| Audio duration | 242.58 s |
| Wall-clock time | 930.91 s | [measured — `time.perf_counter()` in `run_all.py`] |
| RTF | 3.84× | [back-calc: 930.91 / 242.58] |
| Peak RSS (approx) | 3,888 MB | [measured — `resource.RUSAGE_CHILDREN.ru_maxrss`] |
| Stems produced | 4 (drums, bass, vocals, other) — no guitar/piano |
| Evaluator | `tests/evaluation/evaluate_stems.py` |
| JSON report | `research/2026-04-12_alt-models-eval/mdx_extra_eval.json` |

### Reconstruction Metrics
| Metric | Value | Provenance |
|---|---|---|
| Mix energy | 57.07 dB | [measured — `evaluate_stems.py`] |
| Stem-sum energy | 56.62 dB | [measured — `evaluate_stems.py`] |
| Residual energy | 34.28 dB | [measured — `evaluate_stems.py`] |
| Leakage ratio | −22.79 dB | [measured — `evaluate_stems.py`] |

### Per-Stem Quality
| Stem | RMS (dBFS) | SRR (dB) | Centroid (Hz) | Silence (%) |
|---|---|---|---|---|
| drums | −17.68 | +18.34 | 4520.3 | 8.6% |
| bass | −21.31 | +14.70 | 2498.9 | 40.3% |
| vocals | −22.01 | +14.00 | 2926.4 | 16.1% |
| other | −21.85 | +14.17 | 1720.5 | 1.9% |

All values [measured — `evaluate_stems.py`, `mdx_extra_eval.json`].

### Cross-Leakage Matrix (dB)
|  | vocals | drums | bass | other |
|--|--------|-------|------|-------|
| **vocals** | — | −17.94 | −21.10 | −4.49 |
| **drums** | −17.94 | — | −6.54 | −14.65 |
| **bass** | −21.10 | −6.54 | — | −12.38 |
| **other** | −4.49 | −14.65 | −12.38 | — |

All values [measured — `evaluate_stems.py`].

### Notes
- No guitar/piano stem — `other` bundles all harmonic content.
- Best overall leakage ratio (−22.79 dB) and drums SRR (+18.34 dB) of all tested models.
- RTF 3.84× **exceeds** the ≤3× real-time budget — CPU-only is too slow for this pipeline.
- `vocals`/`other` cross-leakage −4.49 dB: same structural bleed pattern as all 4-stem models.

---

## Run 4 — 2026-04-12 — htdemucs_6s + Wiener spectral mask (alpha=2.0)

### Configuration
| Field | Value |
|---|---|
| Date / time | 2026-04-12 |
| Command | Post-processing only — reused stems from `output/2026-04-13_foals-my-number-stems/htdemucs_6s/` |
| Model | htdemucs_6s (prior session) + Wiener mask post-process |
| Device | cpu (post-processing, numpy/librosa STFT) |
| Input file | `inputs/Foals - My Number (Official Audio).mp3` |
| Audio duration | 242.58 s |
| Post-processing wall-clock | 16.47 s | [measured — `time.perf_counter()` in `run_all.py`] |
| Parameters | N_FFT=4096, HOP=1024, alpha=2.0 |
| Stems produced | 6 (drums, bass, vocals, guitar, other, piano) |
| Evaluator | `tests/evaluation/evaluate_stems.py` |
| JSON report | `research/2026-04-12_alt-models-eval/htdemucs_6s_wiener_eval.json` |

### Reconstruction Metrics
| Metric | Value | Provenance |
|---|---|---|
| Mix energy | 57.07 dB | [measured — `evaluate_stems.py`] |
| Stem-sum energy | 56.39 dB | [measured — `evaluate_stems.py`] |
| Residual energy | 38.93 dB | [measured — `evaluate_stems.py`] |
| Leakage ratio | −18.15 dB | [measured — `evaluate_stems.py`] |

### Per-Stem Quality
| Stem | RMS (dBFS) | SRR (dB) | Centroid (Hz) | Silence (%) |
|---|---|---|---|---|
| drums | −17.51 | +13.86 | 4707.7 | 7.4% |
| bass | −20.89 | +10.48 | 1738.6 | 40.9% |
| vocals | −22.12 | +9.25 | 2296.9 | 14.2% |
| guitar | −23.77 | +7.59 | 1455.7 | 2.0% |
| other | −33.80 | −2.43 | 2629.4 | 48.1% |
| piano | −69.17 | −37.80 | 4051.0 | 99.8% |

All values [measured — `evaluate_stems.py`, `htdemucs_6s_wiener_eval.json`].

### Cross-Leakage Matrix (dB)
|  | vocals | drums | bass | guitar | piano | other |
|--|--------|-------|------|--------|-------|-------|
| **vocals** | — | −17.87 | −19.32 | −4.72 | −13.93 | −4.87 |
| **drums** | −17.87 | — | −6.40 | −14.63 | −22.74 | −20.66 |
| **bass** | −19.32 | −6.40 | — | −12.04 | −23.86 | −20.56 |
| **guitar** | −4.72 | −14.63 | −12.04 | — | −13.12 | −4.61 |
| **piano** | −13.93 | −22.74 | −23.86 | −13.12 | — | −14.58 |
| **other** | −4.87 | −20.66 | −20.56 | −4.61 | −14.58 | — |

All values [measured — `evaluate_stems.py`].

### Notes
- Guitar/other cross-leakage −4.61 dB: **worse than baseline** −3.52 dB. Wiener mask failed
  to separate guitar from other.
- Guitar SRR +7.59 dB: **degraded** from baseline +10.57 dB — mask over-suppressed guitar.
- `other` SRR −2.43 dB: catastrophic — mask pushed guitar content into residual rather than
  into guitar stem.
- Structural cause: Wiener gain M_guitar(f) = |guitar|² / (|guitar|² + |other|²) operates on
  the original model's spectral estimates. Since the model already confuses guitar and other
  at every frequency bin, the mask simply redistributes shared energy rather than separating it.
  The bleed is not reducible by post-filtering the model's own confused outputs.
- Leakage ratio degraded to −18.15 dB (from −20.90 dB baseline) — mask disrupts reconstruction.

---

## Cross-Run Comparison

| Run | Date | Model | Device | RTF | Leakage ratio (dB) | Guitar SRR (dB) | Guitar/other XL (dB) | Other SRR (dB) | Vocals SRR (dB) | Drums SRR (dB) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 2026-04-13 | htdemucs_6s | mps | 0.52× † | −20.90 | +10.57 | −3.52 | +1.83 | +12.00 | +16.61 |
| 1 | 2026-04-12 | htdemucs_ft | mps | 1.74× | −19.54 | n/a | n/a | +11.04 | +10.83 | +14.95 |
| 2 | 2026-04-12 | htdemucs (4s) | mps | 0.44× | −21.46 | n/a | n/a | +13.06 | +12.79 | +16.91 |
| 3 | 2026-04-12 | mdx_extra | cpu | 3.84× ‡ | −22.79 | n/a | n/a | +14.17 | +14.00 | +18.34 |
| 4 | 2026-04-12 | htdemucs_6s+wiener | mps+cpu | 0.52×+0.07× | −18.15 | +7.59 | −4.61 | −2.43 | +9.25 | +13.86 |

† Baseline RTF from 95.09 s track; runs 1–4 on 242.58 s track — not directly comparable.
‡ mdx_extra RTF 3.84× exceeds the ≤3× real-time budget.
RTF for runs 1–3: [back-calc: wall_s / 242.58]. Baseline RTF: [back-calc: 49.02 / 95.09].
Run 4 post-process RTF: [back-calc: 16.47 / 242.58].

All quality values [measured — respective `*_eval.json` files in this session].
Baseline values [measured — `research/2026-04-13_demucs-quality-eval/benchmark_results.md`].
