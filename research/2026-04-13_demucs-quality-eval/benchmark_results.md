# Benchmark Results

<!-- Auto-managed by benchmarker agent. Do not edit run entries manually. -->

Topic: Demucs htdemucs_6s stem separation quality — blind metrics
Input: inputs/Foals - My Number (Official Audio).mp3 (242.58 s, 44.1 kHz stereo)

---

## Run 1 — 2026-04-13 — htdemucs_6s MPS quality eval

### Configuration
| Field | Value |
|---|---|
| Date / time | 2026-04-13 |
| Command | `python3 tests/evaluation/evaluate_stems.py --mix "inputs/Foals - My Number (Official Audio).mp3" --stems-dir "output/2026-04-13_foals-my-number-stems/htdemucs_6s/Foals - My Number (Official Audio)" --json-out "output/2026-04-13_foals-my-number-stems/evaluation.json"` |
| Model | htdemucs_6s |
| Device (separation) | MPS (Apple M1) |
| Input file | `inputs/Foals - My Number (Official Audio).mp3` |
| Audio duration | 242.58 s |
| Evaluator | `tests/evaluation/evaluate_stems.py` |
| JSON report | `output/2026-04-13_foals-my-number-stems/evaluation.json` |

### Reconstruction Metrics
| Metric | Value | Provenance |
|---|---|---|
| Mix energy | 57.07 dB | [measured] |
| Stem-sum energy | 56.56 dB | [measured] |
| Residual energy | 36.18 dB | [measured] |
| Leakage ratio | −20.90 dB | [measured] |

### Per-Stem Quality
| Stem | RMS (dBFS) | SRR (dB) | Centroid (Hz) | Silence (%) |
|---|---|---|---|---|
| drums | −17.51 | +16.61 | 4707.7 | 7.4% |
| bass | −20.89 | +13.23 | 1738.6 | 40.9% |
| vocals | −22.12 | +12.00 | 2296.9 | 14.2% |
| guitar | −23.54 | +10.57 | 1449.9 | 2.0% |
| other | −32.29 | +1.83 | 2135.6 | 34.3% |
| piano | −69.17 | −35.05 | 4051.0 | 99.8% |

All values [measured — `evaluate_stems.py`].

### Cross-Leakage Matrix (dB)
|  | vocals | drums | bass | guitar | piano | other |
|--|--------|-------|------|--------|-------|-------|
| **vocals** | — | −17.87 | −19.32 | −4.65 | −13.93 | −4.48 |
| **drums** | −17.87 | — | −6.40 | −14.70 | −22.74 | −19.08 |
| **bass** | −19.32 | −6.40 | — | −12.16 | −23.86 | −19.08 |
| **guitar** | −4.65 | −14.70 | −12.16 | — | −13.05 | −3.52 |
| **piano** | −13.93 | −22.74 | −23.86 | −13.05 | — | −13.52 |
| **other** | −4.48 | −19.08 | −19.08 | −3.52 | −13.52 | — |

All values [measured — `evaluate_stems.py`].

### Notes
- `piano` stem is effectively silent (99.8% silence, SRR −35 dB) — track contains no piano.
- `guitar`/`other` cross-leakage at −3.52 dB is the worst pair; subjectively confirmed as audible bleed.
- `vocals`/`other` and `vocals`/`guitar` also problematic at −4.48 and −4.65 dB.
- `other` SRR of +1.83 dB is marginal — barely above the residual noise floor.

---

## Cross-Run Comparison

| Run | Date | Model | Track | Leakage ratio (dB) | Guitar SRR (dB) | Guitar/other XL (dB) | Vocals SRR (dB) | Drums SRR (dB) |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-04-13 | htdemucs_6s | Foals - My Number (rock) | −20.90 | +10.57 | −3.52 | +12.00 | +16.61 |
