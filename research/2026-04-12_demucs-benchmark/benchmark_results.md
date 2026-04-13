# Benchmark Results

<!-- Auto-managed by benchmarker agent. Do not edit run entries manually. -->

Topic: Demucs htdemucs_6s stem separation on Apple M1
Input: output/Foals - My Number - Harp.mp3 (95.09 s, 44.1 kHz stereo)

---

## Run 1 — 2026-04-13 — MPS

### Configuration
| Field | Value |
|---|---|
| Date / time | 2026-04-13T00:54:17Z |
| Command | `python3 -m demucs --name htdemucs_6s --device mps --out /tmp/demucs_bench/mps "<input>"` |
| Model | htdemucs_6s |
| Device | mps |
| Input file | `output/Foals - My Number - Harp.mp3` |
| Audio duration | 95.09 s |
| PyTorch | 2.2.2 |
| torchaudio | 2.2.2 |
| Demucs | 4.0.1 |

### Timings & Memory
| Metric | Value | Provenance |
|---|---|---|
| Wall-clock time | 49.02 s | [measured — `/usr/bin/time -l`] |
| User CPU time | 34.41 s | [measured] |
| Sys CPU time | 29.95 s | [measured] |
| Peak RSS | 988 MB (1,035,780,096 bytes) | [measured] |
| Peak memory footprint | 3,302 MB (3,538,064,576 bytes) | [measured] |

### Derived Metrics
| Metric | Value | Formula |
|---|---|---|
| RTF (real-time factor) | 0.52× | [back-calc: 49.02 / 95.09] |
| Speedup vs CPU | 2.74× | [back-calc: 134.51 / 49.02] |

### Output
| Stem | File size |
|---|---|
| bass.wav | ~16 MB |
| drums.wav | ~16 MB |
| guitar.wav | ~16 MB |
| other.wav | ~16 MB |
| piano.wav | ~16 MB |
| vocals.wav | ~16 MB |

### Notes
- torchaudio 2.11.0 (installed by demucs pip dep) requires torchcodec for WAV
  save, which is unavailable on Apple Silicon. Downgraded to torchaudio==2.2.2.
- SSL cert issue on model download; resolved via certifi env vars.
- An optical-flow interpolation workload was running concurrently on the GPU
  during this run. MPS timings may be inflated vs. isolated measurement.

---

## Run 2 — 2026-04-13 — CPU

### Configuration
| Field | Value |
|---|---|
| Date / time | 2026-04-13T00:55:14Z |
| Command | `python3 -m demucs --name htdemucs_6s --device cpu --out /tmp/demucs_bench/cpu "<input>"` |
| Model | htdemucs_6s |
| Device | cpu |
| Input file | `output/Foals - My Number - Harp.mp3` |
| Audio duration | 95.09 s |
| PyTorch | 2.2.2 |
| torchaudio | 2.2.2 |
| Demucs | 4.0.1 |

### Timings & Memory
| Metric | Value | Provenance |
|---|---|---|
| Wall-clock time | 134.51 s | [measured — `/usr/bin/time -l`] |
| User CPU time | 450.10 s | [measured] |
| Sys CPU time | 202.97 s | [measured] |
| Peak RSS | 1,843 MB (1,932,247,040 bytes) | [measured] |
| Peak memory footprint | 976 MB (1,023,448,768 bytes) | [measured] |

### Derived Metrics
| Metric | Value | Formula |
|---|---|---|
| RTF (real-time factor) | 1.41× | [back-calc: 134.51 / 95.09] |

### Output
| Stem | File size |
|---|---|
| bass.wav | ~16 MB |
| drums.wav | ~16 MB |
| guitar.wav | ~16 MB |
| other.wav | ~16 MB |
| piano.wav | ~16 MB |
| vocals.wav | ~16 MB |

### Notes
- Concurrent optical-flow interpolation workload on GPU. CPU workload should
  be unaffected, but system memory pressure may have impacted timing slightly.
- User+sys CPU time (653 s) >> wall-clock (134 s): confirms high parallelism
  across M1 CPU cores.

---

## Cross-Run Comparison

| Run | Date | Device | Audio dur (s) | Wall-clock (s) | RTF | Peak RSS (MB) | Peak mem footprint (MB) | Stems |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-04-13 | mps | 95.09 | 49.02 | 0.52× | 988 | 3,302 | 6 |
| 2 | 2026-04-13 | cpu | 95.09 | 134.51 | 1.41× | 1,843 | 976 | 6 |
