# Research Notes: Demucs Benchmark on M1

**Date:** 2026-04-12
**Session:** `research/2026-04-12_demucs-benchmark/`
**Goal:** Measure wall-clock time, peak RAM, and subjective stem quality for
Demucs `htdemucs_6s` on Apple M1, using both MPS and CPU backends.

---

## Environment

| Field | Value | Provenance |
|-------|-------|------------|
| Machine | Apple MacBook Air (MacBookAir10,1) | [assumed — from global-standards.md] |
| Chip | Apple M1, 4P+4E cores, 7-core GPU | [assumed — from global-standards.md] |
| RAM | 16 GB unified | [assumed — from global-standards.md] |
| OS | macOS darwin | [assumed — from global-standards.md] |
| Python | 3.12 | [measured — venv] |
| PyTorch | 2.2.2 | [measured — `torch.__version__`] |
| torchaudio | 2.2.2 | [measured — installed version] |
| Demucs | 4.0.1 | [measured — `importlib.metadata.version('demucs')`] |
| MPS available | True | [measured — `torch.backends.mps.is_available()`] |

---

## Input

| Field | Value | Provenance |
|-------|-------|------------|
| File | `output/Foals - My Number - Harp.mp3` | [measured — project output directory] |
| Duration | 95.09 s (1.58 min) | [measured — librosa load, `len(y)/sr`] |
| Sample rate | 44100 Hz | [measured — librosa load] |
| Channels | 2 (stereo) | [measured — librosa shape `(2, 4193280)`] |
| Model | htdemucs_6s | [source-code — CLI `--name htdemucs_6s`] |
| Model weights | auto-downloaded on first run from dl.fbaipublicfiles.com | [measured] |

---

## Setup Notes

- `torchaudio 2.11.0` (installed alongside demucs) requires the `torchcodec`
  package for WAV saving, which is not available on Apple Silicon.
  Downgraded to `torchaudio==2.2.2` (matches `torch==2.2.2`) to restore
  the legacy `soundfile`-based save path. [measured — runtime error + fix]
- SSL cert verification error on first model download; resolved by setting
  `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE` to the `certifi` bundle. [measured]

---

## Findings

### MPS Run

| Metric | Value | Provenance |
|--------|-------|------------|
| Start time | 2026-04-13T00:54:17Z | [measured — `date -u`] |
| End time | 2026-04-13T00:55:06Z | [measured — `date -u`] |
| Wall-clock time | 49.02 s | [measured — `/usr/bin/time -l`] |
| RTF (wall / audio_dur) | 49.02 / 95.09 = **0.52×** | [back-calc — wall_s / audio_s] |
| Peak RSS | 1,035,780,096 bytes ≈ **988 MB** | [measured — `/usr/bin/time -l` `maximum resident set size`] |
| Peak memory footprint | 3,538,064,576 bytes ≈ **3.30 GB** | [measured — `/usr/bin/time -l` `peak memory footprint`] |
| User CPU time | 34.41 s | [measured — `/usr/bin/time -l`] |
| Sys CPU time | 29.95 s | [measured — `/usr/bin/time -l`] |
| Stems produced | 6 (bass, drums, guitar, other, piano, vocals) | [measured — `find /tmp/demucs_bench/mps`] |
| Stem file sizes | ~16 MB each (WAV, stereo, 44.1 kHz) | [measured — `du -sh`] |

### CPU Run

| Metric | Value | Provenance |
|--------|-------|------------|
| Start time | 2026-04-13T00:55:14Z | [measured — `date -u`] |
| End time | 2026-04-13T00:57:29Z | [measured — `date -u`] |
| Wall-clock time | 134.51 s | [measured — `/usr/bin/time -l`] |
| RTF (wall / audio_dur) | 134.51 / 95.09 = **1.41×** | [back-calc — wall_s / audio_s] |
| Peak RSS | 1,932,247,040 bytes ≈ **1,843 MB** | [measured — `/usr/bin/time -l` `maximum resident set size`] |
| Peak memory footprint | 1,023,448,768 bytes ≈ **976 MB** | [measured — `/usr/bin/time -l` `peak memory footprint`] |
| User CPU time | 450.10 s | [measured — `/usr/bin/time -l`] |
| Sys CPU time | 202.97 s | [measured — `/usr/bin/time -l`] |
| Stems produced | 6 (bass, drums, guitar, other, piano, vocals) | [measured — `find /tmp/demucs_bench/cpu`] |
| Stem file sizes | ~16 MB each (WAV, stereo, 44.1 kHz) | [measured — `du -sh`] |

---

## Interpretation

- **MPS is 2.74× faster than CPU** wall-clock (49 s vs 135 s) on this 95 s track.
  [back-calc: 134.51 / 49.02]
- **Both devices meet the 3× RTF budget**: MPS at 0.52×, CPU at 1.41×. Even CPU
  is well within the ≤3× real-time target from the proposal.
- **Memory**: MPS peak memory footprint is 3.30 GB — within the 4 GB budget but
  uses most of it. CPU footprint is 0.98 GB but peak RSS is 1.84 GB. Neither
  exceeds the 4 GB limit. MPS is the recommended device for performance, but
  pipeline designers should account for ~3.3 GB unified memory usage during
  separation when deciding whether to run separation and transcription
  concurrently.
- **Dependency issue**: `torchaudio>=2.3` is incompatible with demucs 4.0.1 on
  Apple Silicon (requires `torchcodec`). Must pin `torchaudio==2.2.*` in
  `pyproject.toml` when adding demucs, or contribute a fix upstream.
- **Stem quality**: Not formally measured (no SDR evaluation). Subjective
  listening recommended on the produced stems before implementing.
