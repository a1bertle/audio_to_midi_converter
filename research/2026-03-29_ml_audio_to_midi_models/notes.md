# Notes: ML-based audio → MIDI candidates (local, macOS Apple Silicon)

Date: 2026-03-29
Topic: ML-based audio-to-MIDI transcription models suitable for integrating into this repository’s
Python pipeline (`audio2midi/`).

## Current repo state (baseline)
- This repo already integrates two ML backends:
  - `piano-transcription-inference` (`audio2midi/transcribers/piano_transcription_inference.py`)
  - `basic-pitch` with an ONNX preference (`audio2midi/transcribers/basic_pitch.py`)
- Output format: Standard MIDI (`.mid`) written via `mido`.
  - `piano-transcription-inference` can produce sustain pedal events (via its generated MIDI).
  - `basic-pitch` path currently maps only note events; pedals are empty.

## Candidates

### 1) `piano_transcription_inference` (PTI) — best-fit for piano + pedal
- What it does: high-resolution piano transcription with pedal support; produces MIDI output.
- Stack: PyTorch + librosa; checkpoint hosted on Zenodo.
- License: MIT for the `piano-transcription-inference` package; training code in ByteDance repo is
  Apache-2.0.
- Notes:
  - The upstream `PianoTranscription(device=...)` interface is documented as `cpu|cuda`; on M1 this
    typically implies CPU unless a fork adds MPS support.
  - Strong match for this repo’s “noisy piano + pedal” focus.

Sources:
- https://github.com/qiuqiangkong/piano_transcription_inference
- https://pypi.org/project/piano-transcription-inference/
- https://github.com/bytedance/piano_transcription

### 1b) `Transkun` — modern piano AMT (event-based / semi-CRF), pip-installable
- What it does: automatic piano transcription to MIDI (notes + velocity; pedal handling depends on
  checkpoint/config).
- Stack: PyTorch; ships as a pip package with a CLI (`transkun input.mp3 output.mid`).
- License: MIT.
- Notes:
  - Repo documentation explicitly notes the default shipped checkpoint is trained *without* sustain
    pedal extension of notes, which may or may not match your target MIDI convention.
  - Practical to integrate as a third backend because the repo already depends on `torch` and has a
    transcriber abstraction.
  - Worth evaluating if PTI compatibility/performance becomes problematic, or if you want to test a
    newer architecture against your real-world noisy YouTube audio.

Source:
- https://github.com/Yujia-Yan/Transkun

### 2) Spotify `basic-pitch` — best-fit for general polyphony, lightweight deploy
- What it does: instrument-agnostic polyphonic note transcription; can include pitch bends in its
  MIDI output.
- Stack: Python library; this repo already prefers ONNX artifacts when present, run via
  `onnxruntime` (avoids TF SavedModel issues).
- License: Apache-2.0.
- Notes:
  - Great “always works” fallback and good for guitar/other instruments.
  - No explicit sustain pedal modeling; if pedal is required, it’s not a drop-in replacement for PTI.

Sources:
- https://github.com/spotify/basic-pitch

### 3) Magenta `mt3` — strong multi-instrument research model, heavy integration cost
- What it does: multi-instrument automatic music transcription; uses T5X/JAX ecosystem.
- Stack: T5X/JAX; typically heavier install + runtime footprint than PTI/Basic Pitch.
- License: Apache-2.0 (repo).
- Notes:
  - Likely too heavyweight for a simple offline CLI unless it’s an optional “extra” backend.
  - Apple Silicon support depends on JAX wheels + compatibility; may be workable but is higher risk.

Source:
- https://github.com/magenta/mt3

### 3b) Sony `hFT-Transformer` — high-quality piano AMT research code (heavier)
- What it does: piano transcription model (research implementation).
- Stack: research codebase; packaging/inference ergonomics may be less turnkey than PTI/Transkun.
- License: see repo (verify before integration).
- Notes:
  - Potentially strong quality, but likely higher integration cost than Transkun/PTI due to fewer
    “pip install + simple inference API” affordances.

Source:
- https://github.com/sony/hFT-Transformer

### 4) Magenta “Onsets and Frames” (Python TF implementation) — conceptually relevant, repo status caveat
- What it does: dual-objective piano transcription (onsets + frames).
- Stack: Magenta Python code (TensorFlow).
- License: Apache-2.0 for Magenta.
- Notes:
  - The main `magenta/magenta` repo is archived (read-only as of 2026-01-06), which increases
    maintenance risk for new integrations.
  - The JS implementation (`@magenta/music`) exists, but that’s a Node/TFJS path rather than Python.

Sources:
- https://github.com/magenta/magenta (archived)
- https://magenta.tensorflow.org/onsets-frames

### 5) `omnizart` — broad AMT toolbox, but not a good match for ARM macOS
- What it does: multiple AMT tasks (music notes, drums, chords, beat, vocal).
- Stack: heavy dependencies; provides checkpoints and a CLI.
- License: MIT.
- Notes:
  - Project documentation explicitly calls out ARM-based macOS compatibility issues, making it a
    poor default candidate for this repo’s target machine.

Sources:
- https://github.com/Music-and-Culture-Technology-Lab/omnizart
- https://music-and-culture-technology-lab.github.io/omnizart-doc/index.html

## Quick comparison (practical integration lens)

| Candidate | Best for | Pedals | Typical runtime stack | Integration risk on M1 | License |
|---|---|---:|---|---|---|
| PTI (`piano_transcription_inference`) | Solo piano, noisy recordings | Yes | PyTorch | Low–Med | MIT |
| Transkun | Solo piano (newer AMT) | Checkpoint-dependent | PyTorch | Low–Med | MIT |
| Basic Pitch | General polyphony (incl. guitar) | No | ONNXRuntime (preferred) | Low | Apache-2.0 |
| MT3 | Multi-instrument transcription | Model-dependent | JAX/T5X | Med–High | Apache-2.0 |
| Magenta O&F (Python) | Piano transcription | No/partial (varies) | TensorFlow | Med + archived | Apache-2.0 |
| Omnizart | Multi-task AMT | Varies | Heavy deps | High (ARM macOS issues) | MIT |

## Preliminary recommendation
1) Keep PTI as the primary “piano + pedal” backend and invest in small robustness/perf improvements
   around it (caching, chunking, optional acceleration where possible).
2) Keep Basic Pitch as the cross-instrument fallback; consider extending mapping to retain pitch bends
   (if desired) and expose relevant thresholds in CLI for tuning.
3) Treat MT3 as an experimental/optional backend only (separate extras + clear “heavy install” docs).
4) Avoid new reliance on archived Magenta Python O&F unless absolutely necessary.
5) Do not pursue Omnizart for this repo’s default flow on Apple Silicon.
