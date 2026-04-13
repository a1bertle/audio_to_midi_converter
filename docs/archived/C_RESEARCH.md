# Audio to MIDI Research and Solution Selection

## Objective
Select an algorithm and implementation strategy for converting noisy YouTube piano performances into MIDI in a C++ pipeline.

## Problem Characteristics
- Input is solo piano from YouTube videos, not studio-clean stems.
- Audio may include room noise, audience noise, and reverb.
- Output must be polyphonic note events and sustain pedal behavior, not only pitch contours.
- Pipeline must support automated YouTube download and offline batch processing.

## Evaluation Criteria
1. Piano transcription quality for polyphonic passages.
2. Pedal event support in model outputs.
3. Robustness to real-world timing misalignment and noisy conditions.
4. C++ deployment path (ONNX Runtime compatibility).
5. Licensing and maintenance risk.

## Algorithm Research

### Option A: Classical DSP (onset + multipitch heuristics)
- Strengths: Simple, lightweight, easy C++ integration.
- Weaknesses: Not competitive for dense polyphonic piano, weak pedal handling, fragile in noisy room recordings.
- Decision: Rejected as primary approach for this scope.

### Option B: Onsets and Frames (dual-objective onset/frame model)
- Source: arXiv 1710.11153 and Magenta overview.
- Key idea: Predict onsets and frame activations jointly, then restrict note starts to onset-supported events.
- Strengths: Strong and well-established piano baseline; good event structure.
- Weaknesses: Does not directly prioritize pedal transcription as a first-class output in the original formulation.
- Decision: Keep as conceptual baseline and fallback architecture reference.

### Option C: High-resolution Piano Transcription with Pedals
- Source: arXiv 2010.01815 and `bytedance/piano_transcription`.
- Key idea: Regress high-resolution onset/offset timing and pedal events, with decoding that refines event times beyond frame hop.
- Strengths:
- Piano-specific model family.
- Explicit pedal modeling, aligned with product requirements.
- Reported strong MAESTRO performance and pedal benchmarks in the paper abstract.
- Weaknesses:
- Upstream repo is archived (read-only as of Dec 8, 2025), which increases long-term maintenance risk.
- Default implementation is PyTorch, so ONNX export/inference plumbing is required for C++ runtime.
- Decision: Selected as primary algorithmic direction for v1 due to piano + pedal fit.

### Option D: Basic Pitch (lightweight instrument-agnostic AMT)
- Source: arXiv 2203.09893 and `spotify/basic-pitch`.
- Key idea: Lightweight multi-output model with broad instrument generalization.
- Strengths:
- Practical deployment profile.
- ONNX model path is already part of upstream distribution.
- Apache-2.0 licensing.
- Weaknesses:
- Instrument-agnostic by design, not piano-specialized.
- Does not provide the same pedal-centric objective as Option C.
- Decision: Selected as fallback implementation path if Option C ONNX export or pedal quality becomes a blocker.

## Dataset and Benchmark Research
- MAESTRO is the primary reference dataset for piano transcription research and includes aligned audio/MIDI with pedal controls.
- MAESTRO alignment is documented around 3 ms and includes sustain/sostenuto/una corda pedal information.
- Relevance: Suitable for sanity checks and threshold calibration before noisy YouTube validation.

## Ingestion Tooling Research
- `yt-dlp` supports:
- `--no-playlist` to avoid accidental playlist expansion.
- `-x/--extract-audio` with ffmpeg-backed extraction.
- `--retries` for transient network failures.
- `ffmpeg` is a general media converter and standard tool for deterministic PCM conversion using explicit input/output configuration.
- Decision:
- Use `yt-dlp` for download and metadata extraction.
- Use `ffmpeg` for deterministic WAV conversion (`44.1 kHz`, `float32`) in a separate step.

## Runtime Research
- ONNX Runtime provides C/C++ APIs for loading and executing ONNX models.
- Decision:
- Standardize inference layer on ONNX Runtime in C++.
- Keep model selection behind an interface to swap model artifacts without changing pipeline wiring.

## Selected Solution
1. Primary model strategy:
- High-resolution piano transcription with pedal-aware decoding (Option C), exported to ONNX and served through ONNX Runtime.

2. Inference/runtime strategy:
- C++ ONNX Runtime wrapper with model-agnostic tensor interface.

3. Ingestion strategy:
- Automated YouTube download via `yt-dlp` with `--no-playlist` and bounded retries.
- Audio extraction/normalization via `ffmpeg` into deterministic WAV for preprocessing and inference.

4. Post-processing strategy:
- Onset/offset confidence hysteresis.
- Minimum note duration filtering.
- Pedal-aware note extension logic.
- Optional quantization mode for DAW workflows.

5. Fallback strategy:
- If Option C export quality or runtime behavior is not acceptable, switch to Basic Pitch ONNX artifacts for a stable deploy baseline, while preserving the same C++ pipeline and MIDI writer.

## Why This Selection
- It matches the core requirement most directly: noisy piano + pedal-aware MIDI.
- It keeps the production runtime in C++.
- It preserves execution pragmatism through a fallback model path with ready ONNX support.

## Known Risks and Mitigations
- Risk: Archived upstream for primary model implementation.
- Mitigation: Pin model artifacts, vendor conversion scripts, and treat upstream as reference-only.

- Risk: Domain gap between curated benchmarks and noisy YouTube audio.
- Mitigation: Add a noisy YouTube validation set and tune preprocessing and post-processing thresholds on real clips.

- Risk: Pedal over-extension in reverberant recordings.
- Mitigation: Add release hysteresis and pedal confidence thresholds; expose tunables in CLI.

## Source Links
- Onsets and Frames paper (arXiv): https://arxiv.org/abs/1710.11153
- Onsets and Frames overview (Magenta): https://magenta.tensorflow.org/onsets-frames
- High-resolution Piano Transcription with Pedals (arXiv): https://arxiv.org/abs/2010.01815
- ByteDance piano_transcription repo: https://github.com/bytedance/piano_transcription
- Basic Pitch paper (arXiv): https://arxiv.org/abs/2203.09893
- Basic Pitch repo: https://github.com/spotify/basic-pitch
- MAESTRO dataset: https://magenta.tensorflow.org/datasets/maestro
- ONNX Runtime C/C++ API docs: https://onnxruntime.ai/docs/api/c/index.html
- yt-dlp repo and options: https://github.com/yt-dlp/yt-dlp
- ffmpeg documentation: https://www.ffmpeg.org/ffmpeg.html
