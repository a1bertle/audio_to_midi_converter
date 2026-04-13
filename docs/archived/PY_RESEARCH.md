# Audio to MIDI Research and Solution Selection (Python)

## Objective
Select a Python-first algorithm and implementation strategy for converting noisy YouTube piano performances into MIDI.

## Problem Characteristics
- Input is solo piano from YouTube videos, not studio-clean stems.
- Audio may include room noise, audience noise, and reverb.
- Output must be polyphonic note events and sustain pedal behavior, not only pitch contours.
- Pipeline must support automated YouTube download and offline batch processing.

## Evaluation Criteria
1. Piano transcription quality for polyphonic passages.
2. Pedal event support in model outputs.
3. Robustness to noisy real-world recordings.
4. Python implementation simplicity and maintainability.
5. Licensing and dependency risk.

## Algorithm and Stack Research

### Option A: `piano-transcription-inference` (PyPI package)
- What it is:
- Python package for piano transcription inference built from the high-resolution piano transcription work.
- Research and package pages document direct transcription to MIDI and CPU/GPU usage.

- Evidence:
- PyPI lists `piano-transcription-inference` 0.0.6 released on January 26, 2025.
- PyPI metadata lists MIT license and Python `>=3.6`.
- Package usage examples include:
- Loading audio and creating `PianoTranscription(device='cuda' | 'cpu')`.
- Writing MIDI via `transcribe(audio, output_midi_path)`.

- Strengths:
- Piano-specific and pedal-aware lineage.
- Fast path to production in Python with minimal custom model plumbing.
- MIT licensing.

- Risks:
- Upstream training repo `bytedance/piano_transcription` is archived (read-only as of December 8, 2025).
- Original repo environment was Python 3.7 and PyTorch 1.4.0, so version-compatibility testing is required for modern stacks.

### Option B: `basic-pitch` (Spotify)
- What it is:
- Python AMT package with CLI and programmatic API.

- Evidence:
- README documents `pip install basic-pitch`.
- README states instrument-agnostic, polyphonic transcription, and notes that it works best on one instrument at a time.
- Repo uses Apache-2.0 license.
- README documents multiple inference runtimes and CLI usage.

- Strengths:
- Maintained and easy to install.
- Strong deployment ergonomics and clean Python API.

- Risks:
- Not piano- or pedal-specialized, so sustain/pedal behavior quality may lag piano-specific systems.

### Option C: Classical DSP-only (onset + multipitch heuristics)
- Strengths:
- Lightweight and easy to debug in Python.

- Weaknesses:
- Not competitive for dense polyphonic piano with room noise.
- Weak sustain pedal reconstruction.

- Decision:
- Rejected for primary implementation.

## Ingestion Research (YouTube)
- `yt-dlp` supports:
- `--no-playlist` to avoid accidentally processing full playlists.
- `-R, --retries` for transient failures.
- Python embedding via `from yt_dlp import YoutubeDL`.
- `yt-dlp` docs explicitly recommend avoiding parsing default stdout; use structured outputs/options instead.

## Audio Processing Research
- `ffmpeg` documentation confirms:
- `-ar` for output sample rate.
- `-ac` for channel count.
- `-sample_fmt` for output sample format.
- `librosa.load` documentation confirms:
- default float32 loading and optional resampling.
- `sr=None` preserves native sample rate.

## MIDI Handling Research
- `pretty_midi` PyPI (0.2.11 released October 8, 2025) remains an active option for post-write normalization/cleanup workflows.
- `mido` PyPI (1.3.3) provides low-level MIDI message control and robust file IO.

## Selected Solution
1. Primary transcription strategy:
- Use `piano-transcription-inference` as the default transcription engine in Python.

2. Ingestion strategy:
- Use embedded `yt_dlp.YoutubeDL` with explicit options (`no_playlist`, bounded retries).
- Use `ffmpeg` subprocess conversion to deterministic WAV prior to transcription.

3. Pipeline strategy:
- Add light preprocessing for noisy YouTube audio (normalization, high-pass, optional denoise).
- Run transcription on CPU by default with optional CUDA path.
- Keep a post-processing stage for thresholding and note cleanup before final MIDI write.

4. Fallback strategy:
- If package compatibility or pedal quality is unacceptable on target machines, switch engine to `basic-pitch` behind a shared transcription interface.

## Why This Selection
- Best fit for current product needs (piano + pedal + Python-first delivery speed).
- Lowest implementation complexity for a usable v1.
- Practical fallback exists without changing CLI contract.

## Risks and Mitigations
- Risk: Archived upstream training repo increases long-term maintenance risk.
- Mitigation: Pin working package/model versions, add transcription backend abstraction, and keep fallback backend tested.

- Risk: Domain gap between benchmark-quality data and noisy YouTube recordings.
- Mitigation: Build a noisy validation set and tune preprocessing/post-processing thresholds.

- Risk: Environment mismatch across Python, PyTorch, and system libraries.
- Mitigation: Provide tested environment matrix and lock dependencies in `requirements.txt`.

## Source Links
- `piano-transcription-inference` PyPI: https://pypi.org/project/piano-transcription-inference/
- `bytedance/piano_transcription` repository: https://github.com/bytedance/piano_transcription
- `basic-pitch` repository: https://github.com/spotify/basic-pitch
- `yt-dlp` repository and options: https://github.com/yt-dlp/yt-dlp
- `ffmpeg` CLI documentation: https://ffmpeg.org/ffmpeg-doc.html
- `librosa.load` docs: https://librosa.org/doc/main/generated/librosa.load.html
- `pretty_midi` PyPI: https://pypi.org/project/pretty_midi/
- `mido` PyPI: https://pypi.org/project/mido/
