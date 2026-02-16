# Audio to MIDI Converter Plan (Python)

## Goal
Build an offline Python tool that converts YouTube piano performance videos (including room noise) into MIDI files with usable note, velocity, and sustain pedal events.

## Scope
- Input: YouTube video URL (`youtube.com` or `youtu.be`)
- Audio conditions: Piano performances with room noise, audience noise, and mild reverb
- Output: Standard MIDI file (`.mid`)
- Mode: Offline batch conversion (not real-time)

## Requirements
### Functional Requirements
- Accept a single YouTube video URL via CLI and generate a MIDI file from that source.
- Automatically download source audio with `yt-dlp` and reject playlists by default.
- Retry transient download failures up to 3 attempts before failing.
- Cache downloaded media by video ID and reuse cached content on repeated runs.
- Convert media to deterministic WAV (`44.1 kHz`, `float32`) before inference.
- Transcribe polyphonic piano note events (pitch + onset + offset) from noisy recordings.
- Emit velocity and sustain pedal events in output MIDI.
- Support optional `--keep-intermediate` behavior for debugging artifacts.
- Return non-zero exit codes and actionable error messages on fatal failures.

### System Requirements
- Target runtime environment: macOS or Linux.
- CPU baseline support is required; GPU acceleration is optional.
- Minimum memory target: 8 GB RAM.
- Temporary working storage must be available for downloaded media, WAV intermediates, and model artifacts.

### Software Requirements
- Python 3.10+ runtime.
- `pip` for dependency installation.
- `yt-dlp` Python package (or CLI) available at runtime.
- `ffmpeg` installed and accessible on `PATH`.
- `piano-transcription-inference` package for primary transcription backend.
- `torch` runtime compatible with selected hardware (CPU or CUDA).
- Optional fallback backend: `basic-pitch`.

### Model Requirements
- A local, version-pinned model artifact must be available before inference.
- The selected model must support polyphonic piano transcription.
- Preferred backend includes sustain pedal-aware outputs; fallback backend must still produce reliable note events.

### Quality and Performance Requirements
- Must complete end-to-end conversion for typical clips without process crashes or hangs.
- Must reduce short false-note bursts from room noise using post-processing thresholds.
- Must preserve expressive timing by default (quantization disabled unless explicitly requested).
- Must produce deterministic outputs for identical inputs and configuration.

## Non-Goals (v1)
- General multi-instrument transcription
- Fully accurate transcription for heavily mixed or non-piano content
- Browser or GUI app

## Architecture
1. `Downloader`
- Uses embedded `yt_dlp.YoutubeDL` for download and metadata extraction.
- Enforces single-video behavior (`no_playlist`) by default.
- Caches by video ID to avoid repeated downloads.

2. `Extractor`
- Uses `ffmpeg` to convert downloaded media to WAV (`44.1kHz`, `float32`).
- Stores normalized intermediate filename in working directory.

3. `Preprocess`
- Loudness normalization.
- High-pass filter around `30-40 Hz` for low-end rumble.
- Optional light denoise for stationary room noise.

4. `Transcriber`
- Primary backend: `piano-transcription-inference` (PyTorch).
- Backend abstraction to support fallback engine (`basic-pitch`).

5. `PostProcess`
- Confidence hysteresis (separate note-on and note-off thresholds).
- Minimum duration filter to remove short false positives.
- Pedal-aware note extension handling.
- Optional quantization mode for DAW editing.

6. `MidiWriter`
- Writes `.mid` output using backend output or post-processed events.
- Preserves expressive timing by default (no forced quantization).

## CLI
```bash
audio2midi \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output out.mid \
  --workdir .cache/audio2midi \
  --keep-intermediate
```

## Dependencies
- Runtime:
- Python 3.10+
- `yt-dlp`
- `ffmpeg` (system binary)
- `numpy`
- `scipy`
- `librosa`
- `torch`
- `piano-transcription-inference`
- Optional: `basic-pitch`

- Dev/Test:
- `pytest`
- `pytest-cov`
- `ruff`

## Package/Layout Plan
- `audio2midi/cli.py`: CLI parsing and top-level orchestration
- `audio2midi/downloader.py`: URL validation, cache logic, yt-dlp integration
- `audio2midi/extractor.py`: ffmpeg conversion wrapper
- `audio2midi/preprocess.py`: normalization, filtering, optional denoise
- `audio2midi/transcribers/base.py`: backend interface
- `audio2midi/transcribers/piano_transcription_inference.py`: primary backend
- `audio2midi/transcribers/basic_pitch.py`: fallback backend
- `audio2midi/postprocess.py`: event cleanup and pedal-aware adjustments
- `audio2midi/midi_writer.py`: final MIDI emission
- `tests/`: unit and integration tests

## Error Handling
- Retry download failures up to 3 attempts for transient errors.
- Surface clear errors for missing `ffmpeg` or unsupported URL inputs.
- Fail fast on invalid model/backend configuration.
- Return non-zero exit codes for all fatal pipeline failures.

## Milestones
1. Scaffold
- Python package skeleton, CLI parsing, module interfaces.

2. Ingestion
- Implement downloader + extractor with cache and retries.
- Verify stable WAV output from YouTube URLs.

3. Baseline Transcription
- Integrate `piano-transcription-inference` on local WAV input.
- Produce first MIDI output without advanced filtering.

4. End-to-End YouTube Pipeline
- Connect ingestion output to transcription pipeline.
- Generate MIDI directly from URL input.

5. Noise Robustness
- Add denoise + threshold tuning against noisy YouTube piano clips.
- Reduce false onsets and fragmented notes.

6. Pedal and Velocity Quality
- Improve sustain pedal handling.
- Improve velocity consistency.

7. Validation and Hardening
- Add integration tests and regression fixtures.
- Benchmark runtime and memory for typical clip lengths.

## Test Strategy
- Unit tests:
- URL parsing and validation.
- Cache hit/miss behavior.
- yt-dlp option construction.
- ffmpeg command construction.
- Post-processing threshold logic.

- Integration tests:
- Local WAV to MIDI smoke test.
- YouTube URL to MIDI smoke test (optional/flagged in CI due to network dependency).
- Golden-output comparisons for selected clips.

## Acceptance Criteria (v1)
- Converts a valid YouTube piano URL to a MIDI file from a single CLI command.
- Handles moderate room noise without overwhelming false notes.
- Produces sustain pedal events in output MIDI when present in source performance.
- Re-running same URL uses cache and avoids re-download.
- Failures report actionable messages and exit with non-zero status.

## Open Decisions
- Final default backend (`piano-transcription-inference` vs fallback path behavior).
- Dependency lock strategy (`requirements.txt` only vs constraints/lock tooling).
- CPU-only baseline vs optional CUDA acceleration in first release.
