# Audio to MIDI Converter Plan

## Goal
Build an offline C++ tool that converts YouTube piano performance videos (including room noise) into MIDI files with usable note, velocity, and sustain pedal events.

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
- C++17-compatible compiler.
- CMake build system.
- ONNX Runtime C++ API available at build/runtime.
- `midifile` library available for MIDI writing.
- `yt-dlp` installed and accessible on `PATH`.
- `ffmpeg` installed and accessible on `PATH`.
- Internet connectivity is required only for the YouTube download step.

### Model Requirements
- A local ONNX transcription model artifact must be present before inference.
- The selected model must support polyphonic piano transcription.
- Preferred model path includes sustain pedal-aware outputs; fallback model path must still produce note events reliably.

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
1. `YouTubeDownloader`
- Uses `yt-dlp` to download best available audio stream for a single video
- Caches by video ID to avoid repeated downloads
- Rejects playlists unless `--allow-playlist` is set

2. `AudioExtractor`
- Uses `ffmpeg` to convert downloaded media to WAV (`44.1kHz`, `float32`)
- Stores normalized intermediate filename in working directory

3. `Preprocess`
- Loudness normalization
- High-pass filter around `30-40 Hz` for low-end rumble
- Optional light denoise for stationary room noise

4. `Transcriber`
- ONNX model inference via ONNX Runtime
- Piano-focused polyphonic transcription model (onset/frame style outputs preferred)

5. `PostProcess`
- Confidence hysteresis (separate note-on and note-off thresholds)
- Minimum duration filter to remove short false positives
- Pedal-aware note extension handling
- Optional quantization mode for DAW editing

6. `MidiWriter`
- Writes `.mid` output via `midifile`
- Preserves expressive timing by default (no forced quantization)

## CLI
```bash
audio2midi \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output out.mid \
  --workdir .cache/audio2midi \
  --keep-intermediate
```

## Dependencies
- Build/runtime:
- C++17+
- CMake
- ONNX Runtime (C++ API)
- `midifile` (MIDI writing)

- External tools:
- `yt-dlp`
- `ffmpeg`

## File/Module Plan
- `src/main.cpp`: CLI entrypoint and orchestration
- `src/downloader/*`: YouTube URL handling, cache logic, `yt-dlp` invocation
- `src/audio/*`: extraction, waveform loading, preprocessing
- `src/inference/*`: ONNX Runtime wrapper and tensor preparation
- `src/postprocess/*`: note event decoding, thresholding, pedal logic
- `src/midi/*`: MIDI event writing
- `include/audio2midi/*`: public headers
- `tests/*`: unit and integration tests

## Error Handling
- Retry `yt-dlp` download up to 3 attempts for transient failures
- Surface clear errors for missing binaries (`yt-dlp`, `ffmpeg`)
- Validate URL format and fail fast on unsupported sources
- Return non-zero exit codes for all fatal pipeline failures

## Milestones
1. Scaffold
- CMake project skeleton, CLI parsing, module interfaces

2. Ingestion
- Implement downloader + extractor with cache and retries
- Verify stable WAV output from YouTube URLs

3. Baseline Transcription
- Integrate ONNX Runtime with local WAV input path
- Produce first MIDI output without advanced filtering

4. End-to-End YouTube Pipeline
- Connect ingestion output to transcription pipeline
- Generate MIDI directly from URL input

5. Noise Robustness
- Add denoise + threshold tuning against noisy YouTube piano clips
- Reduce false onsets and fragmented notes

6. Pedal and Velocity Quality
- Improve sustain pedal handling
- Improve velocity mapping consistency

7. Validation and Hardening
- Add integration tests and regression fixtures
- Benchmark runtime and memory for typical clip lengths

## Test Strategy
- Unit tests:
- URL parsing and validation
- Cache hit/miss behavior
- Command invocation argument construction
- Post-processing threshold logic

- Integration tests:
- Local WAV to MIDI smoke test
- YouTube URL to MIDI smoke test (optional/flagged in CI due to network dependency)
- Golden-output comparisons for selected clips

## Acceptance Criteria (v1)
- Converts a valid YouTube piano URL to a MIDI file from a single CLI command
- Handles moderate room noise without overwhelming false notes
- Produces sustain pedal events in output MIDI when present in source performance
- Re-running same URL uses cache and avoids re-download
- Failures report actionable messages and exit with non-zero status

## Open Decisions
- Final transcription model choice (piano-specific ONNX candidate)
- Default quantization behavior (`off` by default is currently planned)
- CPU-only baseline vs optional GPU execution providers
