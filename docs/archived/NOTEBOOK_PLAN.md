# Notebook Implementation Plan: PTI Theory + Algorithm Visualization

## Summary
This plan defines an algorithm-focused Jupyter notebook that explains and visualizes how piano transcription works with the PTI (High-resolution Piano Transcription with Pedals) approach and how those outputs are handled by this application.

Deliverables:
1. `NOTEBOOK_PLAN.md` (this file)
2. `notebook/midi_conversion_algorithm.ipynb`
3. `notebook/README.md`
4. `requirements-notebook.txt`

## PTI Research Findings
### 1) Front-end and time-frequency representation
- PTI operates at `16 kHz` sample rate.
- It computes STFT with `n_fft=2048` and hop from `frames_per_second=100` (hop size `160`).
- It uses a log-mel representation with `229` mel bins.
- Keys modeled: `88` piano notes.

### 2) Note and pedal head structure
- Note outputs:
- regression onset
- regression offset
- frame activity
- velocity
- Pedal outputs:
- pedal onset regression
- pedal offset regression
- pedal frame output
- PTI combines note and pedal models in `Note_pedal`.

### 3) Regression-to-event decoding logic
- PTI converts regression outputs to binarized events by threshold + local monotonic neighborhood logic.
- Note decoding is onset-first, then offset/frame-based completion.
- Onset/offset shift terms provide sub-frame timing refinement.
- Pedal decoding is handled separately and returned as pedal events.

### 4) Segmentation and deframing
- Long audio is segmented into overlapping chunks.
- Per-segment outputs are stitched back to a continuous timeline through deframing logic.

## Notebook Section Design (Cell-by-Cell)
1. Intro and learning goals.
2. Notation and task formulation for AMT.
3. PTI architecture walkthrough.
4. Feature pipeline theory and parameter grounding.
5. Decoder theory and pseudo-code.
6. Synthetic decoder demo using PTI VAD functions.
7. App mapping:
- PTI transcription -> MIDI
- `read_midi_as_transcription`
- `postprocess_transcription`
8. Postprocess parameter sweep visualization.
9. Piano roll, velocity histogram, pedal timeline visualizations.
10. Accuracy caveats (PTI internals vs app-level behavior).
11. Conclusion with algorithm levers.

## Public Notebook Config Contract
The first config code cell defines:
- `RUN_SYNTHETIC_DECODER_DEMO = True`
- `RUN_OPTIONAL_HEAVY_PTI = False`
- `USE_CACHED_ARTIFACTS = True`
- `AUDIO_WAV_PATH` (local wav path used for optional heavy PTI and app-level demos)
- `PTI_CHECKPOINT_PATH` (optional)
- `MIN_DURATION_SECONDS`
- `NOTE_THRESHOLD`
- `QUANTIZE_GRID_SECONDS`

## Test Scenarios
1. Fast mode runs without heavy PTI (`RUN_OPTIONAL_HEAVY_PTI=False`).
2. Synthetic decoder demo produces deterministic event tuples.
3. Parameter sweep changes event counts/durations predictably.
4. Optional heavy PTI mode runs when dependencies and checkpoint are available.
5. Notebook claims match PTI code behavior and this repo implementation.

## Assumptions and Defaults
- Single notebook under `notebook/`.
- Audience is mixed developers (conceptual + implementation aware).
- Scope is algorithm-focused only.
- Lightweight mode is the default.
- Heavy inference is optional and explicitly guarded.

## Source References
- PTI paper: https://arxiv.org/abs/2010.01815
- PTI codebase: https://github.com/bytedance/piano_transcription
- Local PTI package sources used for algorithm details:
- `piano_transcription_inference/models.py`
- `piano_transcription_inference/utilities.py`
- `piano_transcription_inference/piano_vad.py`
- `piano_transcription_inference/inference.py`
- Application mapping sources:
- `audio2midi/transcribers/piano_transcription_inference.py`
- `audio2midi/postprocess.py`
- `audio2midi/midi_writer.py`
- `audio2midi/models.py`
