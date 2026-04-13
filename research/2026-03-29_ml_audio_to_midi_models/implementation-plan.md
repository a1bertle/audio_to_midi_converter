# Implementation plan: adding/expanding ML audio → MIDI backends

Date: 2026-03-29
Scope: Planning only (no code changes in this document).

## Goals
- Keep the existing pipeline stable on macOS Apple Silicon.
- Improve backend selection and “good enough” output quality for:
  - Piano (pedal-aware) → prioritize PTI.
  - Guitar/other polyphonic → prioritize Basic Pitch.
- Optionally allow a “research-grade” multi-instrument backend without making it a required install.

## Proposed work items (minimal + safe)
1) Document the backend contract and output expectations
   - File: `audio2midi/transcribers/base.py`
   - Define: what a `TranscriptionResult` should contain (notes, optional pedals, optional bends).

2) Add a third piano backend option: `transkun` (optional dependency)
   - Files:
     - `audio2midi/transcribers/transkun.py` (new)
     - `audio2midi/transcribers/base.py` (factory update)
     - `README.md` (backend docs + install notes)
   - Dependency strategy:
     - Keep it optional (separate requirements file or optional extra) so the default install stays
       stable.
   - Evaluation focus:
     - Verify pedal behavior and the MIDI convention (pedal extension vs. explicit CC events)
       matches your target.

3) Basic Pitch: preserve pitch bends (optional feature)
   - File: `audio2midi/transcribers/basic_pitch.py`
   - Add: extraction of pitch-bend events if `basic_pitch` returns them (store in model data).
   - File: `audio2midi/midi_writer.py`
   - Add: MIDI pitch bend writing (and round-trip parsing if needed).

4) PTI: Apple Silicon quality-of-life/performance guardrails
   - File: `audio2midi/transcribers/piano_transcription_inference.py`
   - Add: clearer device help text (CPU on macOS; “cuda” only on NVIDIA).
   - Consider: chunked inference for very long files (if upstream supports it) to reduce peak memory.
   - Keep: checkpoint download behavior and size validation.

5) MT3 backend as an optional “extra”
   - Files:
     - `audio2midi/transcribers/mt3.py` (new)
     - `audio2midi/transcribers/base.py` (factory update)
     - `README.md` (backend docs)
   - Dependency strategy:
     - Add an optional dependency group (e.g. `extras_require`) or a separate requirements file
       to keep default install light.
   - Operational constraints:
     - Document expected RAM usage and runtime; mark as experimental.

## Validation steps (repeatable)
- Run import checks for each backend in the venv.
- Run `pytest -q` for the existing unit tests.
- Smoke test (manual) using an existing local audio file already in the repo:
  - `Foals - My Number - Harp.mp3` for Basic Pitch (instrument-agnostic).
  - A short piano clip (if present/approved) for PTI.

## Non-goals
- Training or fine-tuning models.
- Introducing new external datasets or synthetic benchmark audio without an explicit request.
