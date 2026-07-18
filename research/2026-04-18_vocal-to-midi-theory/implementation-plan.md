# Implementation Plan: Vocal-to-MIDI Pipeline — Theory-Grounded Design

**Date:** 2026-04-18
**Session:** `research/2026-04-18_vocal-to-midi-theory/`
**Based on:** `notes.md` findings F1–F8 and prior session findings
**Scope:** This plan translates the theoretical survey into concrete implementation priorities and
component design. No code is written in this session.

---

## Problem Statement (from theory)

Prior sessions confirmed the pipeline fails at Stage 2 (F0 estimation) due to Stage 1 (source
separation) routing vocal fundamentals to the wrong stem. The theoretical research identifies:

1. Monophonic pitch estimators require clean vocal (PESTO, FCPE, pYIN, CREPE)
2. Polyphonic-robust pitch estimators can operate on the mix (RMVPE, SPICE)
3. Note segmentation remains the hardest stage regardless of pitch estimator quality

---

## Recommended Pipeline Architecture

```
Audio Input
    │
    ├─[Path A: Clean Track]──────────────────────────────────────────────────┐
    │  Use when: user has isolated vocal stem                                 │
    │  Stage 2: FCPE (fastest, 96.79% RPA)                                   │
    │                                                                         │
    ├─[Path B: Mix → Separate → Pitch]───────────────────────────────────────┤
    │  Use when: clean separation is achievable (e.g. future better model)    │
    │  Stage 1: Better separation model (NOT current htdemucs)                │
    │  Stage 2: FCPE or PESTO                                                 │
    │                                                                         │
    └─[Path C: Mix → Polyphonic Pitch Estimator]─────────────────────────────┤
       Use when: mix input only; no separation available                      │
       Stage 2: RMVPE (designed for polyphonic; best accuracy on mix)         │
       Current state: not yet benchmarked in this project (see 2026-04-17     │
                      session; next priority is RMVPE trial)                  │
                                                                              ▼
                                                               [Stage 3: Voicing Detection]
                                                                Use: output confidence from
                                                                FCPE/RMVPE (threshold TBD)
                                                                              │
                                                                              ▼
                                                               [Stage 4: Note Segmentation]
                                                                Tier 1 (fast): dF0/dt +
                                                                  contiguous-pitch regions
                                                                Tier 2 (quality): HMM /
                                                                  Viterbi over pitch lattice
                                                                              │
                                                                              ▼
                                                               [Stage 5: MIDI Assembly]
                                                                Hard quantize to semitones
                                                                Velocity from RMS per note
                                                                Optional: pitch-bend track
```

---

## Implementation Priority Order

### Priority 1 — RMVPE trial on raw mix (blocking)

Per 2026-04-17 session Conclusion: RMVPE is the strongest candidate for polyphonic-robust pitch
estimation. It has not been run in this project yet.

**Steps:**
1. Clone RMVPE repo: `git clone https://github.com/Dream-High/RMVPE`
2. Install in venv: `cd RMVPE && pip install -e .`
3. Verify M1 compatibility: check for MPS/CPU fallback in model loading code
4. Run on raw mix of the test track (same track used in 2026-04-17 session)
5. Compare pitch track output against FCPE-on-mix (F-T3: FCPE yielded vocal line but with noise)
6. Document RPA proxy (subjective + note count + voiced fraction) in results.md

**Files to create:**
- `research/2026-04-18_vocal-to-midi-theory/run_rmvpe.py` — inference script mirroring
  `research/2026-04-17_vocal-pitch-tracker-research/run_fcpe.py` pattern
- `research/2026-04-18_vocal-to-midi-theory/rmvpe_results.json` — output metrics

**Validation:** Run `validate.sh` — checks that RMVPE output MIDI file exists and has
note count > 0.

---

### Priority 2 — Shared PitchToMIDI utility (one-time cost)

Per notes.md F5 (prior session) and Stage 4/5 analysis: all pitch-track tools share the same
post-processing pipeline. A shared `PitchToMIDI` class should be implemented once and reused.

**Location:** `audio2midi/pitch_to_midi.py` (new file in main package)

**Interface:**
```python
class PitchToMIDI:
    def __init__(
        self,
        frame_rate: float,          # frames/sec (e.g. 50.0 for 20ms hop)
        confidence_threshold: float, # voicing threshold (0–1)
        min_note_duration_ms: int,   # reject notes shorter than this
        quantize_pitch: bool,        # round to nearest semitone
        include_pitch_bend: bool,    # encode sub-semitone deviation as bend
    ): ...

    def convert(
        self,
        f0_hz: np.ndarray,           # shape (T,), Hz or 0 for unvoiced
        confidence: np.ndarray,      # shape (T,), voicing probability
        rms: np.ndarray,             # shape (T,), amplitude per frame
    ) -> mido.MidiFile: ...
```

**Note segmentation algorithm (Tier 1):**
```
1. Threshold confidence at threshold → voiced_mask
2. Apply minimum duration filter → remove short voiced segments
3. Within each voiced segment, detect pitch changes > 0.5 semitone as note boundaries
4. Assign note pitch = median F0 over note window (robust to vibrato)
5. Assign velocity = RMS-to-velocity mapping (log-compressed)
6. If include_pitch_bend: generate pitch-bend events at frame rate
```

**Validation:** Unit test in `tests/test_pitch_to_midi.py` — synthetic sine wave at 440 Hz
→ verify MIDI output contains A4 note.

---

### Priority 3 — Note segmentation quality (HMM upgrade path)

Once Tier 1 segmentation is implemented and produces subjectively reasonable results, upgrade to
HMM-based segmentation:

**Algorithm:**
- 3-state HMM per note: attack / sustain / release
- Observation: (F0 in cents, energy_db, confidence) per frame
- Transition matrix: penalize transitions that skip states or have unrealistic durations
- Emission: Gaussian around note target pitch (sustain state), wider Gaussian for attack/release
- Viterbi decoding: `scipy.optimize` or manual implementation (~50 lines)

**Note:** This is a research-only step until RMVPE trial confirms the pipeline is worth
investing in. Do not implement until Priority 1 and 2 are complete.

---

### Priority 4 — End-to-end models (future research)

ROSVOT (2024) and ACL 2025 unified frameworks represent the quality ceiling. Revisit when:
- RMVPE trial shows residual note boundary failures that Tier 1/2 segmentation cannot solve
- Labeled note-boundary training data is available for this project's genre

---

## Stage-by-Stage Design Decisions (Theory-Grounded)

### Stage 2 algorithm selection matrix

| Scenario | Recommended algorithm | Reason |
|---|---|---|
| Clean isolated vocal | FCPE | Fastest (RTF 0.013×); 96.79% RPA |
| Polyphonic mix | RMVPE | Designed for polyphonic; U-Net rejects accompaniment |
| No GPU, minimal deps | pYIN | Zero new installs; Viterbi smoothing built in |
| Self-supervised / no labeled data | PESTO | <30k params; equivariance objective |

### Stage 3 voicing threshold

Use the confidence output from FCPE/RMVPE directly. Do not apply a static energy threshold first
(causes vibrato frame dropout). Start with threshold = 0.5; tune on real track output.

### Stage 4 segmentation strategy

Start with derivative + contiguous-regions (Tier 1). The theoretical analysis shows this fails
on portamento; accept this limitation in first implementation. Flag for HMM upgrade if output
quality is insufficient.

### Stage 5 pitch encoding

Default: hard quantize to nearest semitone. Add `include_pitch_bend=True` flag for future
vocal synthesis use cases. Do not implement bend encoding until Stage 4 is stable.

---

## Validation Plan

For each Priority above, validation is:

1. **RMVPE trial:** MIDI file produced; note count > 0; subjective comparison vs. FCPE-on-mix
2. **PitchToMIDI utility:** unit test passes; 440 Hz sine → A4 MIDI note
3. **HMM segmentation:** compare note count and boundary accuracy against Tier 1 on same track

All results to be recorded in session `results.md` with provenance tags.
