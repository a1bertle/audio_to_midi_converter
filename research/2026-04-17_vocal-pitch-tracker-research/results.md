# Trial Results: Vocal Pitch Tracker Evaluation

**Date:** 2026-04-17
**Session:** `research/2026-04-17_vocal-pitch-tracker-research/`
**Input:** `output/2026-04-13_escene-htdemucs-4s-wav/E.scene ＂意識＂ Music Video/vocals.wav`
**Track duration:** 245.59 s
**Baseline (prior session):** basic-pitch on same stem → 650 notes, 2.65 notes/s, mean conf 0.592, subjectively poor

---

## SPICE Status

SPICE (TensorFlow Hub) was skipped: `tensorflow-hub 0.15` imports `pkg_resources` which was removed
from setuptools 82+, and `tensorflow-hub 0.16` imports `tf.compat.v1.estimator` which was removed
in TensorFlow 2.21. The TF 2.21 already in the venv makes both hub versions incompatible without a
full TF downgrade. SPICE deferred to a future session with a clean venv.

---

## Quantitative Results

| Tool | RTF | Notes | Notes/s | Voiced frac | Mean conf | Provenance |
|------|-----|-------|---------|-------------|-----------|------------|
| **basic-pitch** (prior baseline) | 0.3–0.5× [assumed] | 650 | 2.65 | — | 0.592 | [measured — prior session] |
| **pYIN** | 0.249× | 556 | 2.26 | 57.4% | — | [measured] |
| **PESTO** | 0.099× | 265 | 1.08 | 45.6% | 0.985 | [measured] |
| **FCPE** | **0.013×** | 484 | 1.97 | 55.9% | n/a | [measured] |

All RTF values [measured] on Apple M1 (16 GB). FCPE ran on CPU (MPS not supported for STFT op).
PESTO ran on CPU via 60s chunking to avoid OOM.

---

## Validation Against Target Thresholds

| Metric | Target | pYIN | PESTO | FCPE |
|--------|--------|------|-------|------|
| Notes | 200–800 | ✓ 556 | ✓ 265 | ✓ 484 |
| Notes/s | 1–5 | ✓ 2.26 | ✓ 1.08 | ✓ 1.97 |
| Voiced fraction | 40–70% | ✓ 57.4% | ✓ 45.6% | ✓ 55.9% |
| RTF | ≤ 3.0× | ✓ | ✓ | ✓ |

All three tools pass all quantitative thresholds. [measured]

---

## Notable Observations

### FCPE — speed standout
[measured]
RTF 0.013× — processes 245s of audio in 3.1s on M1 CPU. This is ~10× faster than pYIN and ~8×
faster than PESTO. MPS backend is not viable (STFT op falls back to CPU, making it 26× slower
than pure CPU). CPU-only is the correct path for FCPE on M1.

### PESTO — highest confidence, lowest note density
[measured]
Mean confidence 0.985 on voiced frames (scale 0–1). Lowest note count (265) and rate (1.08/s)
suggests it is more conservative: it only segments a new note when the pitch contour moves by a
full semitone boundary. The confidence value on this stem is very high despite the stem quality
issues — PESTO's training on MIR-1K (vocal + accompaniment) may make it tolerant of the residual
bleed content.

### pYIN — most similar to basic-pitch baseline
[measured]
556 notes at 2.26 notes/s — closest to the basic-pitch baseline (650 notes, 2.65 notes/s). As a
probabilistic YIN estimator it is expected to track similarly to a CQT-based detector. The lower
count vs. basic-pitch may be due to the more conservative `min_note_duration_s=0.08` applied in
post-processing.

### MIDI quality — subjective assessment needed
[assumed — no ground truth]
Quantitative metrics (note count, rate, voiced fraction) are all plausible. **Subjective listening
assessment of the three MIDI files is the next required step** before selecting a tool. The prior
session (basic-pitch) showed that in-range statistics can be a false indicator — subjective quality
is the binding criterion.

---

## Post-Processing Bug Fixed During Session

The initial `pitch_to_midi.py` was converting frame pitch to MIDI note number per-frame without
first snapping to semitones, causing any sub-semitone pitch drift within a sustained note to be
treated as a new note event (1731 notes reported initially for PESTO). Fixed by snapping f0_hz to
the nearest integer MIDI note before segmenting. [measured — before/after comparison: 1731 → 265
notes for PESTO].

PESTO returns timestamps in **milliseconds**, not seconds. The segment time offset was incorrectly
added in seconds units. Fixed by dividing PESTO timestamps by 1000 before offset addition.
[measured — confirmed via raw timestamp inspection: `times[-1] = 5000` for a 5s clip].

---

---

## Phase 2 Results: RMVPE + BPM-aligned quantization + onset splitting (2026-04-17)

### Pipeline

1. **BPM detection** — `bpm_detect` on raw mix → **93.75 BPM, C# minor, 4/4**
2. **Pitch tracking** — RMVPE (`rmvpe_applio.py`, checkpoint `rmvpe.pt` 173 MB) on htdemucs 4s
   vocals stem, CPU, `thred=0.03`
3. **Note segmentation** — `pitch_to_midi.py` with:
   - Semitone snapping before segmentation
   - `min_note_duration_s = 80ms` (1/32nd note at 93.75 BPM)
   - Silence-aware note merging (only merges adjacent notes, never across gaps ≥ 40ms)
   - Onset-based syllable splitting (`librosa.onset.onset_detect`, `delta=0.07`, `hop=256`)
4. **Onset detection** — 643 onsets detected (2.62/s) → 27 additional note splits

### Key results

| Version | Notes | Notes/s | Provenance |
|---------|-------|---------|------------|
| RMVPE raw (no post-processing) | 457 | 1.86 | [measured] |
| RMVPE + merge (32nd, old merge) | 332 | 1.35 | [measured] |
| RMVPE + silence-aware merge (32nd) | 411 | 1.67 | [measured] |
| RMVPE + silence-aware merge + onsets | **438** | **1.78** | [measured] |

### Findings

**F-P1 — BPM-aligned thresholds are the correct framing** [measured]
At 93.75 BPM, a 32nd note = 80ms. Using this as `min_note_duration_s` produces musically coherent
note density. Arbitrary ms thresholds were less principled.

**F-P2 — Silence-aware merge is essential** [measured]
The original merge collapsed same-pitch notes across phrase boundaries (repeated notes on the same
key merged into one long note). Adding a `max_silence_s=40ms` boundary check fixed this.

**F-P3 — Onset splitting recovers multi-syllabic passages** [measured]
27 additional splits from 643 detected onsets. Subjectively confirmed: passages where the singer
repeats the same pitch across multiple syllables are now more articulated.

**F-P4 — RMVPE on htdemucs 4s stem is the best pipeline so far** [measured]
Subjectively better than all previous approaches (pYIN, PESTO, FCPE on stem/mix/blend).
Root cause: RMVPE's polyphonic robustness means it handles the bleed-contaminated stem better
than single-pitch estimators.

### Best output
`rmvpe_32nd_with_onsets.mid` — 438 notes, 1.78 notes/s, BPM=93.75

### Remaining issues
- Some ornaments and slides still produce short spurious notes
- Onset detector misses some syllable boundaries (sustained vowels with no amplitude transient)
- No velocity variation (fixed at 0.8 confidence → ~64 MIDI velocity)

## Recommended Next Steps

1. **Continue tuning** — the current pipeline (RMVPE + 32nd note threshold + silence-aware merge
   + onset splitting) is the best result so far. Further improvements:
   - Tune onset `delta` threshold per-track or adaptively
   - Add velocity variation from RMS envelope of the vocals stem
   - Quantize note start/end times to the nearest 32nd note grid
2. **Integrate into the main pipeline** — the full chain is:
   `bpm_detect` → `htdemucs 4s` → `RMVPE` → `onset_detect` → `pitch_to_midi` → `.mid`
3. **SPICE** (direct-on-mix, no stem separation) remains untested due to TF/hub version conflict.
   Revisit in a fresh venv if stem-based quality plateaus.
