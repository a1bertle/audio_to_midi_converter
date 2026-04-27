# Implementation Plan: Vocal MIDI Coverage and Fragmentation Fix

**Date:** 2026-04-26
**Session:** `plans/2026-04-26_vocal-midi-coverage-fragmentation/`
**Proposal:** `proposal.md` (same directory)

---

## Scope

Two targeted changes to `audio2midi/transcribers/rmvpe.py`, applied inside
`_pitch_track_to_notes`. No other files are touched unless validation reveals
a secondary defect.

**In scope:**
1. Raise `_merge_same_pitch` gap ceiling from 8th-note to quarter-note
2. Add F0-guided gap-fill pass after note segmentation

**Out of scope:**
- RMVPE `thred` / voicing threshold (risk: increases hallucination)
- Stem separation changes
- Velocity / dynamics
- MIDI duration truncation (diagnose separately after main fix)

---

## Affected Files

| File | Change |
|------|--------|
| `audio2midi/transcribers/rmvpe.py` | Two changes to `_pitch_track_to_notes` (see steps 1–2) |
| `scripts/eval_vocal_midi.py` | No change — used as-is for validation |

---

## Implementation Steps

### Step 0 — Understand the current call sites

In `_pitch_track_to_notes` (`rmvpe.py` lines 258–409):

- **Same-pitch merge pass 1** (line ~346):
  ```python
  eighth_s = (60.0 / bpm) / 2.0
  raw = _merge_same_pitch(raw, eighth_s, beat_arr_pre, pre_beat_tol)
  ```
- **Same-pitch merge pass 2** (line ~391):
  ```python
  raw = _merge_same_pitch(raw, eighth_s, beat_arr_pre, pre_beat_tol)
  ```
- The `f0_hz` array (voiced frames > 0 only, at `_HOP_S = 0.01 s/frame`) is
  available at the top of the function and is already median-filtered.

---

### Step 1 — Raise same-pitch merge ceiling to quarter-note

**Location:** `_pitch_track_to_notes`, lines ~339 and ~391.

**Change:** Replace `eighth_s` with `quarter_s = 60.0 / bpm` at both call
sites. Rename the local variable to avoid confusion.

**Before (both calls):**
```python
eighth_s = (60.0 / bpm) / 2.0
...
raw = _merge_same_pitch(raw, eighth_s, beat_arr_pre, pre_beat_tol)
...
raw = _merge_same_pitch(raw, eighth_s, beat_arr_pre, pre_beat_tol)
```

**After:**
```python
quarter_s = 60.0 / bpm  # one beat; was 8th-note before
...
raw = _merge_same_pitch(raw, quarter_s, beat_arr_pre, pre_beat_tol)
...
raw = _merge_same_pitch(raw, quarter_s, beat_arr_pre, pre_beat_tol)
```

The existing beat-gate inside `_merge_same_pitch` prevents merging across
genuine beat-boundary repeated notes, so widening the ceiling is safe.

---

### Step 2 — Add F0-guided gap-fill pass

**Location:** Insert after the second `_merge_same_pitch` call (pass 2) and
before the optional `snap_to_beats` block.

**Purpose:** Extend each note's end time forward into the next gap when:
- The gap contains RMVPE-voiced F0 frames of the same semitone pitch, AND
- The gap duration is ≤ one quarter note.

This fills vibrato-trough drops and short consonant silences without touching
RMVPE's voicing threshold.

**New helper function** (add near `_merge_same_pitch`):

```python
def _gap_fill(
    raw: list[tuple[float, float, int]],
    f0_hz: np.ndarray,
    max_gap_s: float,
) -> list[tuple[float, float, int]]:
    """Extend note end times into adjacent same-pitch voiced gaps.

    For each consecutive pair of notes (n0, n1) where n0.pitch == n1.pitch:
    - Compute the gap [n0.end, n1.start].
    - If gap duration <= max_gap_s AND at least one F0 frame in the gap is
      voiced at n0.pitch (±0 semitones after median filter + snap), extend
      n0.end to n1.start and merge into a single note.
    Runs a single forward pass (no iteration needed — merges are monotonic).
    """
    if not raw:
        return raw
    out: list[tuple[float, float, int]] = []
    i = 0
    while i < len(raw):
        if i + 1 < len(raw):
            s0, e0, n0 = raw[i]
            s1, e1, n1 = raw[i + 1]
            gap = s1 - e0
            if n0 == n1 and 0 < gap <= max_gap_s:
                # Check whether any F0 frame in the gap snaps to n0.
                gap_start_frame = int(round(e0 / _HOP_S))
                gap_end_frame = int(round(s1 / _HOP_S))
                gap_f0 = f0_hz[gap_start_frame:gap_end_frame]
                lo = librosa.note_to_hz("C2")
                hi = librosa.note_to_hz("C6")
                voiced_gap = gap_f0[(gap_f0 > 0) & (gap_f0 >= lo) & (gap_f0 <= hi)]
                if len(voiced_gap) > 0:
                    snap = np.round(
                        12 * np.log2(np.maximum(voiced_gap, 1e-6) / 440.0) + 69
                    ).astype(int)
                    if np.any(snap == n0):
                        # Fill: extend n0 to cover the gap, merge with n1.
                        out.append((s0, e1, n0))
                        i += 2
                        continue
        out.append(raw[i])
        i += 1
    return out
```

**Call site** — add after the second `_merge_same_pitch` call:

```python
# F0-guided gap-fill: extend note boundaries into short same-pitch voiced gaps.
raw = _gap_fill(raw, f0_hz, quarter_s)
```

Note: `f0_hz` at this point is already median-filtered (applied at the top of
`_pitch_track_to_notes`). The gap-fill uses the raw continuous Hz values to
verify voicing, not the discretized `midi_track` array.

---

### Step 3 — Validation

Run `scripts/eval_vocal_midi.py` against the Adounravel track and check all
six success-criteria metrics.

**Command:**
```bash
source .venv/bin/activate
python scripts/eval_vocal_midi.py \
  "outputs/2026-04-26_12-34-26_Adounravel_歌いました/Adounravel_歌いました.mid" \
  "outputs/2026-04-26_12-34-26_Adounravel_歌いました/stems/htdemucs/eZuDklxsaRg/vocals.wav" \
  "outputs/2026-04-26_12-34-26_Adounravel_歌いました/extracted/eZuDklxsaRg.wav"
```

**Pass criteria (from proposal):**

| Metric | Current (baseline) | Target | Pass? |
|--------|-------------------|--------|-------|
| Coverage ratio | 71.2% | ≥ 90% | TBD |
| Median note duration | 150 ms | ≥ 300 ms | TBD |
| IOI coefficient of variation | 2.84 | ≤ 1.0 | TBD |
| Hallucination ratio | 13.3% | ≤ 15% | TBD |
| Frames within ±0.5 semitones | 80.3% | ≥ 85% | TBD |
| Duration mismatch (MIDI vs audio) | 10.76 s | ≤ 1.0 s | TBD (separate root cause) |

> Note: the MIDI being evaluated must be regenerated after the code change.
> The existing `.mid` file was produced by the old pipeline and reflects the
> pre-fix metrics. Run the pipeline again to produce a new MIDI before
> running the eval script.

**If a metric regresses:** revert the failing change independently and
revalidate. The two changes are independent and can be applied separately.

---

### Step 4 — Diagnose MIDI duration truncation (separate)

After the primary fix is validated, check why MIDI ends 10.76 s before the
audio:

```bash
python - <<'EOF'
import librosa
y, sr = librosa.load(
    "outputs/2026-04-26_12-34-26_Adounravel_歌いました/stems/htdemucs/eZuDklxsaRg/vocals.wav",
    sr=None,
)
print(f"Vocals stem duration: {len(y)/sr:.2f} s")
EOF
```

If the vocals stem is shorter than the mix (240.37 s), the truncation is a
demucs output issue (not a segmentation issue) and should be filed as a
separate bug.

---

## Non-Implementation Notes

- Do not change `_ONSET_DELTA`, `_ONSET_HOP`, or `thred` during this fix.
- Do not add new dependencies.
- The `_gap_fill` helper must handle the empty-list edge case (`if not raw: return raw`).
- After validation passes, the `/implement` agent handles the commit per git-workflow.md.
