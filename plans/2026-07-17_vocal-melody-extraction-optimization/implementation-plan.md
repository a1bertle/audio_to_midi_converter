# Implementation Plan: Vocal Melody Extraction Optimization

**Date:** 2026-07-17
**Status:** Implemented (uncommitted; see results note below)
**Goal:** Close the remaining gaps in vocal melody MIDI extraction — coverage,
fragmentation, hallucination — building on the 2026-04-27 proposal
(`plans/2026-04-27_segmentation-coverage-hallucination/proposal.md`).

---

## Current State

Prior research established (all [measured] unless noted):

1. **Stem separation is solved.** Mel-Band RoFormer (MBR) replaced HTDemucs in
   April 2026. Vocals-stem spectral centroid now matches the mix (3676 Hz vs
   3685 Hz), vs HTDemucs's 4427 Hz skew. RTF 0.87× on M1 MPS.
   (`research/2026-04-26_vocal-stem-separation-alternatives/notes.md`,
   `research/2026-04-27_adounravel-mbr-vocal-midi-eval/assessment.md`)
2. **Remaining defects are in RMVPE → note segmentation** (Adounravel MBR
   baseline): coverage 68.9% (target ≥ 90%), median note 150 ms (target
   ≥ 300 ms), IOI CV 2.16 (target ≤ 1.0, intermediate ≤ 1.5), hallucination
   11.2% (target ≤ 5%), pitch ±0.5 st 80.2% (target ≥ 85%), MIDI/audio
   duration mismatch 10.84 s (target ≤ 1.0 s).
3. **The four fixes from the 2026-04-27 proposal are already implemented** in
   the uncommitted working-tree diff of `audio2midi/transcribers/rmvpe.py`
   (+191 lines): raised `split_min_s` floor, half-note gate on the onset
   split, F0 gap-fill ceiling raise, voiced-frame holdout extension. They were
   **never evaluated and never committed**, and the implementation contains a
   bug (Step 1 below).
4. The prior eval workdir (`outputs/2026-04-27_22-03-59_Adounravel_歌いました/`)
   no longer exists; `outputs/` is empty. The pipeline must be rerun to
   evaluate anything.

---

## Scope / Non-Goals

**In scope:** segmentation bug fix, end-to-end re-evaluation, hallucination
gating, duration-truncation diagnosis, multi-track validation, committing the
pending work.

**Non-goals** (carried from 2026-04-27 proposal):
- Replacing RMVPE (confirmed best pitch tracker on M1).
- Further stem-separation model work — MBR centroid fix is confirmed. Only
  revisit (Step 6 contingency) if hallucination frames are shown to correlate
  with accompaniment bleed.
- Velocity/dynamics modeling; beat-snap changes; neural note segmentation.

---

## Steps

### Step 1 — Fix the half-note constant bug (blocking)

`audio2midi/transcribers/rmvpe.py` defines, in two places:

```python
half_note_s = 60.0 / bpm  # one half note = two beats at tempo
```

- `rmvpe.py:462` — onset-split duration gate
- `rmvpe.py:498` — `_gap_fill` ceiling

`60.0 / bpm` is **one beat** (a quarter note, 444 ms at 135 BPM), not a half
note (888 ms). Both the onset-split gate and the gap-fill ceiling are running
at half the value the 2026-04-27 proposal specified. Fix:

```python
half_note_s = 2.0 * 60.0 / bpm
```

Consequence of the bug as-is: the onset split still fires on any note
≥ 444 ms (exactly the IOI mean — i.e. it still re-cuts most merged holds),
and gap-fill closes no gap longer than a quarter note, same as before the
"raise". This plausibly explains why the proposal's expected gains would not
materialize.

### Step 2 — Regenerate the Adounravel workdir and evaluate the pending changes

The baseline metrics are all against "Adounravel — 歌いました" (YouTube
`eZuDklxsaRg`). Comparability requires the same track.

1. Run the pipeline on the Adounravel source to produce a fresh timestamped
   workdir (MBR stems are saved automatically since commit 304d388).
2. Evaluate: `python scripts/eval_vocal_midi.py --workdir outputs/<new>/`
3. Record results in a new research session
   `research/2026-07-17_adounravel-segfix-eval/` per
   `base/research-workflow.md`, with a before/after table against the
   2026-04-27 baseline.

**Pass criteria (from the proposal):** coverage ≥ 90%, median duration
≥ 300 ms, IOI CV ≤ 1.5 (intermediate milestone), hallucination ≤ 13%,
pitch ±0.5 st ≥ 79%.

### Step 3 — Diagnose the 10.84 s duration truncation (independent root cause)

MIDI ends 10.84 s before the audio does — the final phrase is likely dropped.
The proposal flagged this as unaddressed by segmentation changes.

1. Compare durations at each stage: mix WAV → MBR vocals stem → RMVPE F0
   array length → last note end time.
2. The mismatch is either (a) stem/F0 truncation upstream, or (b) the final
   voiced region being discarded by segmentation (e.g. trailing notes below
   `min_note_duration_s`, or no trailing onset).
3. Fix at the stage identified. **Pass:** duration mismatch ≤ 1.0 s.

### Step 4 — Hallucination gate (targets 11.2% → ≤ 5%)

Neither the pending changes nor prior work directly attacks hallucination;
Steps 1–2 may even raise it slightly (extension passes). After Step 2
measurement, if hallucination > 5%:

1. **Diagnose first:** dump hallucinated frame regions (MIDI-active,
   pYIN-unvoiced) from the eval and inspect them against the vocals-stem RMS
   and the F0 contour in `notebooks/vocal_f0_pipeline_visualizer.ipynb`.
   Classify: accompaniment bleed vs breath/consonant noise vs onset overhang
   (note starts slightly before pYIN voicing).
2. **Candidate fix A — stem-energy gate:** require vocals-stem RMS above a
   threshold (e.g. −40 dBFS, matching the eval's own voiced criterion) for
   frames to seed or extend a note. Cheap, O(N), directly kills
   silent-region hallucination.
3. **Candidate fix B — trim onset overhang:** clip note starts forward to the
   first RMVPE-voiced frame of the segment.
4. Re-run Step 2 eval after each candidate; keep what the numbers justify.
   Guard: coverage must not drop below its Step 2 value by > 2 pp.

### Step 5 — Multi-track validation (guard against single-track overfitting)

All tuning so far is against one J-pop track at 135 BPM. Several constants
(merge ceiling, split gate, gap-fill) are tempo-derived but were validated on
one tempo/style only.

1. Run the pipeline + `scripts/eval_vocal_midi.py` on at least two more
   tracks with different character — e.g. `inputs/Foals - My Number.mp3`
   (indie rock, male vocal, already on disk) and one slow ballad (< 90 BPM).
2. Record per-track metrics in the same research session. No hard targets on
   the new tracks; flag any metric that is drastically worse than Adounravel
   (indicates overfit constants) and note it for follow-up.

### Step 6 — Contingency: revisit stem separation only if evidence demands it

If Step 4's diagnosis shows hallucinated frames are dominated by
**accompaniment bleed** (pitched instrument energy in the vocals stem, not
breath/overhang), then and only then:

- Benchmark one alternative checkpoint via the already-installed
  `audio-separator` (e.g. BS-RoFormer variant) against the current
  MelBandRoformerSYHFTV3Epsilon.ckpt using
  `tests/evaluation/evaluate_stems.py` + the vocal MIDI eval.
- Adopt only if hallucination improves ≥ 2 pp without coverage loss.

### Step 7 — Commit

Per `base/git-workflow.md`: `git pull --rebase`, then commit the pending
`rmvpe.py` work + fixes with eval evidence in the message (template with
`Feature changes` / `Bug Fixes` sections — the half-note constant fix goes
under Bug Fixes). Update `README.md` if behavior described there changed.
Keep `CLAUDE.md`/`AGENTS.md` in sync if touched.

---

## Affected Files

| File | Change |
|------|--------|
| `audio2midi/transcribers/rmvpe.py:462,498` | Step 1 bug fix; Steps 3–4 fixes as diagnosed |
| `scripts/eval_vocal_midi.py` | Possibly extend to dump hallucinated-frame regions (Step 4.1) |
| `research/2026-07-17_adounravel-segfix-eval/` | New research session (Steps 2–5 measurements) |
| `README.md` | Update if pipeline behavior changes |

---

## Validation Strategy

- Primary: `scripts/eval_vocal_midi.py` on regenerated Adounravel workdir,
  before/after table vs 2026-04-27 baseline (Step 2 pass criteria).
- Secondary: `notebooks/vocal_f0_pipeline_visualizer.ipynb` re-run for visual
  before/after of coverage and fragmentation.
- Generalization: per-track metric table across ≥ 3 tracks (Step 5).
- Every metric claim recorded with provenance tags per research workflow.

## Sequencing & Risk

Steps 1–2 first (cheap, unblocks everything: can't measure anything until the
bug is fixed and a workdir exists). Step 3 is independent and can interleave.
Step 4 only proceeds on measured evidence from Step 2. Step 6 is contingent.
Biggest risk: even at the correct 888 ms values, coverage may stall below 90%
because ~14 pp of the gap is pYIN-vs-RMS voicing disagreement
(soft onsets/breaths) that segmentation cannot recover — if Step 2 lands at
80–85%, revisit the coverage target or the eval's ground-truth definition
before adding more mechanism.

---

## Implementation Result — 2026-07-17

Implemented and evaluated. The accepted Adounravel result reached 72.93% coverage,
169 ms median duration, 2.04 IOI CV, 13.05% hallucination, 79.49% pitch accuracy
within 0.5 semitones, and 0.702 s vocal-end mismatch. The duration and pitch guards
pass; the aspirational coverage, fragmentation, and hallucination targets do not.

Full evidence, rejected-candidate results, and three-track validation are recorded
in `research/2026-07-17_adounravel-segfix-eval/notes.md`. Step 6 was not triggered
because accompaniment bleed did not dominate hallucinated regions. Step 7 remains
uncommitted because project policy forbids committing in the same response as code
edits.
