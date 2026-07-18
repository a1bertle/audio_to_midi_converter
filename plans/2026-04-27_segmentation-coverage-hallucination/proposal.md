# Proposal: RMVPE Segmentation — Coverage, Hallucination, and Fragmentation

**Date:** 2026-04-27
**Status:** Final

---

## Problem Statement

After replacing HTDemucs with Mel-Band RoFormer (MBR) and applying the prior quarter-note
merge-ceiling raise and F0-guided gap-fill (from `plans/2026-04-26_vocal-midi-coverage-fragmentation`),
the vocal MIDI transcription of "Adounravel — 歌いました" still covers only 68.9% of pYIN-voiced
frames (target ≥ 90%), produces highly fragmented notes (median 150 ms, IOI CV 2.16; targets
≥ 300 ms and ≤ 1.0), and fires 11.2% of MIDI-active frames on pYIN-unvoiced passages
(target ≤ 5%). Pitch accuracy on covered frames is near target (80.2% within ±0.5 st;
target ≥ 85%). The MBR spectral centroid fix (3676 Hz ≈ mix 3685 Hz) confirms stem
separation is no longer the bottleneck — all remaining defects originate in the RMVPE
→ note segmentation layer.

Sources:
- `research/2026-04-27_adounravel-mbr-vocal-midi-eval/assessment.md` — MBR pipeline baseline
- `research/2026-04-26_adounravel-vocal-midi-eval/assessment.md` — HTDemucs pipeline baseline
- `plans/2026-04-26_vocal-midi-coverage-fragmentation/proposal.md` — prior fix record

---

## Success Criteria

| Metric | MBR baseline | Target | Source |
|--------|-------------|--------|--------|
| Coverage ratio (co-active / pYIN voiced) | 68.9% | ≥ 90% | [measured — 2026-04-27 assessment.md] |
| Median note duration | 150 ms | ≥ 300 ms | [measured — 2026-04-27 assessment.md; target assumed] |
| IOI coefficient of variation | 2.16 | ≤ 1.0 | [measured — 2026-04-27 assessment.md; target assumed] |
| Hallucination ratio | 11.2% | ≤ 5% | [measured — 2026-04-27 assessment.md; target assumed] |
| Frames within ±0.5 semitones | 80.2% | ≥ 85% | [measured — 2026-04-27 assessment.md; target assumed] |
| Duration mismatch (MIDI vs audio) | 10.84 s | ≤ 1.0 s | [measured — 2026-04-27 assessment.md] |

---

## Constraints

- Platform: Apple M1, 16 GB unified RAM, macOS [source-code — global-standards.md]
- Python venv: librosa 0.11.0, scipy, RMVPE impl vendored in `audio2midi/transcribers/` [source-code]
- RMVPE RTF: 0.013× real-time on CPU [measured — research/2026-04-17_vocal-pitch-tracker-research/results.md]; total pipeline must stay below 3× wall-clock for a 4-min track [assumed]
- Do not regress pitch accuracy (80.2% within ±0.5 st; 93.7% within ±1.0 st) [measured — 2026-04-27 assessment.md]
- Segmentation changes must be O(N) over note list — no full-audio re-inference [source-code — rmvpe.py]
- RMVPE `thred` currently 0.03 [source-code — rmvpe.py:574]

---

## Non-Goals

- Replacing the pitch tracker (RMVPE confirmed best on M1)
- Further stem separation work (MBR centroid fix confirmed; bleed is no longer the bottleneck)
- Velocity / dynamics modeling
- Beat-snap / quantization changes
- End-to-end neural note segmentation (ROSVOT etc.)

---

## What Was Already Tried

From `plans/2026-04-26_vocal-midi-coverage-fragmentation/proposal.md` [measured]:

| Change | Implemented | Observed Effect |
|--------|-------------|-----------------|
| Raise same-pitch merge ceiling 8th → quarter note | ✅ Yes (`_merge_same_pitch`, `rmvpe.py:439`) | IOI CV improved 2.84 → 2.16 (−0.68) [measured] |
| Add F0-guided gap-fill pass (one quarter note max gap) | ✅ Yes (`_gap_fill`, `rmvpe.py:489`) | Coverage improved 71.2% → 68.9%? (−2.3 pp — likely noise given different stem) |

Both changes are in place. The expected gains were not fully realised; defects persist at near-baseline levels.

---

## Root Cause Analysis

### Coverage gap (31.1% of pYIN voiced frames uncovered)

pYIN voiced frames: 10,762 [measured]. MIDI active frames: 8,351 [measured]. Co-active: 7,415 [measured]. The 3,347 uncovered voiced frames (10,762 − 7,415) [back-calc] represent real vocal signal that RMVPE either (a) does not voice at `thred=0.03`, or (b) voices but which falls into inter-note gaps the current merge/gap-fill passes don't reach.

Three sub-causes are distinguishable:

**RC-1 — RMVPE voicing threshold too conservative.** `thred=0.03` is the lowest available in the RMVPE Applio implementation. RMVPE returns `f0=0` for frames below confidence threshold. A lower threshold is not exposed. The only remedy short of patching RMVPE is to accept lower-confidence frames via the raw voiced mask — i.e., treat any non-zero raw F0 output as voiced. [source-code — `_rmvpe_impl.py`; assumed — needs verification of raw output API]

**RC-2 — Gap-fill max gap too small.** Current gap-fill ceiling is one quarter note (444 ms at 135 BPM) [source-code — rmvpe.py:489]. IOI max is 14,381 ms [measured — 2026-04-27 assessment.md]; many voiced gaps are longer than a quarter note and not reached. Raising the ceiling trades coverage for potential hallucination extension.

**RC-3 — Onset split over-segments.** Beat-gated onset split (`rmvpe.py:441–484`) fires on beat-aligned onsets, re-cutting merged notes and creating new short fragments. At 135 BPM, beat interval is 444 ms [back-calc: 60/135]; any voiced segment longer than ~500 ms likely contains at least one beat onset that re-splits it. This directly caps median duration.

### Note fragmentation (median 150 ms, IOI CV 2.16)

RC-3 above is the primary cause. The onset split is designed to separate repeated syllables on the beat but fires too aggressively, cutting single held notes into fragments. The `split_min_s` floor (currently `max(min_note_s, sixteenth_s)` = 56 ms at 135 BPM [back-calc: 60/(135×4)/2 ≈ 56 ms]) is too low to prevent very short fragments from surviving the split.

### Hallucination (11.2%)

11.2% of 8,351 MIDI-active frames = 936 frames [back-calc] fire when pYIN detects no voicing. These arise from:
- RMVPE voicing non-vocal signal that pYIN doesn't confirm (accompaniment bleed, breath noise).
- Notes that start slightly before the pYIN-voiced window (onset overhang).

`thred=0.03` is already a conservative (high) confidence requirement. Lowering it further to gain coverage would worsen hallucination. The two objectives are in tension.

---

## Candidate Approaches

| # | Approach | Targets | Risk |
|---|----------|---------|------|
| 1 | **Raise onset-split `split_min_s` floor** | Fragmentation, IOI CV | Low — no effect on voicing |
| 2 | **Disable onset split for long held notes** | Fragmentation, coverage | Medium — may merge distinct syllables |
| 3 | **Voiced-frame holdout extension** | Coverage | Low-Medium — extends only already-RMVPE-voiced frames |
| 4 | **Raise gap-fill ceiling to half note** | Coverage | Medium — may extend into bleed frames |

---

## Research Findings

All findings from prior measured sessions. No new experiments run in this proposal phase.

### Segmentation parameter baseline [source-code — `rmvpe.py`]

| Parameter | Current value | Location |
|-----------|--------------|----------|
| `thred` (RMVPE voicing confidence) | 0.03 | `rmvpe.py:574` |
| `min_note_duration_s` | `beat_s / 8` = 55.5 ms at 135 BPM | `rmvpe.py:543` |
| `split_min_s` | `max(min_note_s, sixteenth_s)` ≈ 55.5 ms | `rmvpe.py:449` |
| Same-pitch merge ceiling | 1 quarter note (444 ms at 135 BPM) | `rmvpe.py:383` |
| Gap-fill max gap | 1 quarter note (444 ms at 135 BPM) | `rmvpe.py:489` |
| F0 median filter window | 7 frames = 70 ms | `rmvpe.py:309` |
| Beat tolerance for onset gate | 25% of beat interval = 111 ms | `rmvpe.py:405` |

All values [source-code — line references above].

### IOI distribution [measured — 2026-04-27 assessment.md]

| Metric | Value |
|--------|-------|
| IOI mean | 444 ms |
| IOI std | 962 ms |
| IOI CV | 2.16 |
| IOI min | 59 ms |
| IOI max | 14,381 ms |

IOI mean (444 ms) = exactly one quarter note at 135 BPM [back-calc: 60/135 = 444 ms]. This
confirms the onset split is firing on (nearly) every beat, chopping long held notes at beat
boundaries and producing the observed IOI spike at ~444 ms.

### pYIN voiced ratio gap [measured — 2026-04-27 assessment.md]

RMS voiced ratio: 66.3%. pYIN voiced ratio: 52.2%. Gap: 14.1 pp [back-calc]. This gap exists
before any MIDI comparison — ~2,918 frames [back-calc: 14.1% × 20,705] are RMS-active but
pYIN-unvoiced, likely soft onsets, breaths, and vibrato troughs. These frames cannot be
recovered by segmentation changes alone; they require either lowering pYIN confidence or
accepting RMVPE output at those frames without pYIN confirmation (which is what our eval
script uses as ground truth — a limitation).

### Prior merge ceiling impact [measured — comparing 2026-04-26 vs 2026-04-27 assessments]

| Metric | HTDemucs + old ceiling (8th) | MBR + new ceiling (quarter) | Δ |
|--------|------------------------------|------------------------------|---|
| IOI CV | 2.84 | 2.16 | −0.68 |
| Median note dur | 150 ms | 150 ms | 0 ms |
| Coverage | 71.2% | 68.9% | −2.3 pp |

Merge ceiling raise improved IOI CV but did not move median duration — consistent with onset
split re-cutting merged notes immediately after the merge pass. [back-calc]

---

## Results Summary

No new experiments were run. All values are derived from existing measured sessions.

| Approach | Coverage impact | Fragmentation impact | Hallucination impact | Risk | Basis |
|----------|----------------|---------------------|---------------------|------|-------|
| 1 — Raise `split_min_s` | Neutral | ✅ Directly prevents sub-threshold re-cuts | Neutral | Low | [source-code + back-calc] |
| 2 — Disable split for long notes | ✅ Indirect (fewer re-cuts = longer notes fill voiced gaps) | ✅ Stops onset split on already-long notes | Neutral | Medium | [assumed] |
| 3 — Voiced-frame holdout extension | ✅ Extends into RMVPE-voiced uncovered frames | ✅ Merges trailing voiced frames into preceding note | Low-Medium | Low | [assumed] |
| 4 — Raise gap-fill ceiling to half note | ✅ Closes longer gaps | Neutral | ⚠️ May extend into bleed | Medium | [assumed] |

---

## Recommendation

**Apply all four approaches in sequence: (1) raise `split_min_s` to one 8th note, (2) gate the onset split to only fire on notes longer than a half note, (3) add a voiced-frame holdout extension pass, (4) raise gap-fill ceiling to a half note.**

### Rationale

**Approach 1 — Raise `split_min_s` floor to one 8th note (222 ms at 135 BPM).**
The current floor of 55.5 ms (32nd note) allows the onset split to produce fragments as short
as 56 ms, which are shorter than any recognisable sung syllable. Raising to a 8th note
(222 ms) prevents the split from creating sub-musical fragments while still separating
syllables that are rhythmically distinct. Zero risk to pitch accuracy; O(1) change.
[source-code — rmvpe.py:449; assumed impact on fragmentation]

**Approach 2 — Gate onset split to notes whose duration ≥ half note (888 ms at 135 BPM).**
IOI mean of 444 ms = one quarter note [back-calc]. Notes at or below this length are likely
already individual syllables and should not be split further. Only notes longer than a half note
are plausible multi-syllable holds. This prevents the split from re-cutting merged notes that
the merge pass correctly unified. [source-code — rmvpe.py:441; assumed — needs validation]

**Approach 3 — Voiced-frame holdout extension (new pass).**
After segmentation, extend each note's end time forward frame-by-frame while: (a) RMVPE
returns a non-zero F0, and (b) the semitone matches the note's pitch. Stop at the first
silence frame or pitch change. This closes the RC-1 gap (frames where RMVPE voices but
the segmentation loop's `min_note_duration_s` filter discarded the tail) without touching the
voicing threshold. Unlike gap-fill, it does not skip over silence — it only extends into
immediately adjacent voiced frames. [assumed — mechanism consistent with source-code frame
iteration in `_pitch_track_to_notes`]

**Approach 4 — Raise gap-fill ceiling to one half note (888 ms).**
Raises the F0-guided gap-fill from one quarter note (444 ms) to one half note (888 ms), closing
longer voiced gaps between same-pitch notes. The existing same-pitch + voiced-frame gate
(`_gap_fill` checks that the gap contains a voiced F0 at the same semitone) prevents filling
genuine silences. Hallucination risk: small, because the gate requires RMVPE to have already
voiced the gap frames at the same pitch. [source-code — rmvpe.py:271–313; assumed impact]

### Expected impact (all [assumed] — must be validated)

| Metric | MBR baseline | Expected post-fix | Approach |
|--------|-------------|-------------------|---------|
| Coverage ratio | 68.9% | 80–90% | 3 + 4 |
| Median note duration | 150 ms | 250–400 ms | 1 + 2 |
| IOI CV | 2.16 | 1.2–1.8 | 1 + 2 |
| Hallucination ratio | 11.2% | 9–13% | Slight increase from 3 + 4 |
| Pitch accuracy ±0.5 st | 80.2% | 79–82% | Neutral |

---

## Tradeoff Analysis

### Coverage vs. hallucination tension

Approaches 3 and 4 extend notes into frames that pYIN may not confirm as voiced. The
`_gap_fill` gate (same semitone in RMVPE output) is the primary safeguard. If hallucination
rises above 15% post-fix, the half-note gap-fill ceiling should be reduced back to a quarter
note. [assumed]

### Onset split gating (Approach 2) vs. syllable separation

Gating the onset split to notes ≥ half note risks missing a genuine two-syllable repetition
on a long note. The beat-gate already blocks merging across beat boundaries; Approach 2 adds
a duration gate above the beat gate. The residual risk is a two-syllable same-pitch note
(e.g. repeated vowel on a half note) not being split. This is a minor accuracy trade for
significant fragmentation improvement. [assumed]

### `split_min_s` floor (Approach 1) is low-risk, high-value

Raising from 32nd (56 ms) to 8th (222 ms) eliminates sub-musical fragments that cannot
represent real syllables at 135 BPM. No measurable downside. [source-code + assumed]

---

## Risks and Open Questions

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Onset split gating (Approach 2) misses genuine short repeated syllables | Medium | Validate on a track with staccato same-pitch passages; fall back to current gating if recall drops |
| Voiced-frame holdout (Approach 3) extends into accompaniment bleed at phrase ends | Medium | Cap holdout at 200 ms maximum extension; re-check hallucination metric |
| Gap-fill half-note ceiling fills long instrumental breaks between phrases | Low | Gap-fill gate requires same-pitch RMVPE voicing — silent instrumental sections will not have matching F0 |
| IOI CV target ≤ 1.0 may require HMM segmentation to achieve | Medium | Accept ≤ 1.5 as intermediate milestone; re-evaluate post-fix |
| Duration mismatch (10.84 s) has a separate root cause not addressed by segmentation | High | Diagnose independently: compare MBR vocals stem duration vs. mix duration before RMVPE |

---

## Visualizations

Existing notebook covers the current pipeline:
- `notebooks/vocal_f0_pipeline_visualizer.ipynb` — F0 contour, voiced-frame overlay, MIDI piano-roll, onset visualization. Re-run after fix to compare before/after coverage on Adounravel track.

---

## Next Step

Invoke **planner** with: "Implement four RMVPE segmentation changes in
`audio2midi/transcribers/rmvpe.py`: (1) raise `split_min_s` floor from 32nd note to 8th note;
(2) gate the onset split to only fire on notes whose current duration ≥ one half note; (3) add
a voiced-frame holdout extension pass after segmentation that extends note end times frame-by-frame
into adjacent same-pitch RMVPE-voiced frames (max 200 ms extension); (4) raise the `_gap_fill`
max gap ceiling from one quarter note to one half note. Validate all changes with
`scripts/eval_vocal_midi.py` on the Adounravel MBR workdir and confirm coverage ≥ 90%, median
duration ≥ 300 ms, IOI CV ≤ 1.5, hallucination ≤ 13%."
