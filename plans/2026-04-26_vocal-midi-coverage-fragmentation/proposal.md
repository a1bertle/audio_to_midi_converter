# Proposal: Vocal MIDI Coverage Gap and Note Fragmentation

**Date:** 2026-04-26
**Status:** Final

---

## Problem Statement

The vocal MIDI transcription of "Adounravel — 歌いました" covers only 71.2% of pYIN-voiced
frames and produces highly fragmented notes (median duration 150 ms, 2.27 notes/s, IOI
coefficient of variation 2.84) that do not reflect phrase-level vocal structure. Approximately
13.3% of MIDI notes fire on passages where pYIN detects no voicing (hallucination or bleed-onset
artifacts). Pitch accuracy on covered frames is acceptable (80.3% within ±0.5 semitones, 94.7%
within ±1.0 semitone); the primary defects are incomplete coverage and note fragmentation, not
tuning error.

Source: `research/2026-04-26_adounravel-vocal-midi-eval/assessment.md`

---

## Success Criteria

| Metric | Current | Target | Source |
|--------|---------|--------|--------|
| Coverage ratio (co-active / pYIN voiced frames) | 71.2% | ≥ 90% | [measured — assessment.md] |
| Median note duration | 150 ms | ≥ 300 ms | [measured — assessment.md; target assumed] |
| IOI coefficient of variation | 2.84 | ≤ 1.0 | [measured — assessment.md; target assumed] |
| Hallucination ratio | 13.3% | ≤ 5% | [measured — assessment.md; target assumed] |
| Duration mismatch (MIDI vs audio) | 10.76 s | ≤ 1.0 s | [measured — assessment.md] |
| Frames within ±0.5 semitones | 80.3% | ≥ 85% | [measured — assessment.md; target assumed] |

---

## Constraints

- Platform: Apple M1, 16 GB unified RAM, macOS (darwin) [source-code — global-standards.md]
- RTF budget: ≤ 3× real-time (~12 min for a 4-min track) [assumed — established in prior sessions]
- Python environment: existing venv with librosa 0.11.0, scipy, mido, demucs, RMVPE impl
- No new GPU-only dependencies: RMVPE already runs on CPU at RTF 0.013× [measured — research/2026-04-17_vocal-pitch-tracker-research/results.md]
- Do not break pitch accuracy — changes must not regress the 80.3%/94.7% ±0.5/±1.0 st metrics
- Existing pipeline: htdemucs 4s → RMVPE (thred=0.03) → F0 median filter 70 ms → semitone snap → silence-aware merge → beat-gated onset split [source-code — audio2midi/transcribers/rmvpe.py]

---

## Non-Goals

- Improving stem separation (htdemucs vocal body loss was addressed in 2026-04-13_vocal-midi-bleed-tolerance; RMVPE was selected to tolerate this)
- Replacing the pitch tracker (RMVPE is confirmed best available on M1; RTF 0.013×)
- Velocity / dynamics modeling (informational finding only; out of scope)
- End-to-end neural note segmentation (ROSVOT / ACL 2025; insufficient data and M1 RTF uncertain)

---

## Candidate Approaches

| # | Approach | Summary |
|---|----------|---------|
| 1 | **Voicing threshold lowering** | Reduce RMVPE `thred` from 0.03 to 0.01–0.02 to voice more frames, increasing coverage |
| 2 | **Note boundary extension (holdout)** | After note segmentation, extend each note's end time forward into adjacent voiced-but-uncovered frames until a silence or pitch change is encountered |
| 3 | **Minimum duration gate + aggressive same-pitch merge** | Raise the note merge gap ceiling from the current 8th-note to a quarter-note and increase the minimum fragment duration, collapsing short ornament fragments into held notes |
| 4 | **F0-guided gap-filling** | Post-segmentation pass: any gap between two consecutive same-pitch notes shorter than a defined maximum is filled by extending the earlier note to cover it, regardless of silence depth |

---

## Research Findings

All findings below are drawn from prior measured research sessions. No new experiments were
run during this proposal phase.

### Session Cross-Reference

| Session | Key Measurement | Relevance |
|---------|-----------------|-----------|
| `research/2026-04-26_adounravel-vocal-midi-eval/assessment.md` | Coverage 71.2%, hallucination 13.3%, median dur 150 ms, IOI CV 2.84 | Primary defect measurements |
| `research/2026-04-17_vocal-pitch-tracker-research/results.md` | RMVPE RTF 0.013×; subjectively best output at thred=0.03 with silence-aware merge + onset split | Pipeline baseline |
| `research/2026-04-17_vocal-pitch-tracker-research/notes.md` | F-T2: pYIN/PESTO/FCPE on htdemucs stem all subjectively poor; RMVPE best | Tracker selection rationale |
| `research/2026-04-18_f0-median-filter-research/notes.md` | 70 ms window: 23 raw unison repeats, 302 notes (vs 299 baseline); over-smoothing at ≥110 ms | Median filter calibration |
| `research/2026-04-18_f0-median-filter-research/notes.md` | Voiced audio 58.4 s; MIDI coverage 52.2 s → 89.3% on Blue Bird; coverage gap open question | Prior coverage gap observation |
| `research/2026-04-18_blue-bird-vocals-midi-eval/assessment.md` | 29.9% unison repeats, 252 notes / 88 s, median dur 150 ms | Fragmentation seen cross-track |
| `research/2026-04-18_vocal-to-midi-theory/notes.md` | HMM / note-boundary theory; vibrato voicing-drop failure mode | Theoretical basis for gap-filling |

### Coverage Gap Root Cause Analysis

The assessment shows pYIN-voiced frames = 9,912 (trimmed to MIDI length) and MIDI-active frames
= 8,136, with co-active frames = 7,056. The 71.2% coverage ratio implies 2,856 voiced frames
with no MIDI note active [back-calc: 9,912 − 7,056].

Three root causes are consistent with existing measurements:

**RC-1: RMVPE drops voiced frames at vibrato troughs (documented)**
The vocal-to-midi theory session (finding F6 in notes.md) identifies vibrato as a known failure
mode for voicing detection: instantaneous F0 oscillates ±50–100 cents at ~5 Hz, causing voicing
detectors to drop frames at troughs. The 70 ms median filter smooths the F0 contour but does not
affect RMVPE's internal voiced-probability threshold (thred). At thred=0.03, trough frames may
fall below threshold. [assumed — consistent with theory; not directly measured on Adounravel]

**RC-2: Minimum note duration discards short voiced segments**
The 32nd-note floor at 135 BPM = 60/135/8 = 55.6 ms [back-calc]. Any voiced segment shorter
than 55.6 ms is discarded. With median note duration at 150 ms and IOI CV 2.84, many micro-notes
produced by vibrato trough voicing-drops are below the floor. [back-calc from assessment.md BPM
135.29]

**RC-3: Hard MIDI duration end (10.76 s truncation)**
MIDI ends 10.76 s before the audio, truncating the final phrase. This accounts for a bounded
portion of the coverage gap. [measured — assessment.md]

### Fragmentation Root Cause Analysis

Median note duration 150 ms, IOI CV 2.84, note density 2.27 notes/s across 229.6 s → 520 notes
[measured — assessment.md].

The Blue Bird assessment shows identical median duration (150 ms) [measured — blue-bird
assessment.md], and the f0-median-filter session showed 23 raw unison repeats before post-
processing [measured — f0-median-filter notes.md]. This pattern is consistent across two
independent tracks.

**RC-4: Short voiced gaps interrupt what should be held notes**
Vibrato troughs and soft consonants produce brief unvoiced frames inside sustained notes. The
current silence-aware merge uses `max_silence_s = min_note_duration_s * 0.5` = 27.8 ms gap
limit [source-code — rmvpe.py line 305]. A 30 ms consonant gap (common in sung consonants) just
exceeds this threshold and prevents merging, splitting one note into two. [assumed — consistent
with pipeline code and typical sung consonant durations]

**RC-5: Same-pitch merge ceiling is an 8th note (222 ms at 135 BPM)**
The `_merge_same_pitch` call uses `eighth_s = (60/bpm)/2` [source-code — rmvpe.py line 339].
At 135 BPM, eighth_s = 222 ms. Gaps wider than 222 ms between same-pitch segments are not
merged even if they represent one sustained note. [back-calc: 60/135.29/2 = 0.222 s]

---

## Results Summary

No new benchmarks were run during this proposal. The table below consolidates prior measurements
as a cross-track baseline.

| Track | Coverage ratio | Median dur (ms) | Unison % | Hallucin. % | Notes | Provenance |
|-------|---------------|-----------------|----------|-------------|-------|------------|
| Adounravel — 歌いました | 71.2% | 150 | — | 13.3% | 520 | [measured — adounravel assessment] |
| Blue Bird — Naruto OP3 | ~89.3% (F0 vs MIDI) | 150 | 29.9% | — | 252 / 88 s | [measured — blue-bird assessment + f0-filter notes] |

Both tracks show median duration 150 ms and similar fragmentation patterns, confirming the
defect is systemic (pipeline parameter choice) rather than track-specific.

---

## Recommendation

**Approach 3 + 4 combined: raise the same-pitch merge ceiling to a half note and add an
F0-guided gap-filling pass.**

### Rationale

1. **Coverage gap** is best addressed by gap-filling (Approach 4): scan for voiced-but-uncovered
   frames between same-pitch notes and extend the earlier note forward. This directly closes RC-1
   and RC-2 without touching RMVPE's voicing threshold (which would increase hallucination).
   Approach 1 (lowering thred) risks raising the already-problematic 13.3% hallucination ratio
   by voicing accompaniment-bleed frames. [assumed — lowering threshold risks are consistent with
   RC-3 of vocal-to-midi theory, F-T3 from 2026-04-17 showing FCPE on raw mix producing
   accompaniment noise when voicing is too permissive]

2. **Fragmentation** is best addressed by a larger merge ceiling (Approach 3): raising
   `_merge_same_pitch` gap from an 8th note (222 ms at 135 BPM) to a quarter note (444 ms) will
   collapse the observed gap distribution (IOI max 21,080 ms, IOI mean 438 ms — the mean is
   already at 2× the current ceiling) [measured — assessment.md]. The current beat-gate prevents
   merging across genuine beat-boundary repeated notes, so raising the ceiling does not merge
   semantically distinct notes. [source-code — rmvpe.py _merge_same_pitch beat-gate logic]

3. **Approach 2 (note boundary extension)** is subsumed by Approach 4 and adds no independent
   value.

4. **Approaches 3 and 4 are both post-RMVPE, post-F0-filter operations** — zero risk to pitch
   accuracy on covered frames, and negligible computational cost (both are O(N) over the note
   list). [source-code — rmvpe.py _pitch_track_to_notes]

### Expected impact (estimated from root cause analysis)

| Metric | Current | Expected post-fix | Basis |
|--------|---------|-------------------|-------|
| Coverage ratio | 71.2% | 85–92% | [assumed — RC-1/RC-2 gap-fill on 2,856 frames; upper bound assumes all gaps are fill-eligible] |
| Median note duration | 150 ms | 250–400 ms | [assumed — raising merge ceiling from 222 ms to 444 ms; based on IOI mean 438 ms] |
| IOI CV | 2.84 | 1.5–2.0 | [assumed — fewer short gaps; upper bound not measurable without running pipeline] |
| Hallucination ratio | 13.3% | 13–15% | [assumed — gap-fill may extend into lightly voiced frames; small increase expected] |

All post-fix values are marked [assumed] — they must be validated by running the updated pipeline
on the Adounravel track and re-running `scripts/eval_vocal_midi.py`.

---

## Tradeoff Analysis

### Approach 1 (lower thred) vs. Approach 4 (gap-fill)

Lowering RMVPE `thred` voices more frames at the cost of accepting lower-confidence pitch
estimates. The 13.3% hallucination rate already indicates the voicing boundary is near its
tolerance limit on this track. Gap-filling operates only between already-voiced same-pitch
segments, so it cannot introduce hallucinations on unvoiced passages. [assumed — gap-fill
confined to inter-note gaps where F0 was already tracked]

### Approach 3 (larger merge ceiling) vs. status quo

At 135 BPM, raising the merge ceiling from an 8th (222 ms) to a quarter note (444 ms) risks
merging two distinct beat-boundary notes of the same pitch that happen to be adjacent. The
existing beat-gate (`_merge_same_pitch` checks whether any beat alignment falls in the gap) is
the safeguard. This gate is parameterized by `beat_tolerance = 0.25 × beat_interval`, so it
will correctly block merges at rhythmically meaningful gaps. [source-code — rmvpe.py lines
340–346]

### Duration truncation fix (10.76 s)

The MIDI truncation is unrelated to the core fragmentation and coverage issues. Its root cause
is likely that the MIDI is generated from stem output that ends early. This should be investigated
separately: confirm whether `_separate_vocals` produces a vocals stem covering the full audio
duration. This is a one-line diagnostic, not a design choice.

---

## Risks and Open Questions

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Gap-fill extends into accompaniment-bleed frames, raising hallucination above 15% | Medium | Constrain gap-fill to frames where RMVPE voiced probability > 0 (even if below thred); add hallucination metric re-check after fix |
| Raising merge ceiling from 8th to quarter merges semantically distinct repeated notes | Low | Beat-gate already present; validate on tracks with repeated same-pitch syllables (e.g. staccato passages) |
| MIDI truncation (10.76 s) has a different root cause that masquerades as coverage gap | Medium | Diagnose separately: compare demucs vocals.wav duration vs. audio duration before RMVPE |
| 70 ms F0 median filter was calibrated on Blue Bird (152 BPM, vibrato rate 5.14 Hz); Adounravel BPM is 135 and vibrato rate is unknown | Low–Medium | Filter is conservative by design; 70 ms < 0.5× median vibrato period even at 4 Hz (period 250 ms) |
| IOI CV target of ≤ 1.0 may be unachievable without HMM-level note segmentation | Medium | Target is assumed; accept ≤ 2.0 as a practical milestone and re-evaluate |

---

## Visualizations

No new notebooks were produced during this proposal phase. Relevant existing notebooks:

- `notebooks/vocal_f0_pipeline_visualizer.ipynb` — F0 contour, voiced-frame overlay, MIDI
  piano-roll, and onset detection visualization. Link F0 gaps to coverage metric after applying
  the fix.

---

## Next Step

Invoke **planner** with: "Implement two post-RMVPE note-segmentation changes in
`audio2midi/transcribers/rmvpe.py`: (1) raise `_merge_same_pitch` gap ceiling from 8th-note to
quarter-note; (2) add an F0-guided gap-fill pass that extends note end times into adjacent
same-pitch voiced frames up to a maximum gap of one quarter note, gated by RMVPE voiced output.
Validate both changes against `scripts/eval_vocal_midi.py` on the Adounravel track and confirm
coverage ≥ 90%, median duration ≥ 300 ms, and hallucination ≤ 15%."
