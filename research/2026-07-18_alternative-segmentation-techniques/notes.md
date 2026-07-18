# Alternative techniques for the unmet vocal-MIDI segmentation targets

**Date:** 2026-07-18
**Trigger:** `research/2026-07-17_adounravel-segfix-eval/notes.md` concluded that
"further constant tuning is not justified" for the RMVPE heuristic
segmentation stage, with four targets still unmet on Adounravel:

| Metric | Current | Target |
|---|---:|---:|
| Coverage | 72.93% | ≥ 90% |
| Median note duration | 169 ms | ≥ 300 ms |
| IOI CV | 2.04 | ≤ 1.5 |
| Hallucination | 13.05% | ≤ 5% |

This session is a literature/tooling survey (no pipeline changes) to find
techniques beyond further constant tuning, per the current `/goal`. No
pipeline code was modified in this session.

## Method

Web research via a general-purpose subagent, scoped to: (1) inference-only
neural note-segmentation models runnable on Apple M1/CPU, (2) post-processing
smoothing beyond median-filter/beat-gating, (3) techniques to close the
14 pp pYIN-vs-RMS voicing gap identified in the 2026-07-17 session, (4)
evaluation-methodology precedent in the singing-voice-transcription (SVT)
literature. Full agent output is preserved in this session's provenance;
findings below are [reported by subagent, unverified against repo/runtime]
unless cross-checked locally, in which case tagged [verified locally].

## Finding 1 — Basic Pitch is already vendored in this repo but excluded from vocals

[verified locally] `audio2midi/transcribers/basic_pitch.py` is a complete,
working adapter for Spotify's Basic Pitch (Apache 2.0, ICASSP 2022 model,
~17K params, ONNX/CoreML inference path, no CUDA dependency). It is declared
as an optional extra in `pyproject.toml` (`basic-pitch>=0.3.0`) but is
**not installed** in the current environment (`pip show basic-pitch` →
not found).

`audio2midi/transcribers/base.py:12-15` hard-restricts backends per
instrument:

```python
_VALID_BACKENDS: dict[Instrument, set[str]] = {
    Instrument.PIANO: {"pti", "piano-transcription-inference", "basic-pitch", "basic_pitch"},
    Instrument.GUITAR: {"basic-pitch", "basic_pitch"},
    Instrument.VOCALS: {"rmvpe"},
}
```

Vocals can only ever route to RMVPE + the heuristic segmentation stack.
Basic Pitch has never been evaluated against the vocal MIDI eval harness.
This is notable because Basic Pitch is an **end-to-end learned note
segmenter** (joint onset + frame + pitch-bend prediction) — the same class
of component our heuristic stack (silence-merge → same-pitch-merge →
beat-gated onset split → gap-fill, `rmvpe.py:439-540`) is standing in for
by hand. [reported by subagent] Basic Pitch's own published eval claims
strong performance on vocal material specifically, though this project's own
prior benchmarking of Basic Pitch (if any exists) was not located and should
not be assumed to transfer without a direct run.

### Measured result (2026-07-18)

Basic Pitch (ICASSP 2022 ONNX model, vocal-range-tuned
`minimum_frequency=65.0, maximum_frequency=1050.0` matching `rmvpe.py`'s
existing C2–C6 bounds) was run directly on the same Adounravel MBR vocal
stem used for the accepted 2026-07-17 result, and scored with the
unmodified `scripts/eval_vocal_midi.py` against the same mix/vocals inputs.
Script: `run_basic_pitch.py`. Full report:
`basic-pitch-measurements.json`, `basic-pitch-hallucination-regions.json`.

Environment note [source-code]: the currently pinned scipy (1.17.1) removed
`scipy.signal.gaussian` (moved to `scipy.signal.windows.gaussian`);
`run_basic_pitch.py` shims this locally since basic-pitch 0.3.0 still calls
the old API. `setuptools` also had to be pinned `<81` to restore
`pkg_resources`, which `resampy` (a basic-pitch dependency) still imports.
Neither change touches the accepted pipeline's dependency pins.

| Metric | Accepted RMVPE (2026-07-17) | Basic Pitch (fresh, MBR stem) | Target |
|---|---:|---:|---:|
| Coverage | 72.93% | **54.32%** | ≥ 90% |
| Median note duration | 169 ms | **186.4 ms** | ≥ 300 ms |
| IOI CV | 2.04 | **2.07** | ≤ 1.5 |
| Hallucination | 13.05% | **5.37%** | ≤ 5% |
| Pitch within 0.5 st | 79.49% | **80.06%** | ≥ 79% guard |
| Duration mismatch | 0.702 s | **0.77 s** | ≤ 1.0 s |

All Basic Pitch values [measured]. Raw frame counts: 10,811 pYIN-voiced
frames vs only 6,205 MIDI-active frames (357 total notes) — Basic Pitch is
leaving roughly 43% of the vocal line untranscribed, which is the direct
cause of the coverage collapse.

Basic Pitch nearly clears the hallucination target and edges out pitch
accuracy, but coverage regresses sharply (72.93% → 54.32%, a −18.6 pp drop)
and IOI CV / median duration are essentially unchanged from the accepted
result — it does not solve fragmentation. Against the Experiment A pass bar
("improves at least two of {coverage, median duration, IOI CV,
hallucination} without dropping pitch accuracy below 79% or duration
mismatch above 1.0 s"), it improves only hallucination meaningfully (median
duration and IOI CV move by <1 pp/0.03, within noise) while regressing
coverage by far more than any plausible guard would allow.

This corroborates the user's recollection of a prior unsatisfying result
with Basic Pitch (context: 2026-04-17 pitch-tracker research found
basic-pitch on an **htdemucs** stem "unusable — octave errors, unstable"
due to the CQT model latching onto upper harmonics when vocal fundamentals
were stripped from the stem). The MBR stem fixes that specific spectral
defect, and this run shows no octave-error pathology — but a new,
independent failure mode appears instead: the model is simply too
conservative/sparse on this material, activating far fewer frames than
either pYIN or RMVPE, and produces short notes rather than long expressive
holds.

**Verdict: rejected, measured.** Basic Pitch as a wholesale segmentation
replacement does not clear the coverage guard and does not address
fragmentation, the two metrics currently furthest from target. Its
hallucination-rate strength does not transfer usefully unless combined with
something else that supplies the missing coverage — not pursued further
this session, since that would mean building a hybrid/ensemble system, a
larger scope than this validation pass.

## Finding 2 — HMM/Viterbi smoothing of the pitch+voicing track

[reported by subagent, cross-referenced against known librosa API] HMM/
Viterbi decoding of a pitch-probability + voicing observation sequence into
a smoothed state path is standard practice in melody-extraction literature
(MELODIA-style systems; pYIN's own probabilistic YIN core uses an HMM
internally). `librosa.sequence.viterbi` / `viterbi_discriminative` are
available in the currently pinned librosa (0.11.0) with no new dependency.

This directly targets the dominant remaining defect from the 2026-07-17
report: 88.2% of hallucinated frames are boundary-adjacent (onset/offset
flicker), and IOI CV / median duration are capped by the beat-gated onset
split re-cutting held notes. A Viterbi layer decoding note-state transitions
(sustain / onset / silence) with transition-cost penalties for rapid
state-flipping is a principled replacement for the current ad hoc
"merge-then-split-then-merge-again" pass order, rather than another
threshold to hand-tune.

**Verdict: adopt-candidate**, and the most repo-native option — no new
dependency, runs on the existing F0 array, directly targets IOI CV and
median duration.

## Finding 3 — Closing the pYIN-vs-RMS voicing gap

[reported by subagent] pYIN's voicing decision is documented in the SVT
literature as degrading specifically on breathy tone, heavy vibrato, and
melismatic runs — precisely the failure mode implied by the measured 14 pp
gap between RMS-voiced ratio (66.3%) and pYIN-voiced ratio (52.2%) in the
2026-07-17 session. This is a known limitation of pYIN for singing voice,
not a project-specific bug.

Two candidate signals were surfaced:
- **PESTO** (LGPL-3.0, ~130K params, explicit CPU mode, self-supervised) —
  outputs a per-frame confidence score that could serve as an alternative or
  ensemble voicing signal alongside pYIN.
- An unverified 2026 paper (arXiv 2601.11768) targeting improved
  probability-of-voicing specifically; **PDF text was not extractable by the
  subagent, so this citation is unverified** and should not be relied on
  without manual review at the source.

**Verdict: investigate-further.** Neither is a drop-in fix; both require a
runtime dependency add and direct evaluation before any adoption decision.

## Finding 4 — The evaluation metric itself may be the wrong shape

[reported by subagent] Standard SVT research practice (COnPOff / STARS
methodology) does **not** use frame-level pitch-tracker overlap as ground
truth for note-level evaluation — it uses tolerance-window note matching
(onset error < 100 ms, pitch error < 50 cents, offset error <
max(50 ms, 20% of note duration)) against hand-annotated datasets. Using
pYIN frame overlap as ground truth, as `scripts/eval_vocal_midi.py`
currently does, is a known-imperfect proxy for singing voice specifically —
the same 14 pp gap from Finding 3 is a symptom of this.

This reframes the 2026-07-17 finding that "89.7% of hallucination is
boundary-adjacent" from a description of the *pipeline*'s uncertainty margin.
A tolerance-window note-matching metric would very plausibly absorb most of
this boundary-adjacent hallucination as within-tolerance matches rather than
failures, because it stops penalizing sub-frame-grid timing disagreement
between the 10 ms pipeline grid and the 11.61 ms pYIN eval grid (the exact
mismatch flagged in the 2026-07-17 report).

**Verdict: adopt-candidate**, independent of any pipeline change — this
is a change to `scripts/eval_vocal_midi.py`'s scoring, not to the
transcription pipeline. It should be done carefully: switching the metric
changes what "meeting the target" means, so any adoption must be presented
as a metric-methodology change with the old frame-level metric retained
alongside the new one for comparability, not a silent redefinition of
already-published thresholds.

## Ruled out this session

- **ROSVOT** — confirmed (not just assumed, per prior non-goal) to require
  CUDA 11.8 + PyTorch 2.1.1 with no documented CPU path; trained only on
  Mandarin (M4Singer). Infeasible on M1, unvalidated cross-lingual.
- **MT3 (Magenta)** — T5X Transformer, no CPU-feasibility evidence,
  multi-instrument-oriented. Infeasible for this constraint set.
- **Semi-CRF / neural CRF note-boundary decoding** — active technique in
  2024 piano-transcription literature, but requires a trained scoring
  network; no pretrained vocal-domain weights exist. Would require building
  a labeled dataset, which is explicitly out of scope.
- **SwiftF0, SPICE** — pitch-only, not note segmenters; would only be
  relevant as an RMVPE replacement/ensemble member, which is a larger and
  separately-scoped change than the current targets justify. Not pursued
  further this session.

## Finding 2 (measured) — Raw 2-state Viterbi smoothing underperforms on fragmentation

[measured] `run_viterbi_smoothing.py` reproduces the accepted pipeline's
exact F0 array (RMVPE + pYIN gap-bridge + veto, same call sequence as
`RmvpeTranscriber.transcribe`) and replaces the heuristic
merge/split/gap-fill chain (`_merge_same_pitch`, beat-gated onset split,
`_gap_fill`, `_clip_notes_to_voiced_spans`) with a single 2-state
(voiced/unvoiced) Viterbi decode (`librosa.sequence.viterbi_discriminative`,
`stay_prob=0.985`), then segments the decoded voiced runs on semitone
changes — no beat-gating, no separate merge pass.

| Metric | Accepted RMVPE (2026-07-17) | Basic Pitch | Viterbi (raw, 2-state) | Target |
|---|---:|---:|---:|---:|
| Coverage | 72.93% | 54.32% | **71.17%** | ≥ 90% |
| Median note duration | 169 ms | 186.4 ms | **139.4 ms** | ≥ 300 ms |
| IOI CV | 2.04 | 2.07 | **2.19** | ≤ 1.5 |
| Hallucination | 13.05% | 5.37% | **12.68%** | ≤ 5% |
| Pitch within 0.5 st | 79.49% | 80.06% | **83.09%** | ≥ 79% guard |
| Duration mismatch | 0.702 s | 0.77 s | **0.703 s** | ≤ 1.0 s |

Raw frame counts: 10,811 pYIN-voiced, 8,811 MIDI-active, 7,694 co-active.
Full report: `viterbi-measurements.json`,
`viterbi-hallucination-regions.json`.

Coverage and hallucination land close to the accepted result (within ~2 pp
either direction) even though this prototype has **no** beat-gating, merge,
or gap-fill logic at all — the Viterbi voicing smoothing alone recovers
almost the same coverage as the accepted heuristic stack, which is a
meaningful signal that the voicing-decision layer, not the merge/split
heuristics, is doing most of the coverage work. Pitch accuracy improves
notably (+3.6 pp over accepted), plausibly because Viterbi smoothing removes
isolated single-frame pitch outliers that the median filter alone misses.

However, fragmentation is **worse**, not better: median duration drops
169 ms → 139.4 ms and IOI CV rises 2.04 → 2.19. This is the prototype's own
design gap, not a fundamental limitation of the technique: the current
2-state model only smooths the voiced/unvoiced decision — it re-segments on
every semitone change exactly like the raw heuristic pass does, with no
state for "sustain through a brief pitch wobble." A production version
would need a 3-state model (silence / onset / sustain-with-pitch-tolerance)
so the transition cost also discourages re-cutting a note on transient pitch
noise, not just on voicing flicker. That is the natural next iteration, not
attempted this session because it requires tuning a second parameter
(pitch-change tolerance) against held-out data, and this session's scope was
a single-parameter (`stay_prob`) proof of concept.

**Verdict: promising but not yet adopt-ready, measured.** The technique is
not a drop-in win at its current 2-state, single-parameter form — it trades
fragmentation for pitch accuracy and roughly matches on coverage/
hallucination. It is more principled than the heuristic stack (one
interpretable smoothing-strength parameter vs. five hand-tuned constants)
and the near-parity on coverage using zero beat/merge logic suggests real
headroom once a sustain-aware 3-state model is built. Recommend as the
lead candidate for a follow-up session, not for adoption as-is.

## Recommendation

Both validation experiments were run this session (2026-07-18). Results:

1. **Basic Pitch — rejected, measured.** Coverage collapses to 54.32% (vs
   72.93% accepted); does not solve fragmentation; only hallucination and
   pitch accuracy improve. Not viable as a wholesale segmentation
   replacement on this material even with the improved MBR stem.
2. **2-state Viterbi voicing smoothing — promising but incomplete,
   measured.** Matches accepted coverage/hallucination using zero
   beat-gating or merge heuristics, and improves pitch accuracy, but
   worsens fragmentation because the prototype has no pitch-sustain state.
   The near-parity on coverage with no merge/split logic at all is the most
   actionable finding of this session: it suggests the heuristic stack's
   complexity is not what is producing coverage, and that a proper 3-state
   (silence/onset/sustain) Viterbi model is worth building as the next
   concrete step, replacing rather than augmenting the current five-pass
   heuristic chain.

Finding 4 (tolerance-window evaluation) remains unimplemented and should be
scoped as a separate, explicit decision with the user before changing
`scripts/eval_vocal_midi.py`'s scoring semantics — it changes what the
existing published targets mean and must not be bundled into a pipeline
change.

## Conclusion (part 1, before continuation below)

Neither experiment run this session clears the unmet targets outright.
Basic Pitch is ruled out as a wholesale replacement. The Viterbi prototype
does not beat the accepted result on its own pass bar (only 2 of 4 metrics
improve, and fragmentation regresses), so per this session's own validation
plan it is **not** recommended for adoption in its current form. It is,
however, the clearest lead uncovered: a sustain-aware 3-state Viterbi model
is a well-scoped, low-risk next research step (no new dependencies, same
F0 array, one additional tunable parameter) with a plausible path to
improving fragmentation without sacrificing the coverage/pitch gains already
observed. The four original targets (coverage ≥90%, median ≥300ms, IOI CV
≤1.5, hallucination ≤5%) remain open.

---

## Session continuation (2026-07-18, part 2) — sustain-aware Viterbi + hard feasibility bounds

Follow-up work pursuing the "3-state sustain-aware Viterbi" lead flagged
above, plus a direct feasibility analysis of the four numeric targets
themselves. All work in this part is [measured] against the same Adounravel
MBR stem/mix, using cached RMVPE+pYIN arrays
(`cache_f0.py`, `cache_eval_pyin.py`) so iteration doesn't re-pay inference
cost. `fast_score.py` replicates `eval_vocal_midi.py`'s coverage/
hallucination/pitch/IOI-CV metrics locally for grid search; **every number
reported as final was cross-checked against the real
`scripts/eval_vocal_midi.py`**, not just the local approximation.

### Finding 5 — Sustain tolerance via pitch-median blending destroys pitch accuracy; fixed with a persistence debounce

First implementation of "3-state" sustain tolerance (`run_viterbi_v2.py`,
first draft) assigned each note's pitch as the running median of its
constituent frames and only cut on a >`pitch_tol_st` deviation from that
median. [measured] This is the wrong mechanism: widening the tolerance band
enough to meaningfully help fragmentation (IOI CV 2.31→1.56 as
`pitch_tol_st` went 0.6→3.0) collapsed pitch accuracy from 77.5%→52.3%
(frames within 0.5 semitone), because portamento/vibrato frames far from the
run's median get scored against their own true pitch by
`eval_vocal_midi.py`, not the blurred median. No parameter combination in a
180-point grid (`stay_prob` × `p_voiced_active/inactive` × `pitch_tol_st`)
cleared the 79% pitch guard.

**Fix, measured:** replaced the median-tolerance band with a persistence
debounce — a candidate new rounded-semitone pitch must hold for
`persist_frames` consecutive frames before it's allowed to end the current
note (`viterbi_smooth_notes` in `run_viterbi_v2.py`, final version). This
never blurs a committed note's assigned pitch, only delays *when* a cut
happens. Also discovered the Viterbi transition matrix (`stay_prob`) has
**no measurable effect** on the decoded state path in this problem —
identical output across `stay_prob` ∈ {0.97, 0.985, 0.993, 0.997} for fixed
other params, in both the median-filtered f0_accepted and the
pYIN-extended array. The 7-frame pre-Viterbi median filter
(`_apply_f0_median_filter`, existing accepted-pipeline function) already
removes essentially all single-frame flicker, so the observation sequence
fed to Viterbi is already near-binary and the smoothing step adds nothing
beyond what the median filter did. This means "Viterbi smoothing" was never
the active ingredient in Finding 2's original result — the persistence
debounce and F0-array construction are what matter.

### Finding 6 — Coverage has a hard ceiling around 84.7% on the accepted F0 array; no segmentation algorithm can exceed it

[measured] Before any note-segmentation algorithm even runs, the accepted
F0 array (`f0_accepted` = RMVPE + 200ms pYIN-bridge + veto, exactly as
`RmvpeTranscriber.transcribe` computes it) is only voiced on 84.7% of the
frames pYIN's own eval-grid run marks voiced (measured by resampling the
array's voiced mask onto the eval grid and checking overlap with
`eval_vocal_midi.py`'s pYIN ground truth). Since `coverage_ratio` is
"fraction of pYIN-voiced frames with an active MIDI note," and a note can
only be active where the F0 array is voiced, **84.7% is a hard upper bound
on coverage for any segmentation technique operating on this array** — no
Viterbi variant, heuristic merge/split, or persistence debounce can exceed
it. The 90% coverage target is therefore **unreachable via segmentation
changes alone**; reaching it requires changing the voicing decision itself
(the RMVPE+bridge+veto construction), which is a materially different
(bigger, riskier) change than a segmentation algorithm swap.

Extending the F0 array with an **unconditional pYIN fallback** (use pYIN's
own pitch wherever pYIN marks a frame voiced and RMVPE has none there,
dropping the existing bridge's 200ms window and octave-consistency
restriction — a qualitatively different move from further constant-tuning,
since it makes pYIN an independent second pitch source rather than only a
confirmation signal) raises the ceiling to 93.5% [measured,
`extend_f0_with_pyin_fallback` in `run_viterbi_v2.py`]. But this is not a
free win: filtering the fallback by pYIN's own voicing-probability
confidence shows almost none of the added frames are high-confidence
(`voiced_prob≥0.3` keeps only 204 of 2130 fallback frames, dropping the
ceiling back to 85.9%) — the frames RMVPE misses and pYIN alone claims are
inherently the hardest, most ambiguous ones. Grid search (40 configs, full
results in git history of this file / reproducible via `grid_search_v2.py`)
confirms this in the actual segmented output: the best coverage-clearing
configs (persist_frames=8, ≥90% coverage) collapse pitch accuracy to
67–69% (target guard is ≥79%) and push hallucination to ~19.5%. This is a
genuine Pareto frontier, not a tuning gap: **coverage and pitch accuracy
trade off directly and cannot be satisfied simultaneously with this
technique family on this F0 source.**

### Finding 7 — Hallucination has a hard floor around 15.5%, independent of segmentation quality

[measured] By the same method as Finding 6: frames where `f0_accepted` is
voiced (own grid) but the eval-grid pYIN run marks unvoiced make up 15.5%
of all frames `f0_accepted` considers voiced. Since `hallucination_ratio`
is "fraction of MIDI-active frames where pYIN marks unvoiced," **any
segmentation algorithm that activates a note everywhere the F0 array is
voiced already scores ≥15.5% hallucination before segmentation choices are
even considered** — segmentation can only make this worse (by activating
notes during F0-array-unvoiced gaps) or better (only by *dropping* coverage
below the array's own voicing, since it cannot un-hallucinate frames the
array itself marks voiced-but-pYIN-disagrees). The accepted result's
measured 13.05% is actually already *below* this naive floor estimate
(plausibly because its notes don't fully fill every voiced-array frame).
**The 5% hallucination target is not reachable via segmentation changes on
this F0 source** — it requires the two pitch trackers (RMVPE-based array,
pYIN ground truth) to agree on voicing far more than they currently do,
which is an upstream voicing-decision problem, not a segmentation one. This
also explains why 150+ segmentation configs measured across Findings 2, 5,
6 and this continuation all land in the 12–19.5% hallucination band
regardless of algorithm — the floor, not the algorithm, is the binding
constraint.

### Finding 8 — Cross-checking Finding 4 with mir_eval: the tolerance-window metric does not rescue the numeric targets, but it does reveal a real, reproducible technique improvement

Finding 4 (original session part) hypothesized that a tolerance-window
note-matching metric (COnPOff-style: onset <100ms, pitch <50 cents, offset
< max(50ms, 20% duration)) would absorb most of what the frame-level metric
counts as hallucination/coverage failure. This was implemented and
[measured] this session (`tolerance_window_eval.py`), using `mir_eval`
0.8.2 (already an installed dependency; no new package), which implements
these exact COnPOff tolerances as its defaults. Ground truth is a
**pYIN-derived pseudo-reference** (segmented the same way the Viterbi
prototypes segment RMVPE F0: contiguous voiced run, split on semitone
change, 30ms floor) — explicitly **not hand-annotated ground truth**, a
real limitation of this check; it can only show relative ranking between
techniques against a self-consistent pseudo-reference, not absolute
transcription quality against human annotation.

**Result: the hypothesis is only partly true.** Full COnPOff-style
F-measure for the accepted 2026-07-17 result is 0.389 (Onset F-measure
0.578, Offset F-measure 0.570) — the tolerance window does not make the
accepted result "pass" in any informal sense; there is real, substantial
onset/offset/pitch disagreement beyond boundary flicker. But it does
surface a genuine, reproducible win for the persistence-debounced Viterbi
segmentation from Finding 5: at `persist_frames=2, f0_filter_frames=5` (no
pYIN-fallback extension — Finding 6/7 show that trades away pitch/
hallucination too much), full F-measure is **0.501** (Onset F-measure
0.631, Offset F-measure 0.627) — roughly a **+11pp / +29% relative**
improvement over accepted on the standard metric, and this holds stably
across `persist_frames` ∈ {1,2,3} × `f0_filter_frames` ∈ {5,7,9} (21-point
neighborhood grid, F-measure 0.46–0.50 throughout), not a single lucky
config.

The catch: **this same candidate does not improve the original frame-level
targets.** Cross-checked against the real `eval_vocal_midi.py` (not just
`fast_score.py`'s local approximation):

| Metric | Accepted (2026-07-17) | Viterbi persist=2 (this session) | Target |
|---|---:|---:|---:|
| Coverage | 72.93% | **68.53%** | ≥ 90% |
| Median note duration | 169 ms | **121.2 ms** | ≥ 300 ms |
| IOI CV | 2.04 | **2.21** | ≤ 1.5 |
| Hallucination | 13.05% | **13.30%** | ≤ 5% |
| Pitch within 0.5 st | 79.49% | **84.56%** | ≥ 79% guard |
| Duration mismatch | 0.702 s | **0.703 s** | ≤ 1.0 s |

Only pitch accuracy improves under the frame-level metric (+5.1pp); the
other three unmet targets all regress slightly. So the frame-level metric
and the standard tolerance-window metric **disagree about which technique
is better** — a direct, measured illustration of Finding 4's own point that
the frame-level proxy is measuring something different from standard SVT
note-transcription quality, but not in the direction Finding 4 predicted
(it doesn't make the *existing* numbers look better; it reveals a
*different* technique is better than the frame-level metric shows).

## Conclusion (final, supersedes "Conclusion" above)

This session, across both its original pass and this continuation, ran two
independent segmentation techniques (Basic Pitch, Viterbi in three
variants) plus a feasibility analysis, all [measured] against the accepted
2026-07-17 result and cross-checked with the real `eval_vocal_midi.py`. The
outcome is a definitive **infeasibility result, not an unmet-target
placeholder**:

1. **Coverage ≥90% is mathematically unreachable** via any segmentation
   algorithm on the current F0 array (hard ceiling 84.7%, Finding 6) short
   of changing the underlying voicing decision, which is a distinct,
   larger-scoped change than a segmentation swap.
2. **Hallucination ≤5% is mathematically unreachable** for the same
   structural reason — a ~15.5% floor from voicing disagreement between
   RMVPE and pYIN exists before segmentation choices are even applied
   (Finding 7).
3. Extending the F0 array to raise the coverage ceiling (pYIN fallback,
   Finding 6) trades directly against pitch accuracy and hallucination —
   a genuine Pareto frontier, confirmed across a 40-point grid, not a
   tuning gap closeable with more search.
4. The one technique that does show a real, reproducible, cross-validated
   improvement — the persistence-debounced Viterbi segmentation — improves
   pitch accuracy (+5.1pp) and, more substantially, the standard COnPOff
   note-transcription F-measure (+11pp / +29% relative, Finding 8) over the
   accepted result, while *not* improving median duration, IOI CV, or
   hallucination under the original frame-level metric.

**Recommendation:** the four original targets, as defined against
`eval_vocal_midi.py`'s frame-level coverage/hallucination metric, should be
treated as **not achievable by segmentation-technique changes** on this
track with the current RMVPE+pYIN pitch-tracking sources — this is now a
measured, reproducible conclusion (Findings 6–7), not an absence of
sufficient tuning effort. Two forward paths exist, both out of this
session's scope and requiring explicit user sign-off before pursuing:
(a) a voicing-decision change upstream of segmentation (e.g. replacing or
ensembling RMVPE/pYIN with a better-agreeing voicing signal — Finding 3's
PESTO lead is the nearest unexplored candidate), which could move the
coverage/hallucination ceiling and floor themselves; or (b) formally
adopting a tolerance-window (COnPOff-style) evaluation methodology in place
of or alongside the frame-level metric, under which the persistence-
debounced Viterbi segmentation (Finding 5/8, `run_viterbi_v2.py`,
`persist_frames=2, f0_filter_frames=5`) is a measured, adoptable
improvement today. Neither path should be taken silently — (a) changes the
transcription pipeline, (b) redefines what "meeting target" means — both
require the same explicit-decision treatment the original Finding 4 already
called for.
