# Validation plan: Basic Pitch and HMM/Viterbi candidates

Two independent, low-cost experiments to measure against the accepted
2026-07-17 RMVPE+heuristics result on the same track (Adounravel). Neither
requires modifying the accepted pipeline; both are comparison measurements
first. Do not adopt either without a measured result meeting or improving on
the 2026-07-17 numbers without regressing the passing guards (pitch ±0.5 st
≥ 79%, vocal-end mismatch ≤ 1.0 s).

## Experiment A — Basic Pitch on the existing MBR vocal stem

1. `pip install -e '.[basic-pitch]'` in the project venv.
2. Run `basic_pitch.inference.predict()` directly (via a throwaway script, or
   by temporarily adding `"basic-pitch"` to
   `_VALID_BACKENDS[Instrument.VOCALS]` in `audio2midi/transcribers/base.py`
   for local testing only — revert before commit if not adopted) on
   `outputs/2026-07-17_Adounravel_segfix/stems/mbr/eZuDklxsaRg_(vocals)_MelBandRoformerSYHFTV3Epsilon.wav`.
3. Write the resulting MIDI into a new eval-only path (do not overwrite the
   accepted `Adounravel_歌いました.mid`).
4. Score with `python scripts/eval_vocal_midi.py --workdir <path>` against
   the same mix/vocals stem used for the accepted result, so the comparison
   is apples-to-apples.
5. Record coverage, median note duration, IOI CV, hallucination, pitch
   accuracy, and duration mismatch in `notes.md` alongside the 2026-07-17
   table.
6. If Basic Pitch's default vocal-range/threshold kwargs underperform,
   one bounded follow-up: try the existing guitar-tuned kwarg pattern in
   `_INSTRUMENT_PREDICT_KWARGS` (frame/onset thresholds, minimum note
   length) adapted to vocal pitch range (~C2–C6, matching `rmvpe.py`'s
   existing bounds) — but only after the default-kwargs baseline is
   recorded, so tuning doesn't get conflated with the base technique's
   result.

**Pass bar for "adopt-candidate → proposal":** improves at least two of
{coverage, median duration, IOI CV, hallucination} versus the 2026-07-17
accepted result without dropping pitch accuracy below 79% or duration
mismatch above 1.0 s.

## Experiment B — HMM/Viterbi note-state smoothing

1. Prototype in a standalone script (not `rmvpe.py`) against the same
   RMVPE F0 array already computed for Adounravel (recompute once, cache
   locally in this research session's directory as a `.npy` if convenient
   for iteration — do not check large binary artifacts into git).
2. Define a small state space (e.g. voiced-sustain, voiced-onset, unvoiced)
   with transition costs penalizing rapid flips, decoded via
   `librosa.sequence.viterbi_discriminative` over per-frame voicing/pitch
   observation likelihoods derived from the existing F0 confidence.
3. Compare the resulting note segmentation (before any of the current
   merge/split heuristic passes, or as a replacement for a specific subset
   of them — decide based on where the Viterbi output disagrees with the
   current heuristic output) against the accepted result using the same
   eval harness.
4. Record results the same way as Experiment A.

**Pass bar:** same as Experiment A.

## Sequencing

Run A and B independently; they are not mutually exclusive candidates (a
combined RMVPE-F0 → Viterbi-smoothed → Basic-Pitch-cross-check pipeline is
plausible but out of scope until both are measured alone). Do not touch
`audio2midi/transcribers/rmvpe.py`'s accepted heuristics or
`base.py`'s `_VALID_BACKENDS` permanently until one candidate has a measured
result justifying a proposal.

## Explicitly deferred (not in this validation pass)

- Finding 4 (tolerance-window evaluation metric) — scope as its own decision
  with the user before changing `scripts/eval_vocal_midi.py`'s scoring
  semantics, since it redefines what the existing published targets mean.
- PESTO / SwiftF0 / SPICE voicing-signal experiments — investigate-further
  only; not validated enough yet to warrant a runtime dependency add.
- Any multi-track (Foals, Blue Bird) validation — only after a candidate
  beats Adounravel's accepted result; per the 2026-07-17 report, single-track
  tuning risk is real and should gate before generalizing.

## Validation commands

Captured in `validate.sh`.

## Status (2026-07-18, final)

Both experiments were implemented, measured, and validated against the real
`scripts/eval_vocal_midi.py` (see notes.md "Session continuation" section).
Outcome: **infeasibility result**, not an unmet-target placeholder. Coverage
≥90% and hallucination ≤5% are proven unreachable via segmentation-only
changes on the current F0 sources (hard ceiling/floor, measured — notes.md
Findings 6-7). A follow-up technique (persistence-debounced Viterbi,
`run_viterbi_v2.py`) shows a real, cross-validated improvement under the
standard COnPOff tolerance-window metric (+11pp F-measure, notes.md
Finding 8) but does not clear the original frame-level targets. Two
follow-up paths are identified and require explicit user sign-off before
implementation (see notes.md Conclusion): an upstream voicing-decision
change, or formal adoption of the tolerance-window evaluation methodology.
Neither is implemented in this session.
