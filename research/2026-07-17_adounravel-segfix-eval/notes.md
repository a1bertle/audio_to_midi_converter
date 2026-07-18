# Vocal segmentation optimization results

**Date:** 2026-07-17  
**Primary input:** Adounravel — 歌いました (`eZuDklxsaRg`)  
**Accepted configuration:** 70 ms F0 median filter, 200 ms pitch-consistent pYIN
bridge, pYIN overhang veto, final voiced-span clipping

## Findings

The half-note constants were corrected from one beat to two beats. The pending
segmentation changes alone improved coverage but increased hallucination. Region
diagnostics showed that 88.2% of the hallucinated Adounravel frames were adjacent
to pYIN voicing, while only 2.0% were low-energy. This supported boundary trimming
instead of the proposed RMS gate. [measured]

The apparent 10.84-second truncation was an evaluation-definition error, not an
upstream truncation: the mix and MBR vocals stem are both 240.373 seconds, while
the final pYIN-voiced frame is at 230.272 seconds. The accepted MIDI ends at
229.570 seconds, a 0.702-second vocal-end mismatch. [measured]

## Adounravel before/after

| Metric | 2026-04-27 baseline | Step 2 | Accepted final | Target | Result |
|---|---:|---:|---:|---:|---|
| Coverage | 68.9% | 74.16% | 72.93% | ≥ 90% | Fail |
| Median note | 150 ms | 169 ms | 169 ms | ≥ 300 ms | Fail |
| IOI CV | 2.16 | 1.88 | 2.04 | ≤ 1.5 | Fail |
| Hallucination | 11.2% | 16.20% | 13.05% | ≤ 5% | Fail |
| Pitch within 0.5 st | 80.2% | 78.83% | 79.49% | ≥ 79% guard | Pass |
| Vocal-end mismatch | Not measured | 0.702 s | 0.702 s | ≤ 1.0 s | Pass |

All new values are [measured] by `scripts/eval_vocal_midi.py`. Baseline values are
[measured] in `research/2026-04-27_adounravel-mbr-vocal-midi-eval/assessment.md`.
Step 2 and accepted reports are `step2-measurements.json` and
`final-measurements.json`.

The accepted gate loses 1.22 percentage points of coverage versus Step 2, within
the plan's 2-point guard. The remaining hallucination is 89.7% boundary-adjacent;
the 10 ms pipeline grid and 11.61 ms evaluation grid make short-note edges dominate
this metric. [measured + inferred]

## Rejected candidate

A 15-frame (150 ms) median filter raised coverage to 75.73% and median duration to
181 ms, but reduced half-semitone accuracy to 76.34%, below the 79% guard. It was
rejected and the measured 7-frame default retained. [measured]

## Multi-track validation

| Track | BPM | Coverage | Hallucination | Pitch ±0.5 st | Median note | IOI CV | Vocal-end mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adounravel | 135.00 | 72.93% | 13.05% | 79.49% | 169 ms | 2.04 | 0.702 s |
| Foals — My Number | 128.24 | 76.68% | 13.06% | 65.69% | 155 ms | 1.37 | 0.268 s |
| Blue Bird | 152.03 | 79.27% | 3.96% | 90.92% | 170 ms | 1.41 | 4.830 s |

All values are [measured]. Foals has a drastic pitch-accuracy regression, while
Blue Bird passes the hallucination and pitch thresholds but retains an unexplained
late pYIN-voiced region. These differences rule out claiming that the remaining
targets generalize across tracks. No approved slow-ballad input was available, so
the two additional approved local tracks were used rather than adding external
benchmark media. [source inventory]

## Stem-separation contingency

The remaining non-boundary Adounravel hallucination is 10.3% of hallucinated
frames (3.0% low-energy and 7.3% unvoiced high-energy), so accompaniment bleed
does not dominate the measured regions. Step 6 was not triggered. [measured]

## Conclusion

The duration bug and half-note implementation bug are fixed, diagnostics are now
repeatable, and the accepted gate satisfies its coverage-loss and pitch guards.
The 90% coverage, 300 ms median, IOI ≤ 1.5, and 5% hallucination goals remain open
for Adounravel; further constant tuning is not justified by these results.

