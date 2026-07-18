# Implemented validation plan

1. Correct both half-note calculations to two beats and cover them with unit tests.
2. Make the MBR dependency set reproducible and reuse an existing stem on reruns.
3. Extend the evaluator with JSON output, full-duration MIDI alignment, final voiced
   time, and classified hallucination regions.
4. Fix the ineffective pYIN veto, preserve short enclosed gaps, and clip final note
   events to voiced spans.
5. Add local-file CLI input to validate approved media already on disk.
6. Evaluate Adounravel, reject the 150 ms smoothing candidate, and validate the
   accepted 70 ms/200 ms bridge configuration on Foals and Blue Bird.
7. Run unit and full-suite regression checks. Do not update a golden baseline
   without explicit user confirmation.

Validation commands are captured in `validate.sh`. Measurements and interpretation
are in `notes.md` and the adjacent JSON reports.

