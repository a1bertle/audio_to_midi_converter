# /evaluate

Read `claude-agents/claude-agents/agents/evaluator.md` and `claude-agents/claude-agents/base/global-standards.md`, then run a full evaluation session.

Input: $ARGUMENTS

## Process
Follow the evaluator agent responsibilities in order:

0. Validate that the measurement method is known and standard for this input
   type. If uncertain, invoke the researcher agent before proceeding.
1. Characterize the input using approved measurement tools.
2. Write `evaluate.py` (or `evaluate.sh`) in the session directory before
   running any measurements. All values in `assessment.md` must come from
   this script's output.
3. Run the script against the input and capture output.
4. Identify defects and characteristics, each with severity and a measured value.
5. Derive a 1–3 sentence problem statement in measurable terms.
6. Populate `assessment.md` from the script output.

## Output
Produce in `research/<date>_<topic>-eval/`:
- `evaluate.py` — the evaluation script
- `requirements.txt` — if packages beyond the project venv baseline are needed
- `assessment.md` — input properties, measurements, findings table, problem
  statement, suggested success criteria, and recommended next step

## Guardrails
- Never modify the input file. All analysis is read-only.
- Every finding must cite a measured value with a provenance tag
  (`[measured]`, `[back-calc]`, `[source-code]`, or `[assumed]`).
- Do not propose solutions — the problem statement describes the defect,
  not the fix.
- Run all tools inside the project venv. Do not install global packages.
