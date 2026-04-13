# /research

Read `claude-agents/claude-agents/agents/researcher.md` and `claude-agents/claude-agents/base/research-workflow.md`, then run a research session on the following topic.

Topic: $ARGUMENTS

## Process
Follow the required sequence from `base/research-workflow.md`:
1. Research — use only approved existing inputs from the project benchmark
   data directory.
2. Document findings in `notes.md` before acting on them.
3. Plan — write `implementation-plan.md` or `fix-plan.md`.
4. Implement (if in scope for this session).
5. Validate using the session's `validate.sh` (or equivalent).
6. Document results.

## Output
Produce in `research/<date>_<topic>/`:
- `notes.md` — raw findings, measurements, data tables, interpretation
- `implementation-plan.md` or `fix-plan.md` — step-by-step plan with
  file/line references and validation steps
- `validate.sh` (or equivalent) — executable, repeatable checks

## Guardrails
- Tag every recorded value with a provenance tag (`[measured]`, `[back-calc]`,
  `[source-code]`, or `[assumed]`) per `base/global-standards.md`.
- Report measurement method and baseline for every metric.
- Call out tradeoffs: speed, quality, memory, reliability.
- Do not introduce synthetic or external test inputs unless explicitly requested.
