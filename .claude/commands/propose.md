# /propose

Read `claude-agents/claude-agents/agents/project-planner.md` and `claude-agents/claude-agents/base/global-standards.md`, then run a full research-backed proposal session.

Problem: $ARGUMENTS

## Process
Follow the project-planner agent responsibilities in order:

1. Frame the problem: restate precisely, define success criteria, constraints,
   non-goals, and 2–4 candidate approaches.
2. Delegate to the **researcher** agent (per `base/research-workflow.md`) for
   each candidate that needs empirical evaluation. Invoke the **benchmarker**
   agent to record structured results.
3. Delegate to the **visualizer** agent for the top 1–2 candidates.
4. Synthesize `proposal.md` with: recommendation (up front), evidence table,
   tradeoff analysis, risks, visualization links, and a one-line handoff to
   the planner.

## Output
Produce in `plans/<date>_<topic>/`:
- `proposal.md` — the decision document

## Guardrails
- Every value in `proposal.md` must carry a provenance tag and cite its
  source research session. No invented or untraced numbers.
- Do not recommend without measured evidence — mark untestable assumptions
  `[assumed]` with rationale.
- Do not write implementation plans or code.
- Hand off to `/plan` once the approach is decided.
