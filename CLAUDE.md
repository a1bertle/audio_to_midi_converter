# Project Instructions

## Agent framework

Agent instructions live in `claude-agents/claude-agents/`. Read the relevant
file(s) before starting the corresponding workflow. For the full framework
overview and instructions on adding new agents, see `CLAUDE.base.md` in that
directory.

### Base workflows (read these first when applicable)
- `base/git-workflow.md` — commit policy, message format. Read before any commit.
- `base/research-workflow.md` — session directories, required files, sequence.
  Read before any research or benchmarking task.

### Session directories
- `research/<date>_<topic>/` — research and benchmarking sessions (measurements, findings, data).
- `plans/<date>_<topic>/` — implementation plans with no benchmark data (pure planning output).

### Role agents (read the matching file when entering that role)
- `agents/planner.md` — scoping and sequencing.
- `agents/researcher.md` — benchmarking and technique evaluation.
- `agents/implementer.md` — code changes.
- `agents/reviewer.md` — QA and review.
- `agents/visualizer.md` — Jupyter Notebook explainers.
- `agents/validator.md` — test suites, evaluation scripts, golden baselines.
- `agents/benchmarker.md` — structured benchmark entries from script runs.
- `agents/project-planner.md` — research-backed proposal: orchestrates researcher, visualizer, and benchmarker to produce `proposal.md`. Use when the right approach is unknown.
- `agents/evaluator.md` — media/data assessment: analyzes video, audio, or text inputs for measurable defects and derives a problem statement. Use before project-planner when the problem itself is undefined.

## Agent Roles (Slash Commands)
Use these commands to activate role-specific behavior:
- `/plan` - Planning and task breakdown
- `/research` - Benchmarking and method evaluation
- `/implement` - Code changes with guardrails
- `/review` - QA and review pass
- `/visualize` - Jupyter Notebook algorithm and architecture explainers
- `/validate` - Test suites, evaluation scripts, golden baselines, and performance benchmarks
- `/evaluate` - Analyze a media or data input for measurable defects and derive a problem statement
- `/propose` - Research-backed proposal: orchestrates researcher, visualizer, and benchmarker before any implementation

## Repository boundaries

`claude-agents/` is a git submodule (a separate repo). Never write
super-project implementation files inside `claude-agents/`. Those files live
in the super-project root. The submodule contains only agent instructions,
workflow docs, and shared tooling.

# Project-specific instructions below...



## Instruction Sync
- Ensure `CLAUDE.md` and `AGENTS.md` are always aligned.
- Any change to one must be applied to the other in the same commit.

## Git Workflow
- Run `git pull --rebase` immediately before creating a commit.
- Resolve pull/rebase issues before committing.

## Python Style
- Follow PEP 8 styling for all Python code.

## Commit Messages
- Every commit message must include both sections:
- `Feature changes`
- `Bug Fixes`
- If a section has no entries, include `- None`.

## Commit Message Template
<type>: <short summary>

Feature changes:
- <item or None>

Bug Fixes:
- <item or None>
