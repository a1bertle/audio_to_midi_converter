# Project Instructions

# Project Standards

This file provides project instructions for OpenAI Codex and compatible tools.
It mirrors the rules in `CLAUDE.md` and the `claude-agents/` framework.

## Platform Context
- Target: Apple MacBook Air (M1, 4P+4E cores, 7-core GPU, 16 GB unified RAM)
- OS: macOS (darwin)
- Projects may override this profile; performance decisions follow the active target.

## Coding Standards
- Python: PEP-8, 4-space indent, max 99 char lines
- Respect architecture boundaries defined in project docs
- Keep shared modules free of UI-framework dependencies unless explicitly intended

## Error Logging
All code produced — application code, test harnesses, evaluation scripts, and
CLI tools — must include structured error logging to a hidden `.logs/`
directory at the repo root.

- Create `.logs/` on first run if it does not exist. Add `.logs/` to
  `.gitignore` so log files are never committed.
- Log all unhandled exceptions and critical errors to `.logs/error.log`
  using a rotating file handler (5 MB max, 3 backups).
- Every log entry must include: ISO 8601 timestamp with timezone, log level,
  module path, full traceback, process ID, thread name, and relevant context
  (CLI arguments, test name, request ID, etc.).
- Write errors to both the log file and stderr.
- For GUI applications, install a global exception hook that logs and surfaces
  an error dialog.
- Use a single shared logging setup module per project.

## Performance & Parallelism
- Set parallelism intentionally based on core count and workload type
- `ThreadPoolExecutor` / `threading.Thread` for I/O-bound and subprocess work
- `ProcessPoolExecutor` only for pure-Python CPU-bound workloads (GIL bypass)
- Split long workloads into parallel segments over monolithic subprocesses
- Avoid oversubscribing CPU threads across nested workers

### Accelerators
- Prefer native accelerator paths on the target platform
- Fall back to CPU when accelerator dependency is unavailable

### Memory
- Favor in-memory CPU<->accelerator handoff where available
- Avoid unnecessary disk round-trips for large frame buffers
- Define and enforce a memory budget for concurrent workloads with system headroom

## Python Environment
- Always create a virtual environment before installing dependencies or
  running code: `python3 -m venv .venv && source .venv/bin/activate`
- If a `.venv` already exists, activate it instead of creating a new one.
- Install dependencies into the venv (`pip install -r requirements.txt`),
  never into the system Python.

## Git Workflow
- Never commit unless the user explicitly requests it
- Never commit in the same response where code was written or edited
- Update `README.md` for any new feature before committing

### Commit Message Format
Use this structure (omit empty sections):

```text
<short summary line>

Features:
- <bullet per new feature>

Bug Fixes:
- <bullet per bug fix>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

Rules:
- Do not write prose paragraphs in the body.
- Use bullet points under section headers for all details.
- Inspect latest full message with `git log --format="%B" -1` before
  committing to match existing style.

## Documentation Discipline
- Update primary project documentation (usually `README.md`) for any new
  feature before committing.
- Keep documentation synchronized with implementation state at commit time.
- `CLAUDE.md` and `AGENTS.md` must convey the same project standards. When
  either file is updated, the other must be updated to match.

### README.md Standards
Every project must have a `README.md` at the repo root. Create it when
initializing a new project and keep it current as features are added.

Required sections (in order, omit if not applicable):
1. Title & Description — project name and 1-3 sentence summary.
2. Setup / Installation — prerequisites, dependencies, venv creation.
   Must be copy-pasteable.
3. Usage — how to run the project with CLI examples or screenshots.
4. Configuration — all user-facing settings, env vars, config files with
   defaults and allowed values.
5. Project Structure — directory tree with one-line descriptions.
6. Architecture (if non-trivial) — major components, data flow, diagrams.
7. Testing — how to run tests, where golden baselines live, how to add tests.
8. Contributing (if collaborative) — branch conventions, PR workflow.

Guidelines:
- Write for someone who has never seen the codebase.
- Use fenced code blocks for all commands and code snippets.
- Pin version numbers when version matters.
- Link to dedicated docs rather than embedding long explanations.
- Remove entries when features are removed — no stale documentation.

## Research Workflow
- For technique research and performance evaluation, use only approved existing
  inputs from the project-defined benchmark data directory.
- Do not introduce synthetic or external test inputs unless explicitly requested.
- Create a named, dated session folder under the project research root:
  `<research-root>/<date>_<topic>/`
- Each session should contain:
  - `notes.md` — raw findings, data tables, interpretation
  - `fix-plan.md` or `implementation-plan.md` — step-by-step plan with code
    snippets, file/line references, and validation steps
  - Validation scripts (e.g., `validate.sh`) — executable, repeatable checks
- Follow this sequence: research, document findings, plan, document plan,
  implement, validate, document results.

## Agent Roles
The `claude-agents/agents/` directory contains role-specific instructions.
Reference the appropriate file for the active task:

- `planner.md` — scoping, sequencing, and task breakdown
- `researcher.md` — benchmarking and method evaluation
- `implementer.md` — code changes with platform-aware guardrails
- `reviewer.md` — QA, review checklist, and output format
- `visualizer.md` — Jupyter Notebook algorithm and architecture explainers
- `validator.md` — test suites, evaluation scripts, golden baselines, and performance benchmarks


This file provides project instructions for OpenAI Codex and compatible tools.
It mirrors the rules in `CLAUDE.md` and the `claude-agents/` framework.

## Platform Context
- Target: Apple MacBook Air (M1, 4P+4E cores, 7-core GPU, 16 GB unified RAM)
- OS: macOS (darwin)
- Projects may override this profile; performance decisions follow the active target.

## Coding Standards
- Python: PEP-8, 4-space indent, max 99 char lines
- Respect architecture boundaries defined in project docs
- Keep shared modules free of UI-framework dependencies unless explicitly intended

## Error Logging
All code produced — application code, test harnesses, evaluation scripts, and
CLI tools — must include structured error logging to a hidden `.logs/`
directory at the repo root.

- Create `.logs/` on first run if it does not exist. Add `.logs/` to
  `.gitignore` so log files are never committed.
- Log all unhandled exceptions and critical errors to `.logs/error.log`
  using a rotating file handler (5 MB max, 3 backups).
- Every log entry must include: ISO 8601 timestamp with timezone, log level,
  module path, full traceback, process ID, thread name, and relevant context
  (CLI arguments, test name, request ID, etc.).
- Write errors to both the log file and stderr.
- For GUI applications, install a global exception hook that logs and surfaces
  an error dialog.
- Use a single shared logging setup module per project.

## Performance & Parallelism
- Set parallelism intentionally based on core count and workload type
- `ThreadPoolExecutor` / `threading.Thread` for I/O-bound and subprocess work
- `ProcessPoolExecutor` only for pure-Python CPU-bound workloads (GIL bypass)
- Split long workloads into parallel segments over monolithic subprocesses
- Avoid oversubscribing CPU threads across nested workers

### Accelerators
- Prefer native accelerator paths on the target platform
- Fall back to CPU when accelerator dependency is unavailable

### Memory
- Favor in-memory CPU<->accelerator handoff where available
- Avoid unnecessary disk round-trips for large frame buffers
- Define and enforce a memory budget for concurrent workloads with system headroom

## Python Environment
- Always create a virtual environment before installing dependencies or
  running code: `python3 -m venv .venv && source .venv/bin/activate`
- If a `.venv` already exists, activate it instead of creating a new one.
- Install dependencies into the venv (`pip install -r requirements.txt`),
  never into the system Python.

## Git Workflow
- Never commit unless the user explicitly requests it
- Never commit in the same response where code was written or edited
- Update `README.md` for any new feature before committing

### Commit Message Format
Use this structure (omit empty sections):

```text
<short summary line>

Features:
- <bullet per new feature>

Bug Fixes:
- <bullet per bug fix>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

Rules:
- Do not write prose paragraphs in the body.
- Use bullet points under section headers for all details.
- Inspect latest full message with `git log --format="%B" -1` before
  committing to match existing style.

## Documentation Discipline
- Update primary project documentation (usually `README.md`) for any new
  feature before committing.
- Keep documentation synchronized with implementation state at commit time.
- `CLAUDE.md` and `AGENTS.md` must convey the same project standards. When
  either file is updated, the other must be updated to match.

### README.md Standards
Every project must have a `README.md` at the repo root. Create it when
initializing a new project and keep it current as features are added.

Required sections (in order, omit if not applicable):
1. Title & Description — project name and 1-3 sentence summary.
2. Setup / Installation — prerequisites, dependencies, venv creation.
   Must be copy-pasteable.
3. Usage — how to run the project with CLI examples or screenshots.
4. Configuration — all user-facing settings, env vars, config files with
   defaults and allowed values.
5. Project Structure — directory tree with one-line descriptions.
6. Architecture (if non-trivial) — major components, data flow, diagrams.
7. Testing — how to run tests, where golden baselines live, how to add tests.
8. Contributing (if collaborative) — branch conventions, PR workflow.

Guidelines:
- Write for someone who has never seen the codebase.
- Use fenced code blocks for all commands and code snippets.
- Pin version numbers when version matters.
- Link to dedicated docs rather than embedding long explanations.
- Remove entries when features are removed — no stale documentation.

## Research Workflow
- For technique research and performance evaluation, use only approved existing
  inputs from the project-defined benchmark data directory.
- Do not introduce synthetic or external test inputs unless explicitly requested.
- Create a named, dated session folder under the project research root:
  `<research-root>/<date>_<topic>/`
- Each session should contain:
  - `notes.md` — raw findings, data tables, interpretation
  - `fix-plan.md` or `implementation-plan.md` — step-by-step plan with code
    snippets, file/line references, and validation steps
  - Validation scripts (e.g., `validate.sh`) — executable, repeatable checks
- Follow this sequence: research, document findings, plan, document plan,
  implement, validate, document results.

## Agent Roles
The `claude-agents/agents/` directory contains role-specific instructions.
Reference the appropriate file for the active task:

- `planner.md` — scoping, sequencing, and task breakdown
- `researcher.md` — benchmarking and method evaluation
- `implementer.md` — code changes with platform-aware guardrails
- `reviewer.md` — QA, review checklist, and output format
- `visualizer.md` — Jupyter Notebook algorithm and architecture explainers
- `validator.md` — test suites, evaluation scripts, golden baselines, and performance benchmarks


This file provides project instructions for OpenAI Codex and compatible tools.
It mirrors the rules in `CLAUDE.md` and the `claude-agents/` framework.

## Platform Context
- Target: Apple MacBook Air (M1, 4P+4E cores, 7-core GPU, 16 GB unified RAM)
- OS: macOS (darwin)
- Projects may override this profile; performance decisions follow the active target.

## Coding Standards
- Python: PEP-8, 4-space indent, max 99 char lines
- Respect architecture boundaries defined in project docs
- Keep shared modules free of UI-framework dependencies unless explicitly intended

## Error Logging
All code produced — application code, test harnesses, evaluation scripts, and
CLI tools — must include structured error logging to a hidden `.logs/`
directory at the repo root.

- Create `.logs/` on first run if it does not exist. Add `.logs/` to
  `.gitignore` so log files are never committed.
- Log all unhandled exceptions and critical errors to `.logs/error.log`
  using a rotating file handler (5 MB max, 3 backups).
- Every log entry must include: ISO 8601 timestamp with timezone, log level,
  module path, full traceback, process ID, thread name, and relevant context
  (CLI arguments, test name, request ID, etc.).
- Write errors to both the log file and stderr.
- For GUI applications, install a global exception hook that logs and surfaces
  an error dialog.
- Use a single shared logging setup module per project.

## Performance & Parallelism
- Set parallelism intentionally based on core count and workload type
- `ThreadPoolExecutor` / `threading.Thread` for I/O-bound and subprocess work
- `ProcessPoolExecutor` only for pure-Python CPU-bound workloads (GIL bypass)
- Split long workloads into parallel segments over monolithic subprocesses
- Avoid oversubscribing CPU threads across nested workers

### Accelerators
- Prefer native accelerator paths on the target platform
- Fall back to CPU when accelerator dependency is unavailable

### Memory
- Favor in-memory CPU<->accelerator handoff where available
- Avoid unnecessary disk round-trips for large frame buffers
- Define and enforce a memory budget for concurrent workloads with system headroom

## Python Environment
- Always create a virtual environment before installing dependencies or
  running code: `python3 -m venv .venv && source .venv/bin/activate`
- If a `.venv` already exists, activate it instead of creating a new one.
- Install dependencies into the venv (`pip install -r requirements.txt`),
  never into the system Python.

## Git Workflow
- Never commit unless the user explicitly requests it
- Never commit in the same response where code was written or edited
- Update `README.md` for any new feature before committing

### Commit Message Format
Use this structure (omit empty sections):

```text
<short summary line>

Features:
- <bullet per new feature>

Bug Fixes:
- <bullet per bug fix>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

Rules:
- Do not write prose paragraphs in the body.
- Use bullet points under section headers for all details.
- Inspect latest full message with `git log --format="%B" -1` before
  committing to match existing style.

## Documentation Discipline
- Update primary project documentation (usually `README.md`) for any new
  feature before committing.
- Keep documentation synchronized with implementation state at commit time.
- `CLAUDE.md` and `AGENTS.md` must convey the same project standards. When
  either file is updated, the other must be updated to match.

### README.md Standards
Every project must have a `README.md` at the repo root. Create it when
initializing a new project and keep it current as features are added.

Required sections (in order, omit if not applicable):
1. Title & Description — project name and 1-3 sentence summary.
2. Setup / Installation — prerequisites, dependencies, venv creation.
   Must be copy-pasteable.
3. Usage — how to run the project with CLI examples or screenshots.
4. Configuration — all user-facing settings, env vars, config files with
   defaults and allowed values.
5. Project Structure — directory tree with one-line descriptions.
6. Architecture (if non-trivial) — major components, data flow, diagrams.
7. Testing — how to run tests, where golden baselines live, how to add tests.
8. Contributing (if collaborative) — branch conventions, PR workflow.

Guidelines:
- Write for someone who has never seen the codebase.
- Use fenced code blocks for all commands and code snippets.
- Pin version numbers when version matters.
- Link to dedicated docs rather than embedding long explanations.
- Remove entries when features are removed — no stale documentation.

## Research Workflow
- For technique research and performance evaluation, use only approved existing
  inputs from the project-defined benchmark data directory.
- Do not introduce synthetic or external test inputs unless explicitly requested.
- Create a named, dated session folder under the project research root:
  `<research-root>/<date>_<topic>/`
- Each session should contain:
  - `notes.md` — raw findings, data tables, interpretation
  - `fix-plan.md` or `implementation-plan.md` — step-by-step plan with code
    snippets, file/line references, and validation steps
  - Validation scripts (e.g., `validate.sh`) — executable, repeatable checks
- Follow this sequence: research, document findings, plan, document plan,
  implement, validate, document results.

## Agent Roles
The `claude-agents/agents/` directory contains role-specific instructions.
Reference the appropriate file for the active task:

- `planner.md` — scoping, sequencing, and task breakdown
- `researcher.md` — benchmarking and method evaluation
- `implementer.md` — code changes with platform-aware guardrails
- `reviewer.md` — QA, review checklist, and output format
- `visualizer.md` — Jupyter Notebook algorithm and architecture explainers
- `validator.md` — test suites, evaluation scripts, golden baselines, and performance benchmarks

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
