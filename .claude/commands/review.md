# /review

Read `claude-agents/claude-agents/agents/reviewer.md` and `claude-agents/claude-agents/base/global-standards.md`, then review the following.

Target: $ARGUMENTS

## Process
Follow the reviewer agent checklist against the target (file, diff, PR, or
session output). Report findings grouped by severity.

## Output
A structured review report with:
- Blocking issues (must fix before merge/delivery)
- Non-blocking issues (should fix)
- Informational notes
- Explicit pass/fail verdict

## Guardrails
- Do not make code changes during review — report findings only.
- Reference specific file paths and line numbers for every finding.
