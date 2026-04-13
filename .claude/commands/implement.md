# /implement

Read `claude-agents/claude-agents/agents/implementer.md` and `claude-agents/claude-agents/base/global-standards.md`, then implement the following.

Task: $ARGUMENTS

## Process
1. Confirm a plan exists before writing any code. If none exists, run `/plan`
   first.
2. Activate or create the project venv before installing dependencies.
3. Follow PEP-8 and project module boundaries.
4. Apply platform optimization guidance from `base/global-standards.md`
   (executor model, thread counts, accelerator paths).
5. Set up structured error logging per the Error Logging standard if not
   already present.
6. Update `README.md` for any new feature before the implementation is
   considered complete.
7. Keep `CLAUDE.md` and `AGENTS.md` in sync if project instructions change.

## Guardrails
- Do not commit unless explicitly asked.
- Do not commit in the same response where code was written or edited.
- Respect project architecture boundaries.
