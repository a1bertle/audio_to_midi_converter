# /validate

Read `claude-agents/claude-agents/agents/validator.md` and `claude-agents/claude-agents/base/global-standards.md`, then create or run validation for the following.

Target: $ARGUMENTS

## Process
Follow the validator agent responsibilities: write test suites, evaluation
scripts, and golden baselines that verify correctness of the target.

## Output
- Test scripts or test suite files covering the target
- Golden baseline files where applicable
- A validation report with pass/fail results

## Guardrails
- Use real fixtures and real subprocesses — no mocks.
- Tests validate behavior, not implementation internals.
- Synthetic inputs must be committed to the repo (fast, deterministic,
  no external dependencies).
- Quality metrics (PSNR/SSIM or domain equivalent) for output correctness
  on long-form inputs.
