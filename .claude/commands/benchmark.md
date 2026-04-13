# /benchmark

Read `claude-agents/claude-agents/agents/benchmarker.md`, then record the following script output as a structured benchmark entry.

Output to record: $ARGUMENTS

## Process
1. Resolve the target `benchmark_results.md` path:
   - Active research session today → `research/<session-dir>/benchmark_results.md`
   - No active session → `benchmark_runs/<date>_<short-topic>/benchmark_results.md`
2. Create the file with header if it does not exist.
3. Parse the output to extract all canonical fields.
4. Append a new run entry block.
5. Refresh the Cross-Run Comparison table.

## Guardrails
- Never delete or modify existing run entries — only append.
- Never write benchmark data into `notes.md` or any other pre-existing file.
- Never fabricate timestamps or frame counts; mark derived values `(computed)`.
- Tag derived metrics with `[back-calc]` and direct measurements with
  `[measured]` per `base/global-standards.md`.
- Record facts only — no performance conclusions in run entries.
