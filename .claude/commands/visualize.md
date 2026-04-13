# /visualize

Read `claude-agents/claude-agents/agents/visualizer.md` and `claude-agents/claude-agents/base/global-standards.md`, then create a Jupyter Notebook visualizer for the following.

Topic: $ARGUMENTS

## Process
Follow the visualizer agent notebook structure:
1. Title & Overview cell (markdown)
2. Setup & Configuration cell (code) with `# --- Configuration ---` block
3. Algorithm sections — one per stage, each with: explainer (markdown),
   implementation from scratch (code), visualization (code)
4. Summary section — multi-panel figure and printed results

## Output
Produce in `notebooks/<topic>/`:
- `<topic>_visualizer.ipynb` — the notebook
- `requirements.txt` — minimal pinned dependencies
- `README.md` — setup instructions, section index, parameter provenance table

Create `notebooks/open_notebook.sh` if it does not already exist.

## Guardrails
- Tag every constant with a provenance tag inline (`[measured]`, `[back-calc]`,
  `[source-code]`, or `[assumed]`). See `base/global-standards.md`.
- Never use a library's built-in algorithm when the purpose is to teach it.
- Run the notebook end-to-end with `jupyter nbconvert --execute` before
  delivering. Fix all errors before delivery.
- Do not hardcode absolute paths — all user-specific paths go in the
  configuration cell.
- Do not include external hardware data unless explicitly requested.
