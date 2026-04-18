# Vocal F0 Pipeline Visualizer

Jupyter Notebook walkthrough of every stage in the `audio2midi` vocal transcription
pipeline — from raw RMVPE pitch extraction to final MIDI note list.

## Setup

Uses the repo-root `.venv` (which includes `torch`, `librosa`, `demucs`, etc.).
No separate venv needed.

```bash
# From repo root — register the venv as a Jupyter kernel (one-time)
.venv/bin/python -m ipykernel install --user \
    --name audio2midi-venv --display-name "audio2midi (.venv)"

# Open the notebook
bash notebooks/open_notebook.sh \
    notebooks/vocal-f0-pipeline-visualizer/vocal_f0_pipeline_visualizer.ipynb
```

Select kernel **"audio2midi (.venv)"** when the notebook opens.

## Required Input Files

Before running, ensure these exist (produced by the research session):

| File | How to produce |
|------|---------------|
| `research/2026-04-18_f0-median-filter-research/stems/htdemucs/mix/vocals.wav` | Run demucs on `mix.wav` in that session dir |
| `research/2026-04-17_vocal-pitch-tracker-research/rmvpe.pt` | Auto-downloaded on first `audio2midi` run |

## Section Index

| # | Section | What it shows |
|---|---------|---------------|
| 1 | Raw F0 Extraction | Continuous Hz contour from RMVPE at 10 ms/frame |
| 2 | F0 Median Filter | Per-segment vibrato suppression before semitone snapping |
| 3 | Semitone Snap + Min-Duration | Hz → MIDI integers; discard sub-32nd-note fragments |
| 4 | Silence-Aware Merge | Collapse A-B-A ornaments and near-unison stutters |
| 5 | Same-Pitch Merge | Merge sustained notes interrupted by brief silence |
| 6 | Beat-Gated Onset Split | Re-split long notes at beat-aligned syllable boundaries |
| 7 | Snap-to-Beats (optional) | Quantize note boundaries to nearest 16th-note grid |
| 8 | Summary | Multi-panel overview + metrics table |

## Parameter Provenance

| Parameter | Value | Provenance | Source |
|-----------|-------|------------|--------|
| `TARGET_SR` | 16000 Hz | source-code | `rmvpe.py:_TARGET_SR` |
| `HOP_SAMPLES` | 160 | source-code | `rmvpe.py:_HOP_S = 160/16000` |
| `BPM` | 152.05 | measured | bpm_detect on raw mix; `research/2026-04-18_f0-median-filter-research/notes.md` |
| `MIN_NOTE_S` | 49.3 ms | back-calc | 60/152.05/8 |
| `SIXTEENTH_S` | 98.6 ms | back-calc | 60/152.05/4 |
| `EIGHTH_S` | 197.3 ms | back-calc | 60/152.05/2 |
| `F0_FILTER_FRAMES` | 7 (70 ms) | measured | Vibrato period 194.5 ms on this track; window = 0.36× period. Nix et al. JVoice 30(6) 2016: 4.5–6.5 Hz typical |
| `ONSET_HOP` | 256 | source-code | `rmvpe.py:_ONSET_HOP` |
| `ONSET_DELTA` | 0.07 | source-code | `rmvpe.py:_ONSET_DELTA` |
| `BEAT_TOLERANCE_FRAC` | 0.25 | source-code | `rmvpe.py` beat-gated split |
| `VIZ_START` / `VIZ_END` | 4–20 s | assumed | Shows interesting melodic phrase |
