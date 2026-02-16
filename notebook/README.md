# Notebook Guide

Notebook path:
- `notebook/midi_conversion_algorithm.ipynb`

## Purpose
This notebook explains and visualizes the PTI transcription algorithm and maps PTI outputs to this application's event and MIDI processing.

## Scope
- Algorithm theory and visualizations.
- PTI decoding concepts.
- App-level postprocess effects.

Out of scope:
- YouTube ingestion workflow details.
- Packaging/deployment.

## Setup
```bash
scripts/venv_create.sh
source .venv/bin/activate
pip install -r requirements-notebook.txt
```

## Launch
```bash
jupyter lab
```
Open:
- `notebook/midi_conversion_algorithm.ipynb`

## Runtime modes
- Fast mode (default):
- Uses synthetic decoder demo and cached artifacts where available.
- Heavy mode (optional):
- Set `RUN_OPTIONAL_HEAVY_PTI = True` in the notebook config cell.
- Requires PTI dependencies and checkpoint availability.

## Notes
- The notebook intentionally separates PTI internal theory from this app's production pipeline behavior.
