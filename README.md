# audio_to_midi_converter

Python CLI for converting noisy YouTube piano performances to MIDI.

## What this does
- Downloads YouTube audio (`yt-dlp`)
- Converts media to WAV (`ffmpeg`)
- Preprocesses audio (normalize, high-pass, optional denoise)
- Transcribes with piano-focused backend (`piano-transcription-inference`) or fallback (`basic-pitch`)
- Writes output MIDI

## Requirements
- Python `3.10+`
- `ffmpeg` on `PATH`
- `yt-dlp` on `PATH` or in the Python environment
- Network access for YouTube download and first PTI checkpoint download
- Optional: `curl` for checkpoint download fallback

## Setup
1. Create venv
```bash
scripts/venv_create.sh
```

2. Install runtime dependencies
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Alternative without activating shell:
```bash
.venv/bin/pip install -r requirements.txt
```

## Run
```bash
scripts/run_audio2midi.sh \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output out.mid \
  --workdir .cache/audio2midi \
  --keep-intermediate
```

## CLI options
```bash
scripts/run_audio2midi.sh --help
```

Key options:
- `--backend {pti,piano-transcription-inference,basic-pitch}`
- `--device cpu|cuda`
- `--denoise`
- `--min-duration <seconds>`
- `--note-threshold <0..1>`
- `--quantize-grid-seconds <seconds>`
- `--pti-checkpoint-path /path/to/model.pth`

## PTI model details
- Default backend is `pti` (`piano-transcription-inference`)
- Default checkpoint filename:
`note_F1=0.9677_pedal_F1=0.9186.pth`
- Default checkpoint location:
`~/piano_transcription_inference_data/note_F1=0.9677_pedal_F1=0.9186.pth`
- If missing, checkpoint is downloaded automatically
- Download flow: Python HTTP first, then `curl` fallback

Overrides:
- Local checkpoint path:
```bash
scripts/run_audio2midi.sh \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output out.mid \
  --pti-checkpoint-path "/path/to/note_F1=0.9677_pedal_F1=0.9186.pth"
```
- Checkpoint URL (advanced):
```bash
export AUDIO2MIDI_PTI_CHECKPOINT_URL="https://example.com/model.pth"
```
- Checkpoint path via env:
```bash
export AUDIO2MIDI_PTI_CHECKPOINT_PATH="/path/to/model.pth"
```

## Troubleshooting
- `ffmpeg extraction failed`: verify `ffmpeg -version` works and input file is readable.
- `piano-transcription-inference failed`: rerun with `--pti-checkpoint-path` to a valid local `.pth`.
- First PTI run can take time on CPU and may be quiet while inference runs.
- If PTI dependency compatibility blocks execution, try:
```bash
scripts/run_audio2midi.sh \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output out.mid \
  --backend basic-pitch
```

## Dev/test
```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

## Notes
- Historical C++ planning docs are kept in `C_PLAN.md` and `C_RESEARCH.md`.
