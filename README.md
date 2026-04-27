# audio_to_midi_converter

Python CLI for transcribing vocal performances from YouTube into MIDI. Uses RMVPE pitch tracking with beat-aware note snapping by default. Also supports piano and guitar.

## What this does
- Downloads YouTube audio (`yt-dlp`)
- Converts media to WAV (`ffmpeg`)
- Preprocesses audio (normalize, high-pass, optional denoise)
- Transcribes with piano-focused backend (`piano-transcription-inference`) or fallback (`basic-pitch`)
- Transcribes vocals using RMVPE pitch tracker with beat-aware note snapping
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

Outputs (MIDI, extracted WAV, preprocessed WAV) are saved to `outputs/<video_name>/` by default.

### Vocals (default)
```bash
scripts/run_audio2midi.sh \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Piano
```bash
scripts/run_audio2midi.sh \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --instrument piano
```

### Guitar
```bash
scripts/run_audio2midi.sh \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --instrument guitar
```

### Vocals (explicit)
```bash
scripts/run_audio2midi.sh \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output out.mid \
  --instrument vocals \
  --bpm-detect-bin /path/to/bpm_detect \
  --click-track click.wav
```

Vocals with beat snapping:
```bash
scripts/run_audio2midi.sh \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output out.mid \
  --instrument vocals \
  --bpm-detect-bin /path/to/bpm_detect \
  --snap-to-beats
```

Vocals with manual BPM override:
```bash
scripts/run_audio2midi.sh \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output out.mid \
  --instrument vocals \
  --bpm-override 120
```

## CLI options
```bash
scripts/run_audio2midi.sh --help
```

Key options:
- `--instrument {piano,guitar,vocals}` — target instrument (default: vocals)
- `--backend {pti,piano-transcription-inference,basic-pitch,rmvpe}` — override transcription backend
- `--device cpu|cuda`
- `--denoise`
- `--min-duration <seconds>`
- `--note-threshold <0..1>`
- `--quantize-grid-seconds <seconds>`
- `--pti-checkpoint-path /path/to/model.pth`

Vocals-specific options:
- `--bpm-detect-bin <path>` — path to `bpm_detect` binary; falls back to PATH lookup, then 120 BPM
- `--bpm-override <bpm>` — skip auto-detection and use a fixed BPM
- `--click-track <path>` — write a click-track WAV mixed with the original audio
- `--snap-to-beats` — snap note boundaries to 16th-note grid on the detected beat grid
- `--f0-filter-frames <n>` — median filter window in frames (10 ms each) for F0 smoothing (default: 7)

Note segmentation internals (not exposed via CLI; tuned per proposal `plans/2026-04-26_vocal-midi-coverage-fragmentation/`):
- Same-pitch note merge ceiling: **one quarter note** (was one 8th note). Wider ceiling collapses repeated same-pitch fragments from vibrato trough interruptions; the existing beat-gate prevents merging across genuine rhythmic repetitions.
- F0-guided gap-fill: after merge passes, adjacent same-pitch notes whose gap contains a voiced F0 frame of the same semitone are merged. Activates when beat-gating blocked a merge but the gap contains real vocal signal.

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

## Evaluating output quality

`scripts/eval_vocal_midi.py` measures vocal MIDI transcription quality by
comparing the MIDI note roll against the F0 contour extracted from the vocals
stem via pYIN.

```bash
source .venv/bin/activate
# From a workdir (paths auto-discovered):
python scripts/eval_vocal_midi.py --workdir "outputs/<timestamp>_<name>/"

# Or with explicit paths:
python scripts/eval_vocal_midi.py \
  "outputs/<workdir>/<name>.mid" \
  "outputs/<workdir>/stems/htdemucs/<track>/vocals.wav" \
  "outputs/<workdir>/extracted/<track>.wav"
```

Outputs JSON to stdout with:
- `pitch_coverage.coverage_ratio` — fraction of voiced frames with an active MIDI note (target ≥ 0.90)
- `pitch_coverage.hallucination_ratio` — MIDI notes fired on unvoiced frames (target ≤ 0.05)
- `pitch_coverage.frames_within_half_semitone_pct` — pitch accuracy on covered frames (target ≥ 85%)
- `onset_regularity.ioi_cv` — note fragmentation (target ≤ 1.0)
- `midi_properties.note_duration_median_s` — median note length (target ≥ 0.3 s for phrase-level)

Redirect stdout to a file to save measurements for an `assessment.md` report:
```bash
python scripts/eval_vocal_midi.py <midi> <vocals> <mix> \
  > research/<date>_<topic>-eval/measurements.json
```

## Dev/test
```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

## Notes
- Historical C++ planning docs are kept in `C_PLAN.md` and `C_RESEARCH.md`.
