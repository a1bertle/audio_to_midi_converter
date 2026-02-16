# audio_to_midi_converter

Offline C++ and Python pipelines for converting YouTube piano performances (including room noise) into MIDI.

## Python CLI (Implemented)

This repository now includes a Python implementation of the pipeline with:
- YouTube download and cache (`yt-dlp`)
- WAV extraction (`ffmpeg`)
- Preprocessing (normalize, high-pass, optional denoise)
- Transcription backend abstraction (`piano-transcription-inference`, fallback `basic-pitch`)
- MIDI writing

### Quick start
```bash
scripts/venv_create.sh
source .venv/bin/activate
pip install -r requirements.txt
```

### Run
```bash
scripts/run_audio2midi.sh \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output out.mid \
  --workdir .cache/audio2midi \
  --keep-intermediate
```

### PTI checkpoint note
- `piano-transcription-inference` requires a `.pth` checkpoint.
- The app now downloads it directly via Python (no `wget` required).
- If Python download fails, it automatically falls back to `curl` when available.
- Optional override:
```bash
scripts/run_audio2midi.sh \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output out.mid \
  --pti-checkpoint-path "/path/to/note_F1=0.9677_pedal_F1=0.9186.pth"
```
- Optional URL override (advanced):
```bash
export AUDIO2MIDI_PTI_CHECKPOINT_URL="https://example.com/note_model.pth"
```

### Optional: launch activated shell
```bash
scripts/venv_activate.sh
```

### Run tests
```bash
pip install -r requirements-dev.txt
pytest -q
```

## Prerequisites

### Required tools
- C++17 compiler (`clang++` or `g++`)
- CMake
- ONNX Runtime C/C++ package
- `ffmpeg`
- `yt-dlp`

### Install on macOS (Homebrew)
```bash
xcode-select --install
brew install cmake ffmpeg yt-dlp
```

### Install on Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y build-essential cmake ffmpeg python3-pip
python3 -m pip install -U yt-dlp
```

### Install ONNX Runtime (C/C++ API)
Use the official ONNX Runtime install/docs pages and download a prebuilt C/C++ release archive (`.tgz`/`.zip`) for your platform from GitHub releases.

1. Open:
- https://onnxruntime.ai/docs/install/
- https://onnxruntime.ai/docs/get-started/with-cpp.html
- https://github.com/microsoft/onnxruntime/releases

2. Download the asset matching your platform, for example:
- `onnxruntime-osx-arm64-<version>.tgz`
- `onnxruntime-linux-x64-<version>.tgz`

3. Extract it to a local folder, for example:
```bash
mkdir -p third_party
tar -xzf ~/Downloads/onnxruntime-<platform>-<version>.tgz -C third_party
```

4. Point CMake at the extracted directory when configuring:
```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/third_party/onnxruntime-<platform>-<version>"
```

## Verify prerequisite installation
```bash
cmake --version
ffmpeg -version
yt-dlp --version
```
