"""Vocal transcription backend using RMVPE pitch tracker.

Pipeline:
  1. Stem separation (htdemucs 4-stem via demucs) → vocals.wav
  2. BPM detection (bpm_detect CLI) → tempo in BPM
  3. Pitch tracking (RMVPE) → F0 contour at 10 ms hop
  4. Onset detection (librosa) → syllable boundary times
  5. Note segmentation (semitone-snap, silence-aware merge, onset split)
  6. Return TranscriptionResult with NoteEvents

Checkpoint: rmvpe.pt must be present at the path passed to __init__
(default: ~/.cache/audio2midi/rmvpe.pt). Downloaded automatically on first
use from HuggingFace (lj1995/VoiceConversionWebUI).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np

from audio2midi.exceptions import MissingDependencyError, TranscriptionError
from audio2midi.models import Instrument, NoteEvent, TranscriptionResult
from audio2midi.transcribers.base import BaseTranscriber

LOGGER = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT_DIR = Path.home() / ".cache" / "audio2midi"
_CHECKPOINT_FILENAME = "rmvpe.pt"
_HF_REPO = "lj1995/VoiceConversionWebUI"

# RMVPE inference constants
_TARGET_SR = 16000
_HOP_S = 160.0 / _TARGET_SR  # 10 ms per frame

# Onset detection
_ONSET_HOP = 256
_ONSET_DELTA = 0.07


def _ensure_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path
    LOGGER.info("Downloading rmvpe.pt from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as exc:
        raise MissingDependencyError(
            "huggingface_hub is required to auto-download rmvpe.pt. "
            "Install with: pip install huggingface_hub"
        ) from exc
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=_HF_REPO,
        filename=_CHECKPOINT_FILENAME,
        local_dir=str(checkpoint_path.parent),
    )
    return Path(downloaded)


def _detect_bpm(
    audio_path: Path,
    bpm_detect_bin: Path | None,
    click_track_path: Path | None = None,
) -> float | None:
    """Run bpm_detect binary and return detected BPM, or None if unavailable.

    bpm_detect does not support WAV input; if the provided file is a WAV it is
    transcoded to a temporary MP3 via ffmpeg before being passed to bpm_detect.

    If click_track_path is given, bpm_detect writes a click-track WAV overlay
    to that path; otherwise the click output is discarded.
    """
    binary = bpm_detect_bin or shutil.which("bpm_detect")
    if binary is None:
        LOGGER.warning(
            "bpm_detect not found; using default BPM=120. "
            "Pass bpm_detect_bin or put bpm_detect on PATH for accurate tempo."
        )
        return None

    input_path = audio_path
    _tmp_mp3: "tempfile.NamedTemporaryFile | None" = None  # noqa: F821

    if audio_path.suffix.lower() == ".wav":
        import tempfile
        _tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        _tmp_mp3.close()
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path), "-q:a", "2",
                 _tmp_mp3.name],
                capture_output=True,
                check=True,
                timeout=120,
            )
            input_path = Path(_tmp_mp3.name)
        except Exception as exc:
            LOGGER.warning("ffmpeg WAV→MP3 conversion failed: %s; using WAV directly", exc)
            input_path = audio_path

    click_out = str(click_track_path) if click_track_path is not None else "/dev/null"
    try:
        result = subprocess.run(
            [str(binary), "-v", str(input_path), "-o", click_out],
            capture_output=True,
            text=True,
            timeout=120,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Detected BPM:"):
                bpm = float(line.split(":")[1].strip())
                LOGGER.info("Detected BPM: %.2f", bpm)
                return bpm
    except Exception as exc:
        LOGGER.warning("bpm_detect failed: %s", exc)
    finally:
        if _tmp_mp3 is not None:
            Path(_tmp_mp3.name).unlink(missing_ok=True)
    return None


def _separate_vocals(audio_path: Path, out_dir: Path) -> Path:
    """Run htdemucs 4-stem separation and return path to vocals.wav."""
    try:
        import demucs.separate  # noqa: F401 – presence check only
    except ImportError as exc:
        raise MissingDependencyError(
            "demucs is required for vocal stem separation. "
            "Install with: pip install demucs"
        ) from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", "-m", "demucs",
        "--name", "htdemucs",
        "--device", "mps",
        "--out", str(out_dir),
        str(audio_path),
    ]
    LOGGER.info("Running htdemucs stem separation...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise TranscriptionError(
            f"demucs separation failed:\n{result.stderr[-500:]}"
        )
    stem_dir = out_dir / "htdemucs" / audio_path.stem
    vocals_path = stem_dir / "vocals.wav"
    if not vocals_path.exists():
        # demucs sometimes uses the full filename as the stem dir name
        candidates = list(out_dir.rglob("vocals.wav"))
        if not candidates:
            raise TranscriptionError(
                f"vocals.wav not found after separation in {out_dir}"
            )
        vocals_path = candidates[0]
    LOGGER.info("Vocals stem: %s", vocals_path)
    return vocals_path


def _load_rmvpe(checkpoint_path: Path, device: str):
    """Import and instantiate the Applio RMVPE predictor."""
    # The Applio RMVPE implementation is vendored inside the package.
    try:
        from audio2midi.transcribers._rmvpe_impl import RMVPE0Predictor  # type: ignore
    except ImportError as exc:
        raise MissingDependencyError(
            "RMVPE implementation not found. "
            "Ensure audio2midi/transcribers/_rmvpe_impl.py is present."
        ) from exc
    return RMVPE0Predictor(str(checkpoint_path), device=device)


def _pitch_track_to_notes(
    f0_hz: np.ndarray,
    onset_times: np.ndarray,
    min_note_duration_s: float,
    bpm: float,
) -> list[NoteEvent]:
    """Convert F0 contour + onsets to NoteEvents."""
    times = np.arange(len(f0_hz)) * _HOP_S
    lo = librosa.note_to_hz("C2")
    hi = librosa.note_to_hz("C6")
    voiced = (f0_hz > 0) & (f0_hz >= lo) & (f0_hz <= hi)

    # Snap to semitones
    midi_track = np.where(
        voiced,
        np.round(12 * np.log2(np.maximum(f0_hz, 1e-6) / 440.0) + 69).astype(int),
        -1,
    )

    # Segment into raw notes
    raw: list[tuple[float, float, int]] = []  # (start_s, end_s, midi_note)
    in_note = False
    note_start = 0.0
    note_midi = 0

    for t, midi in zip(times, midi_track):
        if midi >= 0:
            if not in_note:
                in_note = True
                note_start = float(t)
                note_midi = int(midi)
            elif int(midi) != note_midi:
                if float(t) - note_start >= min_note_duration_s:
                    raw.append((note_start, float(t), note_midi))
                note_start = float(t)
                note_midi = int(midi)
        else:
            if in_note:
                if float(t) - note_start >= min_note_duration_s:
                    raw.append((note_start, float(t), note_midi))
                in_note = False

    # Silence-aware merge: collapse short ornament notes but never across gaps
    max_silence_s = min_note_duration_s * 0.5
    gap_s = min_note_duration_s * 1.5
    changed = True
    while changed:
        changed = False
        out: list[tuple[float, float, int]] = []
        i = 0
        while i < len(raw):
            if i + 2 < len(raw):
                s0, e0, n0 = raw[i]
                s1, e1, n1 = raw[i + 1]
                s2, e2, n2 = raw[i + 2]
                if (s1 - e0 <= max_silence_s and s2 - e1 <= max_silence_s
                        and n0 == n2 and (e1 - s1) < gap_s):
                    out.append((s0, e2, n0))
                    i += 3
                    changed = True
                    continue
            if i + 1 < len(raw):
                s0, e0, n0 = raw[i]
                s1, e1, n1 = raw[i + 1]
                if (s1 - e0 <= max_silence_s
                        and (e1 - s1) < gap_s
                        and abs(n1 - n0) <= 2):
                    out.append((s0, e1, n0))
                    i += 2
                    changed = True
                    continue
            out.append(raw[i])
            i += 1
        raw = out

    # Onset-based syllable splitting
    if len(onset_times) > 0:
        split: list[tuple[float, float, int]] = []
        onset_arr = np.sort(onset_times)
        margin = min_note_duration_s * 0.5
        for start_s, end_s, note in raw:
            interior = onset_arr[
                (onset_arr > start_s + margin) & (onset_arr < end_s - margin)
            ]
            if len(interior) == 0:
                split.append((start_s, end_s, note))
            else:
                boundaries = [start_s] + list(interior) + [end_s]
                for a, b in zip(boundaries[:-1], boundaries[1:]):
                    if b - a >= min_note_duration_s:
                        split.append((a, b, note))
        raw = split

    return [
        NoteEvent(pitch=note, start=start, end=end, velocity=80)
        for start, end, note in raw
    ]


class RmvpeTranscriber(BaseTranscriber):
    """Vocal melody transcriber: htdemucs → RMVPE → onset split → NoteEvents."""

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        bpm_detect_bin: Path | None = None,
        device: str = "cpu",
        bpm_override: float | None = None,
        click_track_path: Path | None = None,
    ) -> None:
        self._checkpoint_path = Path(
            checkpoint_path or _DEFAULT_CHECKPOINT_DIR / _CHECKPOINT_FILENAME
        )
        self._bpm_detect_bin = Path(bpm_detect_bin) if bpm_detect_bin else None
        self._device = device
        self._bpm_override = bpm_override
        self._click_track_path = Path(click_track_path) if click_track_path else None
        self._model = None  # lazy load

    def _get_model(self):
        if self._model is None:
            ckpt = _ensure_checkpoint(self._checkpoint_path)
            self._model = _load_rmvpe(ckpt, self._device)
        return self._model

    def transcribe(self, wav_path: Path) -> TranscriptionResult:
        """Run the full vocal transcription pipeline."""
        with tempfile.TemporaryDirectory(prefix="audio2midi_vocals_") as tmp:
            tmp_path = Path(tmp)

            # 1. Stem separation
            vocals_path = _separate_vocals(wav_path, tmp_path / "stems")

            # 2. BPM detection
            if self._bpm_override is not None:
                bpm = self._bpm_override
                LOGGER.info("Using BPM override: %.2f", bpm)
            else:
                bpm = _detect_bpm(
                    wav_path, self._bpm_detect_bin, self._click_track_path
                ) or 120.0

            # 3. Pitch tracking
            LOGGER.info("Running RMVPE pitch tracker...")
            model = self._get_model()
            y, _ = librosa.load(str(vocals_path), sr=_TARGET_SR, mono=True)
            f0 = model.infer_from_audio(y, thred=0.03)

            # 4. Onset detection on native-sr vocals stem
            y_native, sr_native = librosa.load(str(vocals_path), sr=None, mono=True)
            onset_frames = librosa.onset.onset_detect(
                y=y_native, sr=sr_native,
                hop_length=_ONSET_HOP,
                delta=_ONSET_DELTA,
                backtrack=True,
            )
            onset_times = librosa.frames_to_time(
                onset_frames, sr=sr_native, hop_length=_ONSET_HOP
            )
            LOGGER.info(
                "Onsets detected: %d (%.2f/s)", len(onset_times),
                len(onset_times) / (len(y) / _TARGET_SR),
            )

            # 5. Note segmentation — 32nd note minimum at detected BPM
            beat_s = 60.0 / bpm
            min_note_s = beat_s / 8.0  # 1/32nd note
            LOGGER.info(
                "Segmenting notes: BPM=%.2f, min_duration=%.1fms (32nd note)",
                bpm, min_note_s * 1000,
            )
            notes = _pitch_track_to_notes(f0, onset_times, min_note_s, bpm)
            LOGGER.info("Notes produced: %d", len(notes))

        return TranscriptionResult(
            notes=notes, pedals=[], instrument=Instrument.VOCALS, bpm=bpm
        )
