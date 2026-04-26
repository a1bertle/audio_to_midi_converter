"""Vocal transcription backend using RMVPE pitch tracker.

Pipeline:
  1. Stem separation (htdemucs 4-stem via demucs) → vocals.wav
  2. Pitch tracking (RMVPE) → F0 contour at 10 ms hop
  3. Onset detection (librosa) → syllable boundary times
  4. Note segmentation (semitone-snap, silence-aware merge, beat-gated onset split)
  5. Optional snap-to-beats: quantize note boundaries to nearest beat subdivision
  6. Return TranscriptionResult with NoteEvents

BPM and beat times are supplied externally (from bpm_detect run on the raw mix
in the CLI layer) rather than detected inside this module.

Checkpoint: rmvpe.pt must be present at the path passed to __init__
(default: ~/.cache/audio2midi/rmvpe.pt). Downloaded automatically on first
use from HuggingFace (lj1995/VoiceConversionWebUI).
"""

from __future__ import annotations

import logging
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


def _apply_f0_median_filter(f0_hz: np.ndarray, window_frames: int) -> np.ndarray:
    """Smooth the raw F0 contour per voiced segment to suppress vibrato.

    Applied before semitone snapping so short pitch oscillations (vibrato,
    portamento) are averaged out in Hz rather than producing alternating
    semitone sequences in the note list.

    Window default of 7 frames (70 ms) is grounded in measured vibrato rate
    of 5.14 Hz (period 194.5 ms) on a representative J-pop vocal track, and
    consistent with Nix et al. (2016) Journal of Voice 30(6) which reports
    typical singing vibrato at 4.5–6.5 Hz. 70 ms = 0.36× the median period,
    safely below the half-period threshold where step smearing begins.
    """
    if window_frames <= 1:
        return f0_hz
    from scipy.ndimage import median_filter as _mf
    result = f0_hz.copy().astype(float)
    voiced = f0_hz > 0
    changes = np.diff(voiced.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for s, e in zip(starts, ends):
        if e - s >= window_frames:
            result[s:e] = _mf(f0_hz[s:e].astype(float), size=window_frames)
    return result


def _parse_beat_times(click_wav_path: Path) -> np.ndarray:
    """Extract beat times from a bpm_detect click-track WAV.

    bpm_detect mixes a 1000 Hz click on every beat and a 1500 Hz click on
    downbeats into the original audio. We isolate the click signal with a
    narrow bandpass around 1000 Hz, then pick peaks to recover beat times.

    Returns a sorted float array of beat times in seconds (may be empty if
    the file cannot be parsed).
    """
    try:
        y, sr = librosa.load(str(click_wav_path), sr=None, mono=True)
    except Exception as exc:
        LOGGER.warning("Could not load click track %s: %s", click_wav_path, exc)
        return np.array([], dtype=float)

    # Bandpass 900–1600 Hz to capture both beat (1000 Hz) and downbeat (1500 Hz) clicks.
    from scipy.signal import butter, sosfilt
    nyq = sr / 2.0
    sos = butter(4, [900 / nyq, 1600 / nyq], btype="band", output="sos")
    filtered = sosfilt(sos, y)

    # Rectify and smooth to get an envelope, then pick peaks.
    envelope = np.abs(filtered)
    hop = int(sr * 0.01)  # 10 ms
    frames = librosa.util.frame(envelope, frame_length=hop, hop_length=hop)
    rms = frames.mean(axis=0)

    # Minimum distance between peaks: 60% of expected beat interval.
    # Estimate from autocorrelation peak spacing if possible; fall back to 0.3 s.
    min_dist_frames = max(1, int(0.3 * sr / hop))
    from scipy.signal import find_peaks
    peak_frames, _ = find_peaks(rms, distance=min_dist_frames, height=rms.mean() * 1.5)
    beat_times = peak_frames * hop / sr

    LOGGER.info("Parsed %d beat times from click track", len(beat_times))
    return beat_times.astype(float)


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


def _merge_same_pitch(
    raw: list[tuple[float, float, int]],
    max_gap_s: float,
    beat_arr: np.ndarray,
    beat_tol: float,
) -> list[tuple[float, float, int]]:
    """Collapse consecutive same-pitch notes whose gap is <= max_gap_s.

    A merge is blocked when a beat-aligned time point falls strictly inside
    the gap, preserving genuine repeated syllables on beat boundaries.
    """
    changed = True
    while changed:
        changed = False
        out: list[tuple[float, float, int]] = []
        i = 0
        while i < len(raw):
            if i + 1 < len(raw):
                s0, e0, n0 = raw[i]
                s1, e1, n1 = raw[i + 1]
                gap = s1 - e0
                if n0 == n1 and 0 <= gap <= max_gap_s:
                    beat_in_gap = beat_arr.size > 0 and any(
                        _nearest_beat_distance(t, beat_arr) <= beat_tol
                        for t in np.linspace(e0, s1, max(2, int(gap / 0.01)))
                        if e0 < t < s1
                    )
                    if not beat_in_gap:
                        out.append((s0, e1, n0))
                        i += 2
                        changed = True
                        continue
            out.append(raw[i])
            i += 1
        raw = out
    return raw


def _nearest_beat_distance(t: float, beat_times: np.ndarray) -> float:
    """Return seconds to the nearest beat in beat_times, or inf if empty."""
    if len(beat_times) == 0:
        return float("inf")
    idx = np.searchsorted(beat_times, t)
    candidates = beat_times[max(0, idx - 1): idx + 1]
    return float(np.min(np.abs(candidates - t)))


def _snap_to_grid(t: float, beat_times: np.ndarray, subdivisions: int) -> float:
    """Snap time t to the nearest beat subdivision on the beat grid.

    Inserts `subdivisions` evenly-spaced grid points between each pair of
    consecutive beats and returns the closest one to t.
    """
    if len(beat_times) < 2:
        return t
    idx = np.searchsorted(beat_times, t)
    # Build a local grid from surrounding beats.
    lo_idx = max(0, idx - 1)
    hi_idx = min(len(beat_times) - 1, idx)
    if lo_idx == hi_idx:
        return t
    beat_dur = beat_times[hi_idx] - beat_times[lo_idx]
    sub_dur = beat_dur / subdivisions
    grid_start = beat_times[lo_idx]
    # Extend grid one beat in each direction for boundary safety.
    grid = np.arange(
        grid_start - beat_dur,
        grid_start + 2 * beat_dur + sub_dur * 0.5,
        sub_dur,
    )
    nearest = grid[np.argmin(np.abs(grid - t))]
    return float(nearest)


def _pitch_track_to_notes(
    f0_hz: np.ndarray,
    onset_times: np.ndarray,
    min_note_duration_s: float,
    bpm: float,
    beat_times: np.ndarray | None = None,
    snap_to_beats: bool = False,
    f0_filter_frames: int = 7,
) -> list[NoteEvent]:
    """Convert F0 contour + onsets to NoteEvents."""
    f0_hz = _apply_f0_median_filter(f0_hz, f0_filter_frames)
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

    # Same-pitch merge (pass 1): collapse same-pitch fragments up to one 8th
    # note gap, preserving beat-boundary separations.
    eighth_s = (60.0 / bpm) / 2.0
    beat_arr_pre = np.sort(beat_times) if beat_times is not None and len(beat_times) > 0 \
        else np.array([], dtype=float)
    pre_beat_tol = (
        float(np.median(np.diff(beat_arr_pre))) * 0.25
        if beat_arr_pre.size > 0 else 0.0
    )
    raw = _merge_same_pitch(raw, eighth_s, beat_arr_pre, pre_beat_tol)

    # Onset-based syllable splitting — beat-gated when beat_times provided.
    # A split is only allowed when the onset aligns to a beat (within 25% of
    # the beat interval) OR when no beat grid is available.
    # The minimum fragment length is raised to a 16th note to suppress
    # sub-rhythmic artifacts.
    beat_arr = np.sort(beat_times) if beat_times is not None and len(beat_times) > 0 \
        else np.array([], dtype=float)
    sixteenth_s = (60.0 / bpm) / 4.0  # one 16th note duration
    split_min_s = max(min_note_duration_s, sixteenth_s)

    if beat_arr.size > 0:
        # Estimate beat interval from median gap between consecutive beats.
        beat_interval = float(np.median(np.diff(beat_arr)))
        beat_tolerance = beat_interval * 0.25
    else:
        beat_interval = 60.0 / bpm
        beat_tolerance = beat_interval  # no gating — allow all onsets

    if len(onset_times) > 0:
        split: list[tuple[float, float, int]] = []
        onset_arr = np.sort(onset_times)
        margin = split_min_s * 0.5
        for start_s, end_s, note in raw:
            interior = onset_arr[
                (onset_arr > start_s + margin) & (onset_arr < end_s - margin)
            ]
            # Filter to beat-aligned onsets only.
            if beat_arr.size > 0:
                interior = np.array([
                    o for o in interior
                    if _nearest_beat_distance(o, beat_arr) <= beat_tolerance
                ])
            if len(interior) == 0:
                split.append((start_s, end_s, note))
            else:
                boundaries = [start_s] + list(interior) + [end_s]
                for a, b in zip(boundaries[:-1], boundaries[1:]):
                    if b - a >= split_min_s:
                        split.append((a, b, note))
        raw = split

    # Same-pitch merge (pass 2): the onset split may have re-cut merged notes;
    # clean up any same-pitch adjacencies it reintroduced.
    raw = _merge_same_pitch(raw, eighth_s, beat_arr_pre, pre_beat_tol)

    # Snap note boundaries to nearest 16th-note subdivision on the beat grid.
    if snap_to_beats and beat_arr.size >= 2:
        snapped: list[tuple[float, float, int]] = []
        for start_s, end_s, note in raw:
            new_start = _snap_to_grid(start_s, beat_arr, subdivisions=4)
            new_end = _snap_to_grid(end_s, beat_arr, subdivisions=4)
            # Guard: snapping must not collapse or invert the note.
            if new_end - new_start >= split_min_s * 0.5:
                snapped.append((new_start, new_end, note))
            else:
                snapped.append((start_s, end_s, note))
        raw = snapped

    return [
        NoteEvent(pitch=note, start=start, end=end, velocity=80)
        for start, end, note in raw
    ]


class RmvpeTranscriber(BaseTranscriber):
    """Vocal melody transcriber: htdemucs → RMVPE → beat-gated onset split → NoteEvents.

    BPM and beat_times are supplied externally (run bpm_detect on the raw mix
    before calling transcribe). This keeps tempo detection on a full-spectrum
    signal rather than a stem-separated vocals track.
    """

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        device: str = "cpu",
        bpm_override: float | None = None,
        beat_times: np.ndarray | None = None,
        snap_to_beats: bool = False,
        f0_filter_frames: int = 7,
        stems_dir: Path | None = None,
    ) -> None:
        self._checkpoint_path = Path(
            checkpoint_path or _DEFAULT_CHECKPOINT_DIR / _CHECKPOINT_FILENAME
        )
        self._device = device
        self._bpm_override = bpm_override
        self._beat_times = beat_times if beat_times is not None else np.array([], dtype=float)
        self._snap_to_beats = snap_to_beats
        self._f0_filter_frames = f0_filter_frames
        self._stems_dir = stems_dir
        self._model = None  # lazy load

    def _get_model(self):
        if self._model is None:
            ckpt = _ensure_checkpoint(self._checkpoint_path)
            self._model = _load_rmvpe(ckpt, self._device)
        return self._model

    def transcribe(self, wav_path: Path) -> TranscriptionResult:
        """Run the full vocal transcription pipeline.

        BPM and beat_times must be set before calling (via constructor args).
        Stem separation and pitch tracking run on the provided wav_path;
        tempo/rhythm information comes from the externally-supplied beat grid.
        """
        bpm = self._bpm_override or (
            float(60.0 / np.median(np.diff(self._beat_times)))
            if len(self._beat_times) >= 2
            else 120.0
        )
        LOGGER.info(
            "Using BPM=%.2f, beat_times=%d beats, snap=%s",
            bpm, len(self._beat_times), self._snap_to_beats,
        )

        with tempfile.TemporaryDirectory(prefix="audio2midi_vocals_") as tmp:
            tmp_path = Path(tmp)
            stems_out = self._stems_dir if self._stems_dir is not None else tmp_path / "stems"

            # 1. Stem separation — runs on raw mix, unaffected by click track
            vocals_path = _separate_vocals(wav_path, stems_out)

            # 2. Pitch tracking
            LOGGER.info("Running RMVPE pitch tracker...")
            model = self._get_model()
            y, _ = librosa.load(str(vocals_path), sr=_TARGET_SR, mono=True)
            f0 = model.infer_from_audio(y, thred=0.03)

            # 3. Onset detection on native-sr vocals stem
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

            # 4. Note segmentation — 32nd note floor, 16th note split minimum
            beat_s = 60.0 / bpm
            min_note_s = beat_s / 8.0  # 1/32nd note floor
            LOGGER.info(
                "Segmenting notes: BPM=%.2f, min_duration=%.1fms (32nd note floor)",
                bpm, min_note_s * 1000,
            )
            notes = _pitch_track_to_notes(
                f0, onset_times, min_note_s, bpm,
                beat_times=self._beat_times,
                snap_to_beats=self._snap_to_beats,
                f0_filter_frames=self._f0_filter_frames,
            )
            LOGGER.info("Notes produced: %d", len(notes))

        return TranscriptionResult(
            notes=notes, pedals=[], instrument=Instrument.VOCALS, bpm=bpm
        )
