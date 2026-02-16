"""Audio preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audio2midi.exceptions import MissingDependencyError


def _import_scipy_signal():
    try:
        from scipy import signal  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("scipy is required for preprocessing.") from exc
    return signal


def _import_librosa():
    try:
        import librosa  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("librosa is required for audio loading.") from exc
    return librosa


def _import_soundfile():
    try:
        import soundfile as sf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError(
            "soundfile is required for audio writing."
        ) from exc
    return sf


def load_audio(audio_path: Path, sample_rate: int = 44100) -> tuple[np.ndarray, int]:
    """Load mono float32 audio at target sample rate."""
    librosa = _import_librosa()
    audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    return np.asarray(audio, dtype=np.float32), int(sr)


def write_audio(audio_path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write float32 WAV audio."""
    sf = _import_soundfile()
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), audio.astype(np.float32), sample_rate)


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize waveform peak to target."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return audio.astype(np.float32, copy=True)
    scaled = audio * (target_peak / peak)
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def high_pass_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = 35.0,
    order: int = 4,
) -> np.ndarray:
    """Apply Butterworth high-pass filter to remove low-frequency rumble."""
    signal = _import_scipy_signal()
    sos = signal.butter(
        order,
        cutoff_hz,
        btype="highpass",
        fs=sample_rate,
        output="sos",
    )
    filtered = signal.sosfilt(sos, audio)
    return np.asarray(filtered, dtype=np.float32)


def spectral_gate_denoise(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 1.25,
) -> np.ndarray:
    """Apply a light spectral gate for stationary noise."""
    signal = _import_scipy_signal()
    _, _, stft = signal.stft(audio, fs=sample_rate, nperseg=2048, noverlap=1536)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    floor = np.percentile(magnitude, 15, axis=1, keepdims=True) * strength
    gated_mag = np.maximum(magnitude - floor, 0.0)
    reconstructed = gated_mag * np.exp(1j * phase)
    _, denoised = signal.istft(
        reconstructed,
        fs=sample_rate,
        nperseg=2048,
        noverlap=1536,
    )
    if denoised.shape[0] < audio.shape[0]:
        denoised = np.pad(denoised, (0, audio.shape[0] - denoised.shape[0]))
    denoised = denoised[: audio.shape[0]]
    return np.asarray(denoised, dtype=np.float32)


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    denoise: bool = False,
) -> np.ndarray:
    """Normalize, high-pass filter, and optionally denoise audio."""
    processed = normalize_audio(audio)
    processed = high_pass_filter(processed, sample_rate=sample_rate)
    if denoise:
        processed = spectral_gate_denoise(processed, sample_rate=sample_rate)
    return normalize_audio(processed)


def preprocess_wav_file(
    input_wav_path: Path,
    output_wav_path: Path,
    denoise: bool = False,
    sample_rate: int = 44100,
) -> Path:
    """Load, preprocess, and persist WAV for downstream inference."""
    audio, sr = load_audio(input_wav_path, sample_rate=sample_rate)
    processed = preprocess_audio(audio, sample_rate=sr, denoise=denoise)
    write_audio(output_wav_path, processed, sample_rate=sr)
    return output_wav_path
