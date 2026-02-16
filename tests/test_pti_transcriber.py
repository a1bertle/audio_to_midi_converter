"""Tests for PTI transcriber checkpoint path handling."""

from pathlib import Path

from audio2midi.transcribers.piano_transcription_inference import (
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_CHECKPOINT_URL,
    PianoTranscriptionInferenceTranscriber,
)


def test_resolve_checkpoint_path_from_constructor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    transcriber = PianoTranscriptionInferenceTranscriber(
        checkpoint_path=checkpoint
    )
    assert transcriber._resolve_checkpoint_path() == checkpoint.resolve()


def test_resolve_checkpoint_path_from_env(monkeypatch) -> None:
    checkpoint = "/tmp/pti_model.pth"
    monkeypatch.setenv("AUDIO2MIDI_PTI_CHECKPOINT_PATH", checkpoint)
    transcriber = PianoTranscriptionInferenceTranscriber()
    assert str(transcriber._resolve_checkpoint_path()) == str(Path(checkpoint))


def test_resolve_checkpoint_path_default_location() -> None:
    transcriber = PianoTranscriptionInferenceTranscriber()
    resolved = transcriber._resolve_checkpoint_path()
    assert resolved.name == DEFAULT_CHECKPOINT_NAME
    assert "piano_transcription_inference_data" in str(resolved)


def test_resolve_checkpoint_url_default_and_env(monkeypatch) -> None:
    transcriber = PianoTranscriptionInferenceTranscriber()
    assert transcriber._resolve_checkpoint_url() == DEFAULT_CHECKPOINT_URL
    override = "https://example.com/custom.pth"
    monkeypatch.setenv("AUDIO2MIDI_PTI_CHECKPOINT_URL", override)
    assert transcriber._resolve_checkpoint_url() == override
