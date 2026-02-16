"""Tests for ffmpeg extraction command construction."""

from pathlib import Path

from audio2midi.extractor import build_ffmpeg_command


def test_build_ffmpeg_command_contains_expected_args() -> None:
    command = build_ffmpeg_command(
        input_media_path=Path("input.webm"),
        output_wav_path=Path("output.wav"),
        sample_rate=44100,
        channels=1,
        sample_fmt="flt",
        audio_codec="pcm_f32le",
    )
    assert command[0] == "ffmpeg"
    assert "-vn" in command
    assert "-ac" in command and "1" in command
    assert "-ar" in command and "44100" in command
    assert "-c:a" in command and "pcm_f32le" in command
    assert "-sample_fmt" in command and "flt" in command
    assert command[-1] == "output.wav"
