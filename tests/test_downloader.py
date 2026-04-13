"""Tests for downloader utilities and cache behavior."""

from pathlib import Path

import pytest

from audio2midi.downloader import (
    YouTubeDownloader,
    build_ydl_options,
    extract_video_id,
    validate_youtube_url,
)
from audio2midi.exceptions import InvalidInputError


def test_extract_video_id_youtube_query() -> None:
    assert extract_video_id("https://www.youtube.com/watch?v=abc123") == "abc123"


def test_extract_video_id_youtu_be() -> None:
    assert extract_video_id("https://youtu.be/xyz987") == "xyz987"


def test_validate_youtube_url_rejects_non_youtube() -> None:
    with pytest.raises(InvalidInputError):
        validate_youtube_url("https://example.com/watch?v=abc123")


def test_build_ydl_options_enforces_no_playlist_by_default(tmp_path: Path) -> None:
    options = build_ydl_options(downloads_dir=tmp_path)
    assert options["noplaylist"] is True
    assert options["retries"] == 3
    assert "%(id)s.%(ext)s" in options["outtmpl"]


def test_find_cached_media_hit_and_ignore_metadata(tmp_path: Path) -> None:
    downloader = YouTubeDownloader(workdir=tmp_path)
    (tmp_path / "downloads").mkdir(exist_ok=True)
    media = tmp_path / "downloads" / "abc123.webm"
    metadata = tmp_path / "downloads" / "abc123.json"
    media.write_bytes(b"data")
    metadata.write_text("{}", encoding="utf-8")
    found = downloader.find_cached_media("abc123")
    assert found == media


def test_find_cached_media_miss(tmp_path: Path) -> None:
    downloader = YouTubeDownloader(workdir=tmp_path)
    assert downloader.find_cached_media("missing123") is None


def test_build_ydl_options_mp3_adds_postprocessor(tmp_path: Path) -> None:
    options = build_ydl_options(downloads_dir=tmp_path, mp3=True)
    assert "postprocessors" in options
    pp = options["postprocessors"][0]
    assert pp["key"] == "FFmpegExtractAudio"
    assert pp["preferredcodec"] == "mp3"


def test_build_ydl_options_no_mp3_has_no_postprocessor(tmp_path: Path) -> None:
    options = build_ydl_options(downloads_dir=tmp_path, mp3=False)
    assert "postprocessors" not in options


def test_youtube_downloader_mp3_flag_stored(tmp_path: Path) -> None:
    downloader = YouTubeDownloader(workdir=tmp_path, mp3=True)
    assert downloader.mp3 is True
