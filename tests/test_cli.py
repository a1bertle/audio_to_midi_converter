"""Tests for CLI source selection.

Run standalone with: ``python -m pytest tests/test_cli.py``.
"""

import pytest

from audio2midi.cli import _build_parser


def test_cli_accepts_local_input_file() -> None:
    args = _build_parser().parse_args(["--input-file", "performance.mp3"])

    assert args.input_file == "performance.mp3"
    assert args.youtube_url is None


def test_cli_requires_exactly_one_media_source() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--input-file", "performance.mp3", "--youtube-url", "https://example.com",
        ])
