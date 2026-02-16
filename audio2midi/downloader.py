"""YouTube download and cache handling."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from audio2midi.exceptions import DownloadError, InvalidInputError
from audio2midi.models import DownloadResult

try:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError as YtDlpDownloadError
except ImportError:  # pragma: no cover - dependency may be unavailable in CI.
    YoutubeDL = None
    YtDlpDownloadError = Exception


SUPPORTED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def validate_youtube_url(url: str) -> None:
    """Validate that URL belongs to YouTube and includes a video id."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host not in SUPPORTED_HOSTS:
        raise InvalidInputError(f"Unsupported host in URL: {host}")
    extract_video_id(url)


def extract_video_id(url: str) -> str:
    """Extract the video id from supported YouTube URL formats."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
        if not video_id:
            raise InvalidInputError("Missing video id in youtu.be URL.")
        return video_id
    query = parse_qs(parsed.query)
    values = query.get("v")
    if not values or not values[0]:
        raise InvalidInputError("Missing v=VIDEO_ID query parameter in URL.")
    return values[0]


def build_ydl_options(
    downloads_dir: Path,
    retries: int = 3,
    allow_playlist: bool = False,
) -> dict:
    """Build deterministic yt-dlp options."""
    return {
        "format": "bestaudio/best",
        "outtmpl": str(downloads_dir / "%(id)s.%(ext)s"),
        "noplaylist": not allow_playlist,
        "retries": retries,
        "quiet": True,
        "no_warnings": True,
    }


class YouTubeDownloader:
    """Downloader with simple cache-by-video-id behavior."""

    def __init__(
        self,
        workdir: Path,
        retries: int = 3,
        allow_playlist: bool = False,
    ) -> None:
        self.workdir = Path(workdir)
        self.downloads_dir = self.workdir / "downloads"
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self.allow_playlist = allow_playlist

    def find_cached_media(self, video_id: str) -> Path | None:
        """Find an already downloaded media file for the given video id."""
        for candidate in sorted(self.downloads_dir.glob(f"{video_id}.*")):
            if candidate.suffix.lower() == ".json":
                continue
            return candidate
        return None

    def _metadata_path(self, video_id: str) -> Path:
        return self.downloads_dir / f"{video_id}.json"

    def _read_cached_metadata(self, video_id: str) -> dict:
        metadata_path = self._metadata_path(video_id)
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def _write_metadata(self, video_id: str, info: dict) -> None:
        payload = {
            "id": info.get("id"),
            "title": info.get("title"),
            "webpage_url": info.get("webpage_url"),
            "duration": info.get("duration"),
        }
        with self._metadata_path(video_id).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def download(self, youtube_url: str) -> DownloadResult:
        """Download audio stream for the given URL or return cached media."""
        validate_youtube_url(youtube_url)
        video_id = extract_video_id(youtube_url)
        cached_media = self.find_cached_media(video_id)
        if cached_media is not None:
            cached = self._read_cached_metadata(video_id)
            return DownloadResult(
                video_id=video_id,
                media_path=cached_media,
                title=cached.get("title"),
                webpage_url=cached.get("webpage_url", youtube_url),
                duration_seconds=cached.get("duration"),
            )
        if YoutubeDL is None:
            raise DownloadError(
                "yt-dlp Python package is not installed. "
                "Install with: pip install yt-dlp"
            )
        options = build_ydl_options(
            downloads_dir=self.downloads_dir,
            retries=self.retries,
            allow_playlist=self.allow_playlist,
        )
        try:
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                media_path = Path(ydl.prepare_filename(info))
        except YtDlpDownloadError as exc:
            raise DownloadError(f"Failed to download URL: {youtube_url}") from exc
        if not media_path.exists():
            raise DownloadError(
                "yt-dlp reported success but file is missing: "
                f"{media_path}"
            )
        self._write_metadata(video_id, info)
        return DownloadResult(
            video_id=video_id,
            media_path=media_path,
            title=info.get("title"),
            webpage_url=info.get("webpage_url", youtube_url),
            duration_seconds=info.get("duration"),
        )
