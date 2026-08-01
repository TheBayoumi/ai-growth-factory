from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .config import Settings
from .models import VideoPackage
from .policy import Observation, Strategy


@dataclass(frozen=True)
class ChannelContext:
    channel_id: str
    title: str
    subscribers: int
    total_views: int
    uploads_playlist: str


@dataclass(frozen=True)
class RecentVideo:
    video_id: str
    title: str
    description: str
    tags: list[str]
    published_at: datetime
    views: int
    likes: int
    comments: int
    average_view_percentage: float
    subscribers_gained: int
    subscribers_lost: int
    shares: int


class YouTubeClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.youtube_oauth:
            raise RuntimeError("YOUTUBE_OAUTH_JSON is missing")
        self.settings = settings
        self.oauth = settings.youtube_oauth
        self.access_token: str | None = None

    def _token(self) -> str:
        if self.access_token:
            return self.access_token
        response = requests.post("https://oauth2.googleapis.com/token", data={"client_id": self.oauth["client_id"], "client_secret": self.oauth["client_secret"], "refresh_token": self.oauth["refresh_token"], "grant_type": "refresh_token"}, timeout=30)
        response.raise_for_status()
        self.access_token = response.json()["access_token"]
        return self.access_token

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        headers = {"Authorization": f"Bearer {self._token()}"} | kwargs.pop("headers", {})
        response = requests.request(method, url, headers=headers, timeout=120, **kwargs)
        if response.status_code == 401:
            self.access_token = None
            headers["Authorization"] = f"Bearer {self._token()}"
            response = requests.request(method, url, headers=headers, timeout=120, **kwargs)
        response.raise_for_status()
        return response

    def channel_context(self) -> ChannelContext:
        item = self._request("GET", "https://www.googleapis.com/youtube/v3/channels", params={"part": "snippet,statistics,contentDetails", "mine": "true"}).json()["items"][0]
        return ChannelContext(item["id"], item["snippet"]["title"], int(item["statistics"].get("subscriberCount", 0)), int(item["statistics"].get("viewCount", 0)), item["contentDetails"]["relatedPlaylists"]["uploads"])

    def recent_videos(self, context: ChannelContext) -> list[RecentVideo]:
        playlist = self._request("GET", "https://www.googleapis.com/youtube/v3/playlistItems", params={"part": "contentDetails", "playlistId": context.uploads_playlist, "maxResults": str(self.settings.max_recent_videos)}).json()
        ids = [item["contentDetails"]["videoId"] for item in playlist.get("items", [])]
        if not ids:
            return []
        items = self._request("GET", "https://www.googleapis.com/youtube/v3/videos", params={"part": "snippet,statistics", "id": ",".join(ids)}).json().get("items", [])
        result = []
        for item in items:
            snippet, stats = item.get("snippet", {}), item.get("statistics", {})
            result.append(RecentVideo(item["id"], snippet.get("title", ""), snippet.get("description", ""), list(snippet.get("tags", [])), datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00")), int(stats.get("viewCount", 0)), int(stats.get("likeCount", 0)), int(stats.get("commentCount", 0)), 0, 0, 0, 0))
        return result

    @staticmethod
    def already_published(recent: list[RecentVideo], daily_tag: str) -> bool:
        return any(daily_tag in video.tags or daily_tag in video.description for video in recent)

    @staticmethod
    def observations(recent: list[RecentVideo]) -> list[Observation]:
        now = datetime.now(timezone.utc)
        values = []
        for video in recent:
            tags = [tag for tag in video.tags if tag.startswith("agfs-")]
            if not tags:
                match = re.search(r"\b(agfs-[a-f0-9]{10})\b", video.description)
                tags = [match.group(1)] if match else []
            if tags:
                values.append(Observation(tags[0], video.views, video.likes, video.comments, video.shares, video.subscribers_gained, video.subscribers_lost, video.average_view_percentage, max((now - video.published_at).total_seconds() / 3600, 0.1)))
        return values

    def upload(self, video_path: Path, thumbnail_path: Path, package: VideoPackage, strategy: Strategy, daily_tag: str) -> str:
        metadata = {"snippet": {"title": package.title, "description": (package.description + f"\n\nExperiment: {strategy.tag}\nRun: {daily_tag}")[:4900], "tags": list(dict.fromkeys(package.tags + [strategy.tag, daily_tag, "altered-content"]))[:20], "categoryId": "28", "defaultLanguage": self.settings.language}, "status": {"privacyStatus": self.settings.privacy_status, "selfDeclaredMadeForKids": False, "containsSyntheticMedia": True}}
        initial = self._request("POST", "https://www.googleapis.com/upload/youtube/v3/videos", params={"uploadType": "resumable", "part": "snippet,status"}, json=metadata, headers={"X-Upload-Content-Length": str(video_path.stat().st_size), "X-Upload-Content-Type": "video/mp4"})
        location = initial.headers["Location"]
        uploaded = self._request("PUT", location, data=video_path.read_bytes(), headers={"Content-Type": "video/mp4", "Content-Length": str(video_path.stat().st_size)})
        video_id = uploaded.json()["id"]
        self._request("POST", "https://www.googleapis.com/upload/youtube/v3/thumbnails/set", params={"videoId": video_id, "uploadType": "media"}, data=thumbnail_path.read_bytes(), headers={"Content-Type": "image/jpeg"})
        return video_id
