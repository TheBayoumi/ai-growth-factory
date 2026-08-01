from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from .config import Settings
from .models import VideoPackage
from .policy import Observation, Strategy


API = "https://www.googleapis.com/youtube/v3"
UPLOAD_API = "https://www.googleapis.com/upload/youtube/v3"
ANALYTICS_API = "https://youtubeanalytics.googleapis.com/v2/reports"
TOKEN_API = "https://oauth2.googleapis.com/token"


class YouTubeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChannelContext:
    channel_id: str
    uploads_playlist: str
    subscribers: int
    total_views: int


@dataclass(frozen=True)
class RecentVideo:
    video_id: str
    title: str
    published_at: datetime
    tags: list[str]
    views: int
    likes: int
    comments: int


class YouTubeClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.youtube_oauth:
            raise YouTubeError("YOUTUBE_OAUTH_JSON is missing")
        self.settings = settings
        self.oauth = settings.youtube_oauth
        self.access_token = self._refresh_access_token()
        self.headers = {"Authorization": f"Bearer {self.access_token}"}

    def _refresh_access_token(self) -> str:
        response = requests.post(
            TOKEN_API,
            data={
                "client_id": self.oauth["client_id"],
                "client_secret": self.oauth["client_secret"],
                "refresh_token": self.oauth["refresh_token"],
                "grant_type": "refresh_token",
            },
            timeout=30,
        )
        if response.status_code >= 400:
            raise YouTubeError(f"OAuth refresh failed: {response.text[:1000]}")
        token = response.json().get("access_token")
        if not token:
            raise YouTubeError("OAuth refresh returned no access token")
        return str(token)

    def _get(self, url: str, **params: Any) -> dict[str, Any]:
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        if response.status_code >= 400:
            raise YouTubeError(f"GET {url} failed: {response.text[:1500]}")
        return response.json()

    def channel_context(self) -> ChannelContext:
        data = self._get(
            f"{API}/channels",
            part="snippet,statistics,contentDetails",
            mine="true",
            maxResults=1,
        )
        items = data.get("items") or []
        if not items:
            raise YouTubeError("Authorized account has no YouTube channel")
        channel = items[0]
        stats = channel.get("statistics") or {}
        return ChannelContext(
            channel_id=str(channel["id"]),
            uploads_playlist=str(channel["contentDetails"]["relatedPlaylists"]["uploads"]),
            subscribers=int(stats.get("subscriberCount", 0)),
            total_views=int(stats.get("viewCount", 0)),
        )

    def recent_videos(self, context: ChannelContext) -> list[RecentVideo]:
        playlist = self._get(
            f"{API}/playlistItems",
            part="contentDetails",
            playlistId=context.uploads_playlist,
            maxResults=self.settings.max_recent_videos,
        )
        ids = [str(item["contentDetails"]["videoId"]) for item in playlist.get("items") or []]
        if not ids:
            return []
        data = self._get(
            f"{API}/videos",
            part="snippet,statistics,status",
            id=",".join(ids),
            maxResults=len(ids),
        )
        result: list[RecentVideo] = []
        for item in data.get("items") or []:
            snippet = item.get("snippet") or {}
            stats = item.get("statistics") or {}
            published = datetime.fromisoformat(str(snippet["publishedAt"]).replace("Z", "+00:00"))
            result.append(
                RecentVideo(
                    video_id=str(item["id"]),
                    title=str(snippet.get("title", "")),
                    published_at=published,
                    tags=[str(tag) for tag in snippet.get("tags") or []],
                    views=int(stats.get("viewCount", 0)),
                    likes=int(stats.get("likeCount", 0)),
                    comments=int(stats.get("commentCount", 0)),
                )
            )
        return sorted(result, key=lambda video: video.published_at)

    def _analytics(self, video: RecentVideo) -> dict[str, float]:
        today = datetime.now(timezone.utc).date()
        end = today - timedelta(days=2)
        start = video.published_at.date()
        if end < start:
            return {}
        metrics = (
            "views,likes,comments,shares,averageViewPercentage,"
            "subscribersGained,subscribersLost"
        )
        response = requests.get(
            ANALYTICS_API,
            headers=self.headers,
            params={
                "ids": "channel==MINE",
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "metrics": metrics,
                "filters": f"video=={video.video_id}",
            },
            timeout=30,
        )
        if response.status_code >= 400:
            return {}
        data = response.json()
        rows = data.get("rows") or []
        if not rows:
            return {}
        headers = [column["name"] for column in data.get("columnHeaders") or []]
        return {str(name): float(value) for name, value in zip(headers, rows[0], strict=False)}

    def observations(self, videos: list[RecentVideo]) -> list[Observation]:
        now = datetime.now(timezone.utc)
        observations: list[Observation] = []
        for video in videos:
            strategy_tag = next((tag for tag in video.tags if tag.startswith("agfs-")), "")
            if not strategy_tag:
                continue
            analytics = self._analytics(video)
            observations.append(
                Observation(
                    strategy_tag=strategy_tag,
                    views=int(analytics.get("views", video.views)),
                    likes=int(analytics.get("likes", video.likes)),
                    comments=int(analytics.get("comments", video.comments)),
                    shares=int(analytics.get("shares", 0)),
                    subscribers_gained=int(analytics.get("subscribersGained", 0)),
                    subscribers_lost=int(analytics.get("subscribersLost", 0)),
                    average_view_percentage=float(analytics.get("averageViewPercentage", 0.0)),
                    age_hours=max((now - video.published_at).total_seconds() / 3600.0, 0.0),
                )
            )
        return observations

    @staticmethod
    def already_published(videos: list[RecentVideo], daily_tag: str) -> bool:
        return any(daily_tag in video.tags for video in videos)

    def upload(
        self,
        *,
        video_path: Path,
        thumbnail_path: Path,
        package: VideoPackage,
        strategy: Strategy,
        daily_tag: str,
    ) -> str:
        tags = list(dict.fromkeys([*package.tags, "AI", "AI news", strategy.tag, daily_tag]))
        metadata = {
            "snippet": {
                "title": package.title,
                "description": package.description,
                "tags": tags[:20],
                "categoryId": "28",
                "defaultLanguage": self.settings.language,
            },
            "status": {
                "privacyStatus": self.settings.privacy_status,
                "selfDeclaredMadeForKids": False,
                "containsSyntheticMedia": True,
                "embeddable": True,
                "publicStatsViewable": True,
            },
        }
        size = video_path.stat().st_size
        init = requests.post(
            f"{UPLOAD_API}/videos",
            headers={
                **self.headers,
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Length": str(size),
                "X-Upload-Content-Type": "video/mp4",
            },
            params={"uploadType": "resumable", "part": "snippet,status"},
            data=json.dumps(metadata),
            timeout=30,
        )
        if init.status_code >= 400 or not init.headers.get("Location"):
            raise YouTubeError(f"Could not start upload: {init.text[:1500]}")
        upload_url = init.headers["Location"]
        last_error = ""
        for attempt in range(4):
            with video_path.open("rb") as handle:
                upload = requests.put(
                    upload_url,
                    headers={
                        **self.headers,
                        "Content-Type": "video/mp4",
                        "Content-Length": str(size),
                    },
                    data=handle,
                    timeout=240,
                )
            if upload.status_code in {200, 201}:
                video_id = str(upload.json()["id"])
                self._set_thumbnail(video_id, thumbnail_path)
                self._post_comment(video_id, package.top_comment)
                return video_id
            last_error = upload.text[:1500]
            if upload.status_code not in {500, 502, 503, 504}:
                break
            time.sleep(2**attempt)
        raise YouTubeError(f"Upload failed: {last_error}")

    def _set_thumbnail(self, video_id: str, thumbnail: Path) -> None:
        response = requests.post(
            f"{UPLOAD_API}/thumbnails/set",
            headers={**self.headers, "Content-Type": "image/jpeg"},
            params={"videoId": video_id, "uploadType": "media"},
            data=thumbnail.read_bytes(),
            timeout=60,
        )
        if response.status_code >= 400:
            raise YouTubeError(f"Thumbnail upload failed: {response.text[:1000]}")

    def _post_comment(self, video_id: str, text: str) -> None:
        if not text.strip():
            return
        response = requests.post(
            f"{API}/commentThreads",
            headers={**self.headers, "Content-Type": "application/json"},
            params={"part": "snippet"},
            json={
                "snippet": {
                    "videoId": video_id,
                    "topLevelComment": {"snippet": {"textOriginal": text[:9000]}},
                }
            },
            timeout=30,
        )
        if response.status_code >= 400 and response.status_code not in {403}:
            raise YouTubeError(f"Comment publishing failed: {response.text[:1000]}")
