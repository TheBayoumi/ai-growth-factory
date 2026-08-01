import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, call, patch

from factory.config import Settings
from factory.models import VideoPackage
from factory.policy import Strategy
from factory.youtube import ChannelContext, RecentVideo, YouTubeClient, YouTubeError


class YouTubeClientTests(unittest.TestCase):
    @staticmethod
    def settings() -> Settings:
        oauth = {
            "client_id": "client",
            "client_secret": "secret",
            "refresh_token": "refresh",
        }
        with patch.dict(
            "os.environ",
            {"YOUTUBE_OAUTH_JSON": json.dumps(oauth)},
            clear=True,
        ):
            return Settings.from_env()

    @classmethod
    def client_without_refresh(cls) -> YouTubeClient:
        client = object.__new__(YouTubeClient)
        client.settings = cls.settings()
        client.oauth = client.settings.youtube_oauth
        client.access_token = "access"
        client.headers = {"Authorization": "Bearer access"}
        return client

    @staticmethod
    def package() -> VideoPackage:
        return VideoPackage(
            topic="AI update",
            narration="Narration",
            title="AI update title",
            description="Description",
            tags=["AI", "qwen", "AI"],
            thumbnail_text="AI update",
            top_comment="What do you think?",
            scenes=[],
            source_urls=["https://example.com/source"],
            source_publishers=["Publisher"],
        )

    @staticmethod
    def strategy() -> Strategy:
        return Strategy("practical", "balanced", "dashboard", "55-62", "subscribe")

    def test_constructor_requires_oauth_configuration(self):
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings.from_env()
        with self.assertRaisesRegex(YouTubeError, "YOUTUBE_OAUTH_JSON"):
            YouTubeClient(settings)

    def test_constructor_refreshes_access_token_and_sets_header(self):
        response = Mock(status_code=200, text="")
        response.json.return_value = {"access_token": "fresh-token"}
        with patch("factory.youtube.requests.post", return_value=response) as post:
            client = YouTubeClient(self.settings())

        self.assertEqual(client.headers, {"Authorization": "Bearer fresh-token"})
        self.assertEqual(post.call_args.kwargs["timeout"], 30)
        self.assertEqual(post.call_args.kwargs["data"]["grant_type"], "refresh_token")

    def test_refresh_rejects_http_error_and_missing_token(self):
        client = self.client_without_refresh()
        failed = Mock(status_code=401, text="invalid credentials")
        missing = Mock(status_code=200, text="")
        missing.json.return_value = {}

        with patch("factory.youtube.requests.post", return_value=failed):
            with self.assertRaisesRegex(YouTubeError, "OAuth refresh failed"):
                client._refresh_access_token()
        with patch("factory.youtube.requests.post", return_value=missing):
            with self.assertRaisesRegex(YouTubeError, "no access token"):
                client._refresh_access_token()

    def test_get_returns_json_and_wraps_http_errors(self):
        client = self.client_without_refresh()
        good = Mock(status_code=200, text="")
        good.json.return_value = {"items": [1]}
        bad = Mock(status_code=500, text="server failure")

        with patch("factory.youtube.requests.get", return_value=good) as get:
            self.assertEqual(client._get("https://api", part="snippet"), {"items": [1]})
        self.assertEqual(get.call_args.kwargs["headers"], client.headers)
        self.assertEqual(get.call_args.kwargs["params"], {"part": "snippet"})

        with patch("factory.youtube.requests.get", return_value=bad):
            with self.assertRaisesRegex(YouTubeError, "GET https://api failed"):
                client._get("https://api")

    def test_channel_context_parses_statistics_and_requires_channel(self):
        client = self.client_without_refresh()
        payload = {
            "items": [
                {
                    "id": "channel-id",
                    "statistics": {"subscriberCount": "12", "viewCount": "345"},
                    "contentDetails": {"relatedPlaylists": {"uploads": "playlist-id"}},
                }
            ]
        }
        with patch.object(client, "_get", return_value=payload):
            context = client.channel_context()
        self.assertEqual(context, ChannelContext("channel-id", "playlist-id", 12, 345))

        with patch.object(client, "_get", return_value={"items": []}):
            with self.assertRaisesRegex(YouTubeError, "no YouTube channel"):
                client.channel_context()

    def test_recent_videos_returns_empty_and_sorts_parsed_results(self):
        client = self.client_without_refresh()
        context = ChannelContext("channel", "uploads", 0, 0)
        with patch.object(client, "_get", return_value={"items": []}):
            self.assertEqual(client.recent_videos(context), [])

        playlist = {
            "items": [
                {"contentDetails": {"videoId": "later"}},
                {"contentDetails": {"videoId": "earlier"}},
            ]
        }
        videos = {
            "items": [
                {
                    "id": "later",
                    "snippet": {
                        "title": "Later",
                        "publishedAt": "2026-07-20T12:00:00Z",
                        "tags": ["agfs-later"],
                    },
                    "statistics": {"viewCount": "20", "likeCount": "3", "commentCount": "1"},
                },
                {
                    "id": "earlier",
                    "snippet": {
                        "title": "Earlier",
                        "publishedAt": "2026-07-19T12:00:00Z",
                    },
                    "statistics": {},
                },
            ]
        }
        with patch.object(client, "_get", side_effect=[playlist, videos]) as get:
            result = client.recent_videos(context)

        self.assertEqual([video.video_id for video in result], ["earlier", "later"])
        self.assertEqual(result[0].views, 0)
        self.assertEqual(result[1].tags, ["agfs-later"])
        self.assertEqual(get.call_args_list[1].kwargs["id"], "later,earlier")

    def test_analytics_handles_immature_error_empty_and_success(self):
        client = self.client_without_refresh()
        immature = RecentVideo(
            "new",
            "New",
            datetime.now(timezone.utc),
            [],
            1,
            0,
            0,
        )
        self.assertEqual(client._analytics(immature), {})

        mature = RecentVideo(
            "old",
            "Old",
            datetime.now(timezone.utc) - timedelta(days=10),
            [],
            10,
            1,
            0,
        )
        failed = Mock(status_code=500)
        empty = Mock(status_code=200)
        empty.json.return_value = {"rows": []}
        good = Mock(status_code=200)
        good.json.return_value = {
            "columnHeaders": [{"name": "views"}, {"name": "averageViewPercentage"}],
            "rows": [[42, 71.5]],
        }

        with patch("factory.youtube.requests.get", return_value=failed):
            self.assertEqual(client._analytics(mature), {})
        with patch("factory.youtube.requests.get", return_value=empty):
            self.assertEqual(client._analytics(mature), {})
        with patch("factory.youtube.requests.get", return_value=good):
            self.assertEqual(
                client._analytics(mature),
                {"views": 42.0, "averageViewPercentage": 71.5},
            )

    def test_observations_skip_untagged_and_use_analytics_with_fallbacks(self):
        client = self.client_without_refresh()
        published = datetime.now(timezone.utc) - timedelta(hours=30)
        videos = [
            RecentVideo("skip", "No tag", published, ["AI"], 10, 2, 1),
            RecentVideo("keep", "Tagged", published, ["AI", "agfs-123"], 100, 5, 2),
        ]
        with patch.object(
            client,
            "_analytics",
            return_value={
                "views": 120,
                "shares": 4,
                "subscribersGained": 3,
                "subscribersLost": 1,
                "averageViewPercentage": 68.5,
            },
        ) as analytics:
            result = client.observations(videos)

        self.assertEqual(len(result), 1)
        observation = result[0]
        self.assertEqual(observation.strategy_tag, "agfs-123")
        self.assertEqual(observation.views, 120)
        self.assertEqual(observation.likes, 5)
        self.assertEqual(observation.comments, 2)
        self.assertEqual(observation.shares, 4)
        self.assertEqual(observation.subscribers_gained, 3)
        self.assertEqual(observation.subscribers_lost, 1)
        self.assertGreaterEqual(observation.age_hours, 29.9)
        analytics.assert_called_once_with(videos[1])
        self.assertTrue(client.already_published(videos, "agfs-123"))
        self.assertFalse(client.already_published(videos, "daily-unknown"))

    def test_upload_retries_transient_failure_then_sets_thumbnail_and_comment(self):
        client = self.client_without_refresh()
        init = Mock(status_code=200, text="", headers={"Location": "https://upload"})
        transient = Mock(status_code=503, text="temporary")
        success = Mock(status_code=200, text="")
        success.json.return_value = {"id": "video-id"}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            thumbnail = root / "thumbnail.jpg"
            video.write_bytes(b"video-bytes")
            thumbnail.write_bytes(b"jpeg")
            with patch("factory.youtube.requests.post", return_value=init) as post, patch(
                "factory.youtube.requests.put", side_effect=[transient, success]
            ) as put, patch("factory.youtube.time.sleep") as sleep, patch.object(
                client, "_set_thumbnail"
            ) as set_thumbnail, patch.object(client, "_post_comment") as post_comment:
                video_id = client.upload(
                    video_path=video,
                    thumbnail_path=thumbnail,
                    package=self.package(),
                    strategy=self.strategy(),
                    daily_tag="daily-20260801",
                )

        self.assertEqual(video_id, "video-id")
        self.assertEqual(put.call_count, 2)
        sleep.assert_called_once_with(1)
        set_thumbnail.assert_called_once_with("video-id", thumbnail)
        post_comment.assert_called_once_with("video-id", "What do you think?")
        metadata = json.loads(post.call_args.kwargs["data"])
        self.assertEqual(metadata["status"]["privacyStatus"], "private")
        self.assertTrue(metadata["status"]["containsSyntheticMedia"])
        self.assertEqual(len(metadata["snippet"]["tags"]), len(set(metadata["snippet"]["tags"])))

    def test_upload_fails_on_missing_location_and_non_retryable_put(self):
        client = self.client_without_refresh()
        missing_location = Mock(status_code=200, text="missing", headers={})
        permanent = Mock(status_code=400, text="bad request")
        init = Mock(status_code=200, text="", headers={"Location": "https://upload"})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            thumbnail = root / "thumbnail.jpg"
            video.write_bytes(b"video")
            thumbnail.write_bytes(b"jpeg")
            with patch("factory.youtube.requests.post", return_value=missing_location):
                with self.assertRaisesRegex(YouTubeError, "Could not start upload"):
                    client.upload(
                        video_path=video,
                        thumbnail_path=thumbnail,
                        package=self.package(),
                        strategy=self.strategy(),
                        daily_tag="daily",
                    )
            with patch("factory.youtube.requests.post", return_value=init), patch(
                "factory.youtube.requests.put", return_value=permanent
            ) as put, patch("factory.youtube.time.sleep") as sleep:
                with self.assertRaisesRegex(YouTubeError, "bad request"):
                    client.upload(
                        video_path=video,
                        thumbnail_path=thumbnail,
                        package=self.package(),
                        strategy=self.strategy(),
                        daily_tag="daily",
                    )
        put.assert_called_once()
        sleep.assert_not_called()

    def test_thumbnail_and_comment_failure_contracts(self):
        client = self.client_without_refresh()
        failed = Mock(status_code=500, text="thumbnail failed")
        forbidden = Mock(status_code=403, text="comments disabled")
        comment_failed = Mock(status_code=500, text="comment failed")

        with tempfile.TemporaryDirectory() as temporary:
            thumbnail = Path(temporary) / "thumbnail.jpg"
            thumbnail.write_bytes(b"jpeg")
            with patch("factory.youtube.requests.post", return_value=failed):
                with self.assertRaisesRegex(YouTubeError, "Thumbnail upload failed"):
                    client._set_thumbnail("video", thumbnail)

        with patch("factory.youtube.requests.post") as post:
            client._post_comment("video", "   ")
            post.assert_not_called()
        with patch("factory.youtube.requests.post", return_value=forbidden):
            client._post_comment("video", "Comment")
        with patch("factory.youtube.requests.post", return_value=comment_failed):
            with self.assertRaisesRegex(YouTubeError, "Comment publishing failed"):
                client._post_comment("video", "Comment")


if __name__ == "__main__":
    unittest.main()
