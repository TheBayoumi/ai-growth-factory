from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import requests

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


class handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        code = (query.get("code") or [""])[0]
        error = (query.get("error") or [""])[0]
        if not code and not error:
            return self._send_json({"scopes": SCOPES, "callback": "/api/oauth"})
        body = (
            "<!doctype html><meta charset='utf-8'><title>YouTube authorization</title>"
            "<h1>YouTube authorization received</h1>"
            "<p>Return to the secure setup flow to complete the token exchange.</p>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            required = {"client_id", "client_secret", "redirect_uri", "code"}
            missing = sorted(required - set(payload))
            if missing:
                return self._send_json({"error": "Missing: " + ", ".join(missing)}, 400)
            response = requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": payload["client_id"],
                    "client_secret": payload["client_secret"],
                    "redirect_uri": payload["redirect_uri"],
                    "code": payload["code"],
                    "grant_type": "authorization_code",
                },
                timeout=30,
            )
            if response.status_code >= 400:
                return self._send_json({"error": response.text[:1500]}, response.status_code)
            token = response.json()
            refresh_token = token.get("refresh_token")
            if not refresh_token:
                return self._send_json({"error": "Google returned no refresh token."}, 400)
            return self._send_json({
                "youtube_oauth_json": {
                    "client_id": payload["client_id"],
                    "client_secret": payload["client_secret"],
                    "refresh_token": refresh_token,
                }
            })
        except Exception as exc:
            return self._send_json({"error": str(exc)}, 500)
