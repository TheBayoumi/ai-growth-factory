from __future__ import annotations

import hmac
import json
import os
from http.server import BaseHTTPRequestHandler

from factory.config import Settings


class handler(BaseHTTPRequestHandler):
    def _authorized(self, settings: Settings) -> bool:
        expected = settings.cron_secret or ""
        auth = self.headers.get("Authorization", "")
        factory = self.headers.get("X-Factory-Secret", "")
        supplied = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else factory
        return bool(expected) and hmac.compare_digest(supplied, expected)

    def _run(self) -> None:
        try:
            settings = Settings.from_env()
            if not self._authorized(settings):
                payload, status = {"status": "unauthorized"}, 401
            elif os.getenv("VERCEL"):
                payload, status = {
                    "status": "local_worker_required",
                    "message": "Qwen inference, rendering and publishing run on the Modal worker, not Vercel.",
                }, 409
            else:
                from factory.pipeline import run_factory

                payload, status = run_factory(settings), 200
        except Exception as exc:
            payload, status = {"status": "failed", "error": str(exc)}, 500
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _run
    do_POST = _run
