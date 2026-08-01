from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler

from factory import __version__
from factory.config import Settings


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        try:
            settings = Settings.from_env()
            payload = {
                "ok": True,
                "service": "ai-growth-factory",
                "version": __version__,
                "setup": settings.setup_status,
                "architecture": "modal-qwen3-plus-qwen-omni-reviewer",
                "generation_api_cost": False,
                "reviewer_api_cost": settings.reviewer_required and settings.reviewer_backend == "openai",
            }
            status = 200
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
            status = 500
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
