from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlencode, urlparse

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
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        code = (query.get("code") or [""])[0]
        error = (query.get("error") or [""])[0]
        if not code and not error:
            return self._send_json({"scopes": SCOPES, "callback": "/api/oauth"})
        safe_code = json.dumps(code)
        safe_error = json.dumps(error)
        html = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>YouTube authorization</title><style>body{{font-family:system-ui;background:#07111f;color:#edf7ff;max-width:760px;margin:60px auto;padding:20px}}pre{{white-space:pre-wrap;background:#0d2138;padding:18px;border-radius:12px}}button{{padding:12px 18px}}</style></head><body><h1>YouTube authorization</h1><pre id='out'>Completing secure token exchange…</pre><button id='copy' hidden>Copy YOUTUBE_OAUTH_JSON</button><script>
const code={safe_code}; const oauthError={safe_error}; const out=document.getElementById('out');
if(oauthError){{out.textContent='Google denied authorization: '+oauthError;}}
else{{
 const client_id=sessionStorage.getItem('yt_client_id');
 const client_secret=sessionStorage.getItem('yt_client_secret');
 const redirect_uri=location.origin+'/api/oauth';
 if(!client_id||!client_secret){{out.textContent='The browser session lost the OAuth client values. Return to the dashboard and restart authorization.';}}
 else fetch('/api/oauth',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{client_id,client_secret,redirect_uri,code}})}}).then(async r=>{{const data=await r.json();if(!r.ok)throw new Error(data.error||'Exchange failed');const value=JSON.stringify(data.youtube_oauth_json);out.textContent=value;const b=document.getElementById('copy');b.hidden=false;b.onclick=()=>navigator.clipboard.writeText(value);sessionStorage.removeItem('yt_client_secret');}}).catch(e=>out.textContent=e.message);
}}
</script></body></html>"""
        body = html.encode()
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
                return self._send_json(
                    {"error": "Google returned no refresh token. Revoke the app grant and retry with prompt=consent."},
                    400,
                )
            return self._send_json(
                {
                    "youtube_oauth_json": {
                        "client_id": payload["client_id"],
                        "client_secret": payload["client_secret"],
                        "refresh_token": refresh_token,
                    }
                }
            )
        except Exception as exc:
            return self._send_json({"error": str(exc)}, 500)
