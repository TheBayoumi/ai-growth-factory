from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

VERSION = "1.3.1"
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]
HOME = r'''<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>AI Growth Factory</title><style>:root{color-scheme:dark;font:16px system-ui;background:#06101d;color:#edf7ff}body{max-width:920px;margin:45px auto;padding:20px;background:radial-gradient(circle at 10% 0,#103754,#06101d 45%)}h1{font-size:clamp(2.4rem,7vw,4.8rem);line-height:.95}.card{background:#0b1c2d;border:1px solid #24516b;border-radius:18px;padding:21px;margin:15px 0}.pill{background:#12364d;color:#72ddff;padding:7px 11px;border-radius:99px;font-weight:700}input{width:96%;padding:12px;margin:7px 0;background:#071522;color:#fff;border:1px solid #315f78;border-radius:10px}button{padding:12px 16px;border:0;border-radius:10px;background:#55d7ff;font-weight:800}pre{white-space:pre-wrap}</style><span class=pill>MODAL T4 · OPEN-WEIGHT REVIEW</span><h1>AI Growth Factory</h1><div class=card><h2>Control-plane status</h2><pre id=s>Loading…</pre><p>Vercel handles status and YouTube OAuth only. The bounded production worker runs on Modal free credits.</p></div><div class=card><h2>Autonomous voice loop</h2><p>Growth policy → Qwen3-TTS segments → deterministic DSP → Qwen2.5-Omni text review → regenerate rejected segments only → approved narration.</p></div><div class=card><h2>Output verification</h2><p>Narration-synchronized scenes, burned-in captions, codec, duration, sampled-frame and thumbnail gates now run before upload.</p></div><div class=card><h2>Cost guard</h2><p>One daily T4 job, one container, a thirty-minute ceiling and private-first uploads. No OpenAI key is required.</p></div><div class=card><h2>YouTube authorization</h2><p>Redirect URI:</p><pre id=r></pre><input id=i placeholder="OAuth client ID"><input id=k type=password placeholder="OAuth client secret"><button id=b>Authorize YouTube</button></div><script>const o=location.origin,u=o+'/api/oauth';r.textContent=u;fetch('/api/health').then(x=>x.json()).then(x=>s.textContent=JSON.stringify(x,null,2));b.onclick=()=>{const c=i.value.trim(),q=k.value;if(!c||!q)return alert('Enter both values');sessionStorage.setItem('yt_client_id',c);sessionStorage.setItem('yt_client_secret',q);location='https://accounts.google.com/o/oauth2/v2/auth?'+new URLSearchParams({client_id:c,redirect_uri:u,response_type:'code',access_type:'offline',prompt:'consent',include_granted_scopes:'true',scope:''' + json.dumps(" ".join(SCOPES)) + r'''})}</script>'''


def _callback_html(code: str, error: str) -> str:
    return f'''<!doctype html><meta charset=utf-8><title>YouTube authorization</title><style>body{{font:16px system-ui;background:#07111f;color:#edf7ff;max-width:760px;margin:60px auto}}pre{{white-space:pre-wrap;background:#0d2138;padding:18px;border-radius:12px}}</style><h1>YouTube authorization</h1><pre id=o>Completing token exchange…</pre><button id=b hidden>Copy YOUTUBE_OAUTH_JSON</button><script>const code={json.dumps(code)},err={json.dumps(error)},out=o;if(err)out.textContent='Google denied authorization: '+err;else{{const client_id=sessionStorage.getItem('yt_client_id'),client_secret=sessionStorage.getItem('yt_client_secret'),redirect_uri=location.origin+'/api/oauth';if(!client_id||!client_secret)out.textContent='Browser session lost the OAuth client values. Restart from the dashboard.';else fetch('/api/oauth',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{client_id,client_secret,redirect_uri,code}})}}).then(async r=>{{const d=await r.json();if(!r.ok)throw Error(d.error||'Exchange failed');const v=JSON.stringify(d.youtube_oauth_json);out.textContent=v;b.hidden=false;b.onclick=()=>navigator.clipboard.writeText(v);sessionStorage.removeItem('yt_client_secret')}}).catch(e=>out.textContent=e.message)}}}}</script>'''


class handler(BaseHTTPRequestHandler):
    def _route(self) -> str:
        return (parse_qs(urlparse(self.path).query).get("route") or ["home"])[0]

    def _send(self, payload: object, status: int = 200, content_type: str = "application/json") -> None:
        body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = self._route()
        if route == "home":
            return self._send(HOME, content_type="text/html; charset=utf-8")
        if route == "health":
            return self._send({
                "ok": True,
                "service": "ai-growth-factory-control-plane",
                "version": VERSION,
                "architecture": "modal-qwen3-plus-qwen-omni-reviewer",
                "output_verification": "narration-synced-render-plus-post-render-qc",
                "cloud_execution": False,
                "publishing_location": "bounded Modal T4 worker",
            })
        if route == "run":
            return self._send({
                "status": "modal_worker_required",
                "message": "Qwen inference, rendering and publishing execute on the bounded Modal worker.",
            }, 409)
        if route == "oauth":
            query = parse_qs(urlparse(self.path).query)
            code = (query.get("code") or [""])[0]
            error = (query.get("error") or [""])[0]
            if not code and not error:
                return self._send({"scopes": SCOPES, "callback": "/api/oauth"})
            return self._send(_callback_html(code, error), content_type="text/html; charset=utf-8")
        return self._send({"error": "not_found"}, 404)

    def do_POST(self) -> None:
        if self._route() != "oauth":
            return self._send({"error": "not_found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            required = {"client_id", "client_secret", "redirect_uri", "code"}
            missing = sorted(required - set(payload))
            if missing:
                return self._send({"error": "Missing: " + ", ".join(missing)}, 400)
            data = urlencode({
                "client_id": payload["client_id"],
                "client_secret": payload["client_secret"],
                "redirect_uri": payload["redirect_uri"],
                "code": payload["code"],
                "grant_type": "authorization_code",
            }).encode()
            request = Request("https://oauth2.googleapis.com/token", data=data, method="POST")
            request.add_header("Content-Type", "application/x-www-form-urlencoded")
            with urlopen(request, timeout=30) as response:
                token = json.loads(response.read())
            refresh_token = token.get("refresh_token")
            if not refresh_token:
                return self._send({"error": "Google returned no refresh token. Revoke the grant and retry with prompt=consent."}, 400)
            return self._send({"youtube_oauth_json": {
                "client_id": payload["client_id"],
                "client_secret": payload["client_secret"],
                "refresh_token": refresh_token,
            }})
        except HTTPError as exc:
            return self._send({"error": exc.read().decode(errors="replace")[:1500]}, exc.code)
        except Exception as exc:
            return self._send({"error": str(exc)}, 500)
