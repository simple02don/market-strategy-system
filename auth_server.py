"""报告登录服务（仅本机监听，nginx /strategy/ 反代；沿用 JCKX 密码体系）。"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
REPORT_DIR = Path(os.environ.get("MARKET_STRATEGY_REPORT", ROOT / "reports" / "html"))
BIND_HOST = os.environ.get("JCKX_BIND_HOST", "127.0.0.1")
PORT = int(os.environ.get("JCKX_AUTH_PORT", "8082"))
PASSWORD = os.environ.get("JCKX_PASSWORD", "")
TOKEN = os.environ.get("JCKX_TOKEN", "")
PUBLIC = os.environ.get("JCKX_PUBLIC_REPORTS", "0").strip().lower() in {"1", "true", "yes"}
BASE_PATH = urlparse(os.environ.get("JCKX_REPORT_BASE_URL", "http://10.66.0.1/strategy")).path.rstrip("/")

COOKIE_NAME = "jckx_report_session"
SESSION_SECONDS = max(3600, int(os.environ.get("JCKX_SESSION_MAX_AGE", str(10 * 365 * 24 * 3600))))


def _signature(expiry: int) -> str:
    return hmac.new(
        TOKEN.encode(),
        f"{PASSWORD}:{expiry}".encode(),
        hashlib.sha256,
    ).hexdigest()


def _valid_session(cookie_value: str) -> bool:
    try:
        expiry, signature = cookie_value.split(":", 1)
        expiry = int(expiry)
        if time.time() > expiry:
            return False
        return hmac.compare_digest(signature, _signature(expiry))
    except Exception:  # noqa: BLE001
        return False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A003
        return

    def _send(self, status: int, body: bytes, content_type: str = "text/html; charset=utf-8", headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self) -> bool:
        if PUBLIC:
            return True
        raw = self.headers.get("Cookie", "")
        cookie = SimpleCookie()
        try:
            cookie.load(raw)
            return _valid_session(cookie.get(COOKIE_NAME).value)
        except Exception:  # noqa: BLE001
            return False

    def _login_page(self) -> bytes:
        return (
            "<!doctype html><html><body style='background:#0f1420;color:#dfe6f2;"
            "font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:90vh'>"
            f"<form method='post' action='{BASE_PATH}/login'><h2>市场策略报告</h2>"
            "<input type='password' name='password' placeholder='密码' autofocus>"
            "<button>登录</button></form></body></html>"
        ).encode()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {BASE_PATH, f"{BASE_PATH}/", "/", "/index.html"}:
            if not self._auth_ok():
                self._send(401, self._login_page())
                return
            files = sorted(REPORT_DIR.glob("market_strategy_*.html"))
            if not files:
                self._send(200, "<html><body>暂无报告</body></html>".encode())
                return
            links = "".join(
                f"<li><a href='{BASE_PATH}/{f.name}'>{f.name}</a></li>"
                for f in files[-30:]
            )
            self._send(200, f"<html><body><ul>{links}</ul></body></html>".encode())
            return
        if path.startswith("/market_strategy_"):
            if not self._auth_ok():
                self._send(401, self._login_page())
                return
            target = (REPORT_DIR / path.lstrip("/")).resolve()
            if not str(target).startswith(str(REPORT_DIR.resolve())) or not target.exists():
                self._send(404, b"not found")
                return
            self._send(200, target.read_bytes(), "text/html; charset=utf-8")
            return
        self._send(404, b"not found")

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/login":
            self._send(404, b"not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode(errors="ignore")
        password = parse_qs(body).get("password", [""])[0]
        if not PASSWORD or not hmac.compare_digest(password, PASSWORD):
            self._send(401, "<html><body>密码错误</body></html>".encode())
            return
        expiry = int(time.time()) + SESSION_SECONDS
        cookie_value = f"{expiry}:{_signature(expiry)}"
        secure = "https" in self.headers.get("X-Forwarded-Proto", "").lower()
        self._send(
            302,
            b"",
            headers={
                "Location": f"{BASE_PATH}/",
                "Set-Cookie": (
                    f"{COOKIE_NAME}={cookie_value}; Path=/; HttpOnly; "
                    f"Max-Age={SESSION_SECONDS}; SameSite=Lax"
                    + ("; Secure" if secure else "")
                ),
            },
        )


if __name__ == "__main__":
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((BIND_HOST, PORT), Handler)
    print(f"Serving market-strategy reports from {REPORT_DIR} on {BIND_HOST}:{PORT}")
    server.serve_forever()
