"""企业微信机器人推送（与 JCKX 共用 webhook）。"""

from __future__ import annotations

import time
from typing import Any

import requests

from .. import config


class WeComPusher:
    def __init__(self, webhook: str | None = None):
        self.webhook = webhook or config.env_str("WECOM_WEBHOOK")
        self.retry = config.env_int("TUSHARE_RETRY", 3)

    def send_markdown(self, content: str) -> dict[str, Any]:
        if not self.webhook:
            return {"ok": False, "error": "webhook_missing"}
        payload = {"msgtype": "markdown", "markdown": {"content": content}}
        last_error = ""
        for attempt in range(self.retry):
            try:
                resp = requests.post(self.webhook, json=payload, timeout=15)
                body = resp.json()
                if body.get("errcode") == 0:
                    return {"ok": True}
                last_error = f"{body.get('errcode')}:{body.get('errmsg')}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            time.sleep(2 * (attempt + 1))
        return {"ok": False, "error": last_error}
