"""免费新闻/政策/公告源：中国政府网、部委列表、财联社（Tushare）、巨潮公告。"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import date, datetime, timedelta
from typing import Any

import requests

from .. import config
from ..timeutil import CST
from .tushare_provider import TushareProvider

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:24]


def _clean(text: Any) -> str:
    value = str(text or "")
    return re.sub(r"\s+", " ", value.replace("<em>", "").replace("</em>", "")).strip()


def _dedup_key(source: str, title: str) -> str:
    """跨源标题去重键；source 参数仅为向后兼容保留。"""
    del source
    normalized = re.sub(r"[\W_]+", "", _clean(title).lower(), flags=re.UNICODE)
    return _hash(normalized)


def _epoch_to_beijing(millis: Any) -> str:
    """巨潮公告 epoch（毫秒，北京时间）→ 'YYYY-MM-DD HH:MM:SS'。

    显式用 Asia/Shanghai 解析，避免服务器本地时区（如 UTC）导致
    PIT 过滤放行截止后公告。
    """
    return datetime.fromtimestamp(float(millis) / 1000, tz=CST).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _extract_publish_time(fragment: str) -> str:
    """从列表项附近提取可审计日期；无法确认时保持空值并由 PIT 过滤剔除。"""
    patterns = (
        r"(?P<y>20\d{2})[-/.年](?P<m>\d{1,2})[-/.月](?P<d>\d{1,2})日?",
        r"(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})",
    )
    for pattern in patterns:
        match = re.search(pattern, fragment)
        if not match:
            continue
        try:
            value = datetime(
                int(match.group("y")),
                int(match.group("m")),
                int(match.group("d")),
            )
        except ValueError:
            continue
        return value.strftime("%Y-%m-%d 00:00:00")
    return ""


class NewsCollector:
    def __init__(self, provider: TushareProvider | None = None):
        self.provider = provider
        self.timeout = config.env_int("NLP_FETCH_TIMEOUT_SEC", 15)
        self.session = requests.Session()

    def govcn_policy(self, limit: int = 50) -> list[dict]:
        resp = self.session.get(
            "https://www.gov.cn/zhengce/zuixin/ZUIXINZHENGCE.json",
            headers={**UA, "Referer": "https://www.gov.cn/zhengce/zuixin/"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        rows = resp.json()
        out = []
        for row in (rows if isinstance(rows, list) else [])[:limit]:
            title = _clean(row.get("TITLE"))
            url = str(row.get("URL") or "")
            if not title or not url:
                continue
            out.append(
                {
                    "source": "govcn_policy",
                    "source_id": _hash(url),
                    "title": title,
                    "summary": "",
                    "url": url,
                    "category": "国家政策/国务院",
                    "publish_time": _clean(row.get("DOCRELPUBTIME")),
                    "tier": 1,
                    "dedup_key": _dedup_key("govcn_policy", title),
                }
            )
        return out

    def official_list(self, source: str, page_url: str, category: str, pattern: str) -> list[dict]:
        resp = self.session.get(page_url, headers=UA, timeout=self.timeout)
        resp.raise_for_status()
        text = resp.text
        out = []
        anchor_pattern = re.compile(
            r'<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            re.S | re.I,
        )
        for match in anchor_pattern.finditer(text):
            href = match.group("href")
            title = match.group("title")
            href = href.strip()
            title = _clean(re.sub(r"<[^>]+>", "", title))
            if not re.search(pattern, href) or not title:
                continue
            if not href.startswith("http"):
                from urllib.parse import urljoin

                href = urljoin(page_url, href)
            nearby = text[max(0, match.start() - 100): min(len(text), match.end() + 180)]
            out.append(
                {
                    "source": source,
                    "source_id": _hash(href),
                    "title": title,
                    "summary": "",
                    "url": href,
                    "category": category,
                    "publish_time": _extract_publish_time(nearby),
                    "tier": 1,
                    "dedup_key": _dedup_key(source, title),
                }
            )
            if len(out) >= 50:
                break
        return out

    def official_policies(self) -> list[dict]:
        items: list[dict] = []
        items.extend(self.govcn_policy())
        items.extend(
            self.official_list(
                "ndrc_policy",
                "https://www.ndrc.gov.cn/xwdt/xwfb/",
                "国家政策/发改委",
                r"/xwdt/xwfb/.*\.html$",
            )
        )
        items.extend(
            self.official_list(
                "miit_policy",
                "https://www.miit.gov.cn/RRSdy/",
                "产业政策/工信部",
                r"/zwgk/.*\.html$",
            )
        )
        items.extend(
            self.official_list(
                "mof_policy",
                "https://www.mof.gov.cn/zhengwuxinxi/zhengcefabu/",
                "国家政策/财政部",
                r"/zhengcefabu/.*\.(?:htm|html)$",
            )
        )
        items.extend(
            self.official_list(
                "csrc_policy",
                "https://www.csrc.gov.cn/csrc/c100039/common_list.shtml",
                "资本市场政策/证监会",
                r"/csrc/c100028/.*/content\.shtml$",
            )
        )
        return items

    def tushare_major_news(self, start_dt: str, end_dt: str) -> list[dict]:
        if not self.provider:
            return []
        rows = self.provider.major_news(start_dt, end_dt)
        out = []
        for row in rows:
            title = _clean(row.get("title"))
            if not title:
                continue
            content = _clean(row.get("content"))[:1200]
            out.append(
                {
                    "source": "cls_telegraph",
                    "source_id": _hash(title + content[:200]),
                    "title": title,
                    "summary": content,
                    "url": str(row.get("url") or ""),
                    "category": "新闻/财联社",
                    "publish_time": str(row.get("pub_time") or ""),
                    "tier": 2,
                    "dedup_key": _dedup_key("cls_telegraph", title),
                }
            )
        return out[:60]

    def cninfo_announcements(self, start_date: date, end_date: date, max_pages: int = 10) -> list[dict]:
        out = []
        seen = set()
        for column, plate in (("szse", "sz"), ("sse", "sh")):
            for page in range(1, max_pages + 1):
                payload = {
                    "pageNum": page,
                    "pageSize": 30,
                    "column": column,
                    "tabName": "fulltext",
                    "plate": plate,
                    "stock": "",
                    "searchkey": "",
                    "secid": "",
                    "category": "",
                    "trade": "",
                    "seDate": f"{start_date:%Y-%m-%d}~{end_date:%Y-%m-%d}",
                    "sortName": "time",
                    "sortType": "desc",
                    "isHLtitle": "true",
                }
                resp = self.session.post(
                    "https://www.cninfo.com.cn/new/hisAnnouncement/query",
                    data=payload,
                    headers={**UA, "Referer": "https://www.cninfo.com.cn/"},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                rows = (resp.json().get("announcements") or [])
                if not rows:
                    break
                for row in rows:
                    title = _clean(row.get("announcementTitle"))
                    code = str(row.get("secCode") or "").zfill(6)
                    ts = row.get("announcementTime")
                    pub = _epoch_to_beijing(ts) if ts else ""
                    adjunct = str(row.get("adjunctUrl") or "")
                    url = "https://static.cninfo.com.cn/" + adjunct if adjunct else ""
                    if not code or not title or (code, title) in seen:
                        continue
                    seen.add((code, title))
                    out.append(
                        {
                            "source": "cninfo_disclosure",
                            "source_id": _hash(url or f"{code}{title}"),
                            "title": f"{code} {title}",
                            "summary": "",
                            "url": url,
                            "category": "公告/巨潮",
                            "publish_time": pub,
                            "tier": 1,
                            "dedup_key": _dedup_key("cninfo_disclosure", f"{code}{title}"),
                        }
                    )
                if len(rows) < 30:
                    break
                time.sleep(0.2)
        return out

    def collect_all(
        self,
        start_dt: str,
        end_dt: str,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        return self.collect_with_status(start_dt, end_dt, start_date, end_date)["items"]

    def collect_with_status(
        self,
        start_dt: str,
        end_dt: str,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        items: list[dict] = []
        status: dict[str, dict[str, Any]] = {}
        fetchers = (
            ("official", lambda: self.official_policies()),
            ("news", lambda: self.tushare_major_news(start_dt, end_dt)),
            ("disclosure", lambda: self.cninfo_announcements(start_date, end_date)),
        )
        for name, fetcher in fetchers:
            try:
                rows = fetcher()
                items.extend(rows)
                status[name] = {"ok": bool(rows), "count": len(rows), "error": ""}
            except Exception as exc:  # noqa: BLE001
                status[name] = {
                    "ok": False,
                    "count": 0,
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                }
                items.append(
                    {
                        "source": "collector_error",
                        "source_id": _hash(str(exc)),
                        "title": f"数据源异常: {type(exc).__name__}",
                        "summary": str(exc)[:500],
                        "url": "",
                        "category": "系统",
                        "publish_time": "",
                        "tier": 5,
                        "dedup_key": _hash("collector_error"),
                    }
                )
        return {"items": items, "status": status}
