"""政策/公告原文事实抽取（DeepSeek）。

只对高价值文档（一级政策源 + 候选池个股公告）做深度抽取；其余新闻仅保留
标题/摘要特征。输出 atomic_facts 入库，供报告与模型使用。
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests
from openai import OpenAI

from .. import config
from ..providers.news_sources import _clean, _hash
from ..storage import Storage

PROMPT = """你是A股研究系统的事实抽取器。把给定文档抽取为 JSON 数组，每项格式：
{"subject":"主体","predicate":"动作","object":"对象","value":数字或null,
 "unit":"单位或空","conditions":"条件或空","effective_time":"生效时间或空",
 "source_span":"原文片段","sector_links":["受影响行业/板块"],
 "verification_status":"verified|unverified"}
规则：只抽取原文明确出现的信息；不推测利好利空；不写原文没有的日期/金额；
只输出 JSON 数组，不要其他文字。文档："""


def _fetch_text(url: str, timeout: int = 15) -> str:
    if not url:
        return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.gov.cn/"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        if url.lower().endswith(".pdf") or resp.headers.get("Content-Type", "").find("pdf") >= 0:
            from io import BytesIO

            from pypdf import PdfReader

            reader = PdfReader(BytesIO(resp.content))
            return "".join(page.extract_text() or "" for page in reader.pages[:8])[:4000]
        text = re.sub(r"<script.*?</script>|<style.*?</style>", "", resp.text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        return _clean(text)[:4000]
    except Exception:  # noqa: BLE001
        return ""


def _llm_extract(document: str, client: OpenAI, model: str) -> list[dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你只输出合法 JSON 数组。"},
            {"role": "user", "content": PROMPT + document},
        ],
        temperature=0.0,
        max_tokens=1800,
    )
    content = response.choices[0].message.content or ""
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M)
    parsed = json.loads(content)
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict) and item.get("subject")]


def extract_facts(
    storage: Storage,
    items: list[dict[str, Any]],
    *,
    model_version: str = "deepseek_fact_v1",
    max_items: int | None = None,
) -> dict[str, Any]:
    api_key = config.env_str("AI_PRIMARY_API_KEY")
    if not api_key or not items:
        return {"extracted": 0, "skipped": len(items), "reason": "no_key_or_items"}
    max_items = max_items or config.env_int("NLP_MAX_ITEMS", 15)
    client = OpenAI(
        api_key=api_key,
        base_url=config.env_str("AI_PRIMARY_BASE_URL", "https://api.deepseek.com"),
    )
    model = config.env_str("AI_PRIMARY_MODEL", "deepseek-v4-flash")
    high_value = [
        item
        for item in items
        if item.get("tier", 5) <= 1 and item.get("title") and "异常" not in item.get("title", "")
    ][:max_items]
    extracted = 0
    errors = 0
    for item in high_value:
        document_id = item.get("source_id") or _hash(item.get("title", ""))
        text = item.get("summary") or ""
        if not text and item.get("url"):
            text = _fetch_text(str(item["url"]))
        if len(text) < 60:
            continue
        try:
            facts = _llm_extract(text, client, model)
        except Exception:  # noqa: BLE001
            errors += 1
            continue
        rows = []
        for fact in facts:
            rows.append(
                {
                    "document_id": document_id,
                    "source": item.get("source", ""),
                    "publish_time": item.get("publish_time", ""),
                    "subject": str(fact.get("subject", "")),
                    "predicate": str(fact.get("predicate", "")),
                    "object": str(fact.get("object", "")),
                    "value": fact.get("value"),
                    "unit": str(fact.get("unit", "")),
                    "conditions": str(fact.get("conditions", "")),
                    "effective_time": str(fact.get("effective_time", "")),
                    "source_span": str(fact.get("source_span", ""))[:500],
                    "sector_links": json.dumps(fact.get("sector_links", []), ensure_ascii=False),
                    "verification_status": str(fact.get("verification_status", "unverified")),
                    "model_version": model_version,
                }
            )
        if rows:
            storage.insert_facts(rows)
            extracted += len(rows)
    return {"extracted": extracted, "errors": errors, "candidates": len(high_value)}
