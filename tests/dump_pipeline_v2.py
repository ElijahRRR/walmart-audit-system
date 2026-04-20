"""把 100 条 DMIT 产品跑 Phase0+L1+L2, 完整输出 JSON + 人类可读 Markdown。

Markdown 格式 — 便于我 (Claude) 逐条用 AI 能力检查:
  - ASIN / title / brand / amazon_category / L1 PT / hits / verdict

JSON 格式 — 供脚本再分析。

输出到 tests/out/pipeline_v2_dump.{json,md}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path

import httpx

from config import get_settings, ROOT
from integrations.dmit_client import ProductDetail
from pipelines import l1_category, l2_rules, phase0_brand
from pipelines.models import ProductInfo

log = logging.getLogger(__name__)

OUT_DIR = ROOT / "tests" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


async def _list_dmit(limit: int) -> list[dict]:
    base = get_settings().dmit_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=30.0) as c:
        resp = await c.get(f"{base}/api/results", params={"limit": limit})
        resp.raise_for_status()
    return resp.json().get("items") or []


def _serialize_hit(h) -> dict:
    return {"stage": h.stage, "rule_code": h.rule_code, "penalty": h.penalty, "detail": h.detail}


async def run(sample: int) -> None:
    items = await _list_dmit(sample * 2)
    items = [it for it in items if it.get("title") and (it.get("category_tree") or "").strip() not in ("", "N/A")][:sample]
    log.warning("dump %d 条 ...", len(items))

    async def _one(it):
        pd = ProductDetail.from_dmit(it)
        p = ProductInfo(
            asin=pd.asin, title=pd.title or "", brand=pd.brand or "",
            amazon_category_path=pd.amazon_category_path or "",
            bullet_points=pd.bullet_points,
            long_description=pd.long_description or "",
            country_of_origin=pd.country_of_origin or "",
        )
        p0 = phase0_brand.check(p)
        if p0.blocked:
            return {
                "asin": p.asin,
                "title": p.title,
                "brand": p.brand,
                "amazon_category_path": p.amazon_category_path,
                "country_of_origin": p.country_of_origin,
                "phase0_blocked": True,
                "phase0_matched_brand": p0.matched_brand,
                "hits": [_serialize_hit(h) for h in p0.hits],
                "verdict": "reject",
                "score_final": 0,
                "stage_stopped_at": "L0",
            }
        l1 = await l1_category.classify(p)
        l2 = l2_rules.evaluate(p, l1)
        return {
            "asin": p.asin,
            "title": p.title,
            "brand": p.brand,
            "amazon_category_path": p.amazon_category_path,
            "country_of_origin": p.country_of_origin,
            "bullet_points": p.bullet_points[:3],
            "phase0_blocked": False,
            "l1": {
                "walmart_product_type": l1.walmart_product_type,
                "pt_confidence": l1.pt_confidence,
                "pt_source": l1.pt_source,
                "walmart_category": l1.walmart_category,
                "zh_can_do": l1.zh_can_do,
                "excluded_category_reason": l1.excluded_category_reason,
                "requires_certificate": l1.requires_certificate,
                "requirements_text": l1.requirements_text,
            },
            "hits": [_serialize_hit(h) for h in l1.hits + l2.hits],
            "verdict": l2.verdict,
            "score_final": l2.score_final,
            "stage_stopped_at": "L2" if l2.verdict == "reject" else None,
        }

    results = await asyncio.gather(*[_one(it) for it in items])

    # JSON
    jp = OUT_DIR / "pipeline_v2_dump.json"
    jp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log.warning("wrote %s", jp)

    # Markdown
    md_lines = [
        "# Pipeline v2 100 条 DMIT 完整输出",
        "",
        f"总数: {len(results)}",
        "",
        f"Phase0 blocked: {sum(1 for r in results if r.get('phase0_blocked'))}",
        f"Reject (非 Phase0): {sum(1 for r in results if r.get('verdict') == 'reject' and not r.get('phase0_blocked'))}",
        f"Pass: {sum(1 for r in results if r.get('verdict') == 'pass')}",
        "",
        "---",
        "",
    ]
    for i, r in enumerate(results, 1):
        md_lines.append(f"## {i}. {r['asin']} — [{r['verdict'].upper()}] score={r['score_final']}")
        md_lines.append(f"**title**: {r['title']}")
        md_lines.append(f"**brand**: `{r['brand']}`  **amazon_cat**: `{r['amazon_category_path']}`")
        if r.get('phase0_blocked'):
            md_lines.append(f"**Phase0 BLOCKED**: matched_brand=`{r.get('phase0_matched_brand')}`")
        else:
            l1 = r["l1"]
            md_lines.append(
                f"**L1 PT**: `{l1['walmart_product_type']}` "
                f"(src={l1['pt_source']}, conf={l1['pt_confidence']}, "
                f"cat=`{l1.get('walmart_category')}`, zh_can_do=`{l1.get('zh_can_do')}`"
                + (f", excluded={l1.get('excluded_category_reason')}" if l1.get('excluded_category_reason') else "")
                + ")"
            )
        if r["hits"]:
            md_lines.append("")
            md_lines.append("**Hits**:")
            for h in r["hits"]:
                detail_summary = _summarize_detail(h)
                md_lines.append(f"- `{h['rule_code']}` ({h['penalty']:+d}) — {detail_summary}")
        md_lines.append("")

    md = OUT_DIR / "pipeline_v2_dump.md"
    md.write_text("\n".join(md_lines), encoding="utf-8")
    log.warning("wrote %s", md)


def _summarize_detail(h: dict) -> str:
    d = h.get("detail", {})
    code = h["rule_code"]
    if code == "phase0_brand_blacklist":
        return f"brand={d.get('brand')} matched={d.get('matched_brand')}"
    if code == "excluded_category":
        return f"reason={d.get('reason')} from_seed={d.get('from_seed_yaml')} pt={d.get('pt')}"
    if code == "cat_forbid_zh_seller":
        first = (d.get("hits") or [{}])[0]
        return f"zh_can_do='{first.get('zh_can_do','')}' mappings={d.get('mappings_count')}"
    if code == "forbidden_mega_cat":
        return f"key={d.get('key')} by={d.get('matched_by')} term='{d.get('matched_term')}'"
    if code == "cat_requires_cert":
        fields = d.get("required_cert_fields", [])
        return f"cert_fields={fields}"
    if code == "title_desc_blacklist":
        matches = d.get("matches", [])
        brands = [m.get("brand") for m in matches][:8]
        return f"count={d.get('count')} brands={brands}{'...' if len(matches) > 8 else ''}"
    if code == "trademark_live":
        marks = (d.get("matched_marks") or [])[:10]
        return f"count={d.get('count')} marks={marks}{'...' if len(d.get('matched_marks') or []) > 10 else ''}"
    return json.dumps(d, ensure_ascii=False)[:150]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=100)
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    asyncio.run(run(args.sample))
