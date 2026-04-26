"""备份当前飞书 上架表 C/D/E 列到 JSON, 用于全量重跑后对比."""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from integrations import lark_client

OUT_DIR = Path(__file__).parent.parent / "tests" / "out"


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"lark_snapshot_{ts}.json"

    print("读飞书 上架表 ...")
    rows = await asyncio.to_thread(
        lark_client.read_paginated,
        "PDsRsfGRrhmWq1tOt25cHgUAnPh", "ea2b2a", "G", 200,
    )
    print(f"读到 {len(rows)} 行 (含表头)")

    snapshot = []
    for i, row in enumerate(rows):
        if i == 0:
            continue
        asin = row[0] if len(row) > 0 and row[0] else ""
        store = row[1] if len(row) > 1 and row[1] else ""
        pt = row[2] if len(row) > 2 and row[2] else ""
        verdict = row[3] if len(row) > 3 and row[3] else ""
        reason = row[4] if len(row) > 4 and row[4] else ""
        if asin:
            snapshot.append({
                "row_num": i + 1,
                "asin": asin,
                "store": store,
                "pt": pt,
                "verdict": verdict,
                "reason": reason,
            })

    out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 备份 {len(snapshot)} 行 → {out_path}")
    print(f"verdict 分布:")
    from collections import Counter
    for v, n in Counter(r["verdict"] for r in snapshot).most_common():
        print(f"  {v or '(空)':<10} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
