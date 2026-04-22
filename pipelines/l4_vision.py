"""L4 视觉审核 — qwen-vl-plus 扫产品主图 + 辅图, 检测 7 类图像违规。

只在 L3 通过后跑 (最贵的一层, 不浪费 LLM vision tokens).

7 类检测 (对应 L3 IP 7 子类, 覆盖纯文本判不出来的):
  1. 大牌 logo       (Nike swoosh / Apple / LEGO / Adidas / Disney wordmark)
  2. ® / ™ 角标       (任何角标出现 = 商标声称)
  3. 卡通 IP 形象     (Mickey / Pokémon / Marvel heroes / Ghibli / Sanrio / SpongeBob)
  4. 运动队/学校 logo (NBA/NFL/MLB/NCAA 队徽)
  5. 名人肖像/签名    (Trump / Taylor Swift / Tom Brady 等)
  6. 冒犯内容        (纳粹/邦联旗/色情/血腥)
  7. 外观撞款        (Stanley Cup 杯型 / iPhone 壳设计 / Yeti Tumbler 风格 / Tiffany 蓝盒)

设计原则:
  - 单次 vision 调用处理 N 张图 (qwen-vl-plus 多图同传), 省 tokens
  - 默认 verdict=pass — 除非**明确证据** (避免过激), 低置信度不拦
  - 命中返回 RuleHit + image_issues 供下游 (人工 review / 图像处理) 使用

并发 / 重试 / 缓存由 llm_client 处理.

public:
  async judge(product, l1, l2, l3) -> L4Result
"""
from __future__ import annotations

import logging
from typing import Any

from config import get_settings
from integrations import llm_client

from .models import L1Result, L2Result, L3Result, L4Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是沃尔玛 Marketplace 视觉合规审核 AI, 站在沃尔玛官方立场审核中国搬运卖家产品图片.
只输出严格 JSON, 不要解释文字或 markdown 前后缀.

# 检测 7 类图像违规

## 1. 大牌 logo (trademark)
图片上明确显示知名品牌 logo 或完整字母 wordmark:
- 运动: Nike / Adidas / Jordan / Under Armour / Puma / Converse / Supreme
- 3C: Apple 苹果 logo / Samsung / Sony / Bose / Dyson / Beats / Rolex / Garmin
- 奢侈: LV / Gucci / Chanel / Hermès / Prada / Burberry
- 娱乐: Disney / Marvel / Star Wars / HBO
- 其他: LEGO / IKEA / Starbucks / McDonald's / Coca-Cola / Pepsi

## 2. ® 或 ™ 角标
图片右上角或文字后跟 ® / ™ / © → 说明商标声称, 若商家非注册方 = 侵权

## 3. 卡通 / 动漫 IP 形象
- 迪士尼角色 (Mickey Mouse / Donald Duck / 公主系列 / 冰雪奇缘 / Toy Story)
- 皮克斯 (超人总动员 / 玩具总动员 / Finding Nemo)
- 漫威/DC 英雄 (蜘蛛侠 / 钢铁侠 / 蝙蝠侠 / 超人)
- 日漫/宝可梦 (皮卡丘 / 哆啦A梦 / 火影忍者 / 海贼王)
- 吉卜力 (龙猫 / 千与千寻 / 魔女宅急便)
- 三丽鸥 (Hello Kitty / Kuromi / Cinnamoroll)
- SpongeBob / Peppa Pig / Paw Patrol / Sesame Street / Muppets

## 4. 运动队 / 学校 logo
- 美国四大职业体育: NBA / NFL / MLB / NHL / MLS 任一球队队徽
- NCAA Division I 大学 (Alabama Crimson Tide / Ohio State Buckeyes / Georgia Bulldogs / Duke Blue Devils 等)
- 世界杯 / 欧冠 / F1 车队 logo

## 5. 名人肖像 / 签名
可识别的公众人物面部/签名:
- 政治: Trump / Biden / Obama / Harris / 普京
- 明星: Taylor Swift / Beyoncé / BTS / Enhypen
- 运动员: LeBron / Kobe / Tom Brady / Messi / Ronaldo / Jordan
- 其他知名公众人物

## 6. 冒犯内容
- 仇恨符号 (卐字纳粹 / 南方邦联旗 / KKK)
- 血腥暴力 / 色情裸露 / 性暗示
- 宗教亵渎 / 种族歧视符号

## 7. 外观撞款 (Trade Dress / Design Patent)
产品形状/配色/包装明显仿知名产品:
- Stanley Cup tumbler (锥形大把手杯)
- YETI Rambler (双壁不锈钢保温杯典型形状)
- iPhone / AirPods / Apple Watch 外形
- Tiffany 蓝 + 白丝带礼盒
- 可口可乐经典瓶型 / 红白波浪字体
- 爱马仕橙色盒
- LV Monogram 格纹 / Gucci 红绿条纹

---

# 判定规则

- **default verdict = pass** — 证据不明确时放行 (避免过激误杀一般产品)
- 命中任一类别时必须**高置信度** (high) 才 reject; medium / low 只记录不 reject
- 必须指明是哪张图 (image_index, 从 0 开始)
- 说明 brand / ip_name 具体是什么 (不能只说"有 logo")

# 输出 JSON schema

{
  "verdict": "pass" | "reject",
  "image_issues": [
    {
      "image_index": 0,
      "issue": "big_brand_logo|registered_symbol|cartoon_ip|sports_team|celebrity|offensive|copycat_design",
      "subtype": "商标 | 版权 | 外观专利 | 商业包装 | 肖像权 | 冒犯 | 假冒",
      "brand_or_ip": "<具体品牌/IP 名, 如 Nike / Mickey Mouse / Stanley Cup>",
      "confidence": "high" | "medium" | "low",
      "detail": "<20 字内中文说明命中位置/形态>"
    }
  ],
  "reasoning": "<30 字内总体判断依据>"
}

约束:
- verdict=reject ⇔ image_issues 里至少一条 confidence=high
- image_issues 可为空数组 (pass)
- 未识别的图不要编造 issue
"""


def _build_user_prompt(
    product: ProductInfo,
    l1: L1Result,
    l3: L3Result | None,
) -> str:
    """给 vision LLM 的 user prompt, 附产品上下文让它更准确判 IP."""
    bullets_txt = "\n".join(f"  - {b}" for b in product.bullet_points[:3]) or "  (无)"
    l3_hint = ""
    if l3 and l3.reason_text:
        l3_hint = f"\n# L3 文本审已标注\nL3 reason: {l3.reason_text[:150]}"

    return f"""# 产品上下文 (辅助判图)

ASIN: {product.asin}
标题: {product.title[:200]}
品牌字段: {product.brand or '(空)'}
Walmart PT: {l1.walmart_product_type or '(unknown)'}
Walmart 类目: {l1.walmart_category or '(空)'}

五点 (前 3 条):
{bullets_txt}
{l3_hint}

# 任务

请逐张检查以下 {{img_count}} 张产品图, 按系统提示里的 7 类标准输出 JSON.
如果产品标题已经明确声称授权 (® / ™ / Official / Licensed / Authorized), 信任之, 不要 reject.
如果商家自填 brand 字段 = 图像上的 brand, 也信任 (是他自己的品牌, 不是冒用).
"""


def _vision_hit(issue: dict[str, Any]) -> bool:
    """high confidence 才算 hit."""
    return issue.get("confidence") == "high"


# DMIT 采集里 Amazon 图 URL 常以 '.' 结尾缺扩展名 (e.g. '..._AC_SL1500_.'),
# qwen-vl-plus 拒收无扩展名 URL. 这里补 jpg. 其他尾部情况保持不动.
def _normalize_image_url(url: str) -> str:
    if not url:
        return url
    # 去 query string (Amazon 图 query 通常没用)
    base = url.split("?", 1)[0]
    # 常见情况: 尾部是 '.' 或 '_.' 补 jpg
    if base.endswith("."):
        return base + "jpg"
    # 没有常见图像扩展名 → 补 .jpg (尾部 alphanum 结尾)
    lowered = base.lower()
    if not any(lowered.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")):
        # Amazon URL 形如 I/xxx._AC_SL1500_ 末尾无点: 补 .jpg
        return base + ".jpg"
    return base


async def judge(
    product: ProductInfo,
    l1: L1Result,
    l2: L2Result | None = None,
    l3: L3Result | None = None,
) -> L4Result:
    """L4 视觉审核主入口.

    仅使用 product.image_urls 前 N 张 (N 由 settings.l4_max_images_per_product, 默认 5).
    无图像则直接 pass.
    """
    s = get_settings()

    # product.image_urls 是 [{url, is_main}] 结构, 但 DMIT 把多张图 URL 用 \n 连在一个 url 字段里,
    # 需要拆分 + 归一化
    raw_imgs: list[dict[str, Any]] = list(product.image_urls or [])
    main_urls: list[str] = []
    other_urls: list[str] = []

    def _explode(u: str) -> list[str]:
        """url 串内若含 \n / \r / space 分隔多 URL, 全拆出; 每条单独 normalize."""
        parts: list[str] = []
        for piece in (u or "").replace("\r", "\n").split("\n"):
            piece = piece.strip()
            if piece.startswith("http"):
                parts.append(_normalize_image_url(piece))
        return parts

    for item in raw_imgs:
        if not isinstance(item, dict):
            if isinstance(item, str):
                for u in _explode(item):
                    other_urls.append(u)
            continue
        u_str = item.get("url") or item.get("src") or ""
        if not isinstance(u_str, str):
            continue
        urls_split = _explode(u_str)
        if not urls_split:
            continue
        if item.get("is_main"):
            main_urls.extend(urls_split)
        else:
            other_urls.extend(urls_split)

    # 去重保序
    seen: set[str] = set()
    ordered: list[str] = []
    for u in (main_urls + other_urls):
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    urls: list[str] = ordered[: s.l4_max_images_per_product]

    if not urls:
        return L4Result(verdict="pass", image_issues=[], hits=[])

    system_prompt = SYSTEM_PROMPT
    user_prompt = _build_user_prompt(product, l1, l3).replace("{img_count}", str(len(urls)))

    # 组装多模态消息: system + user(text + images)
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for url in urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    try:
        # llm_client.vision_json 只接受 (image_urls, prompt), 这里绕开直接走 chat API 以带 system prompt
        from integrations.llm_client import _client, _sem
        async with _sem():
            resp = await _client().chat.completions.create(
                model=s.llm_vision_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        log.warning("L4 vision 调用失败 %s: %s - 回落 pass", product.asin, e)
        return L4Result(verdict="pass", image_issues=[], hits=[])

    # 解析 JSON
    from integrations.llm_client import _json_from_text
    raw = _json_from_text(text)
    if not raw or "_raw" in raw:
        log.warning("L4 vision JSON 解析失败 %s: %r - 回落 pass", product.asin, text[:200])
        return L4Result(verdict="pass", image_issues=[], hits=[])

    issues = raw.get("image_issues") or []
    if not isinstance(issues, list):
        issues = []

    # 判定: 只有 confidence=high 的 issue 才真让 reject
    high_hits = [i for i in issues if isinstance(i, dict) and _vision_hit(i)]
    verdict = "reject" if high_hits else "pass"

    hits: list[RuleHit] = []
    if verdict == "reject":
        hits.append(RuleHit(
            stage="L4",
            rule_code="l4_vision_violation",
            penalty=-100,
            detail={
                "issues": high_hits,
                "all_issues": issues,
                "reasoning": raw.get("reasoning"),
                "images_scanned": len(urls),
            },
        ))

    return L4Result(
        verdict=verdict,
        image_issues=issues,
        hits=hits,
    )


__all__ = ["judge"]
