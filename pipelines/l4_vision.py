"""L4 视觉审核 — 豆包/qwen-vl 扫产品主图 + 辅图, 检测 7 类图像违规。

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

import asyncio
import base64
import logging
from typing import Any

import httpx

from config import get_settings
from integrations import llm_client

from .models import L1Result, L2Result, L3Result, L4Result, ProductInfo, RuleHit

log = logging.getLogger(__name__)

# 本地下载并发 + 超时 (绕开火山服务端下 Amazon 图超时 bug)
_IMG_DL_SEM = asyncio.Semaphore(30)
_IMG_DL_TIMEOUT = 8.0


async def _download_and_b64(url: str) -> str | None:
    """本地下载图 → data URL (base64 内联). 失败返回 None.

    传 base64 给 vision API, 绕开火山服务端下 Amazon 图超时的问题.
    """
    async with _IMG_DL_SEM:
        try:
            async with httpx.AsyncClient(
                trust_env=False, timeout=_IMG_DL_TIMEOUT,
                follow_redirects=True,
            ) as c:
                r = await c.get(url, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code != 200 or not r.content:
                    return None
                # 推断 mime
                ct = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
                if not ct.startswith("image/"):
                    # 从 URL 后缀猜
                    if url.lower().endswith(".png"):
                        ct = "image/png"
                    elif url.lower().endswith(".webp"):
                        ct = "image/webp"
                    else:
                        ct = "image/jpeg"
                b64 = base64.b64encode(r.content).decode("ascii")
                return f"data:{ct};base64,{b64}"
        except Exception as e:  # noqa: BLE001
            log.debug("图下载失败 %s: %s", url, e)
            return None


async def _urls_to_data_urls(urls: list[str]) -> list[str]:
    """并发下载 + 转 base64, 只保留成功的."""
    results = await asyncio.gather(*(_download_and_b64(u) for u in urls))
    return [r for r in results if r]


SYSTEM_PROMPT_EN = """You are a Walmart Marketplace visual compliance reviewer, acting on behalf of Walmart to audit product images submitted by Chinese drop-shipping sellers. Output strict JSON only — no explanations, no markdown wrappers.

# Detect 7 types of visual violations

## 1. Big-brand logo (trademark)
Clearly visible logos or full-word marks of known brands:
- Sportswear: Nike / Adidas / Jordan / Under Armour / Puma / Converse / Supreme
- 3C / Electronics: Apple / Samsung / Sony / Bose / Dyson / Beats / Rolex / Garmin
- Luxury: LV (Louis Vuitton) / Gucci / Chanel / Hermès / Prada / Burberry
- Entertainment: Disney / Marvel / Star Wars / HBO
- Other: LEGO / IKEA / Starbucks / McDonald's / Coca-Cola / Pepsi

## 2. ® / ™ / © symbols
If ® / ™ / © appears after any brand name or text, and the seller is not the brand owner, it's an infringement signal.

## 3. Cartoon / Anime / IP characters
- Disney (Mickey / Donald Duck / Princess line / Frozen / Toy Story)
- Pixar (The Incredibles / Toy Story / Finding Nemo)
- Marvel / DC superheroes (Spider-Man / Iron Man / Batman / Superman / Wolverine)
- Japanese anime / Pokémon (Pikachu / Doraemon / Naruto / One Piece)
- Studio Ghibli (Totoro / Spirited Away / Kiki)
- Sanrio (Hello Kitty / Kuromi / Cinnamoroll)
- SpongeBob / Peppa Pig / Paw Patrol / Sesame Street / Muppets

## 4. Sports team / University logos
- Four major US pro leagues: NBA / NFL / MLB / NHL / MLS — any team logo
- NCAA Division I schools (Alabama Crimson Tide / Ohio State Buckeyes / Kentucky Wildcats / Auburn Tigers / Texas A&M, etc.)
- FIFA World Cup / UEFA Champions League / F1 team logos

## 5. Celebrity portraits / signatures
Recognizable public figures' faces or signatures:
- Politicians: Trump / Biden / Obama / Harris / Putin
- Musicians: Taylor Swift / Beyoncé / BTS / Enhypen
- Athletes: LeBron / Kobe / Tom Brady / Messi / Ronaldo / Michael Jordan / Babe Ruth
- Other notable public figures

## 6. Offensive content
- Hate symbols (Nazi swastika / Confederate flag / KKK)
- Graphic violence / nudity / sexual innuendo
- Religious insult / racist symbols

## 7. Copycat design (Trade Dress / Design Patent)
Product shape / color / packaging clearly imitates a famous product:
- Stanley Cup tumbler (tapered shape with large handle)
- YETI Rambler (distinct double-wall insulated form)
- iPhone / AirPods / Apple Watch silhouette
- Tiffany blue + white ribbon gift box
- Classic Coca-Cola glass bottle shape / red-white wave script
- Hermès orange box
- LV Monogram pattern / Gucci red-green web stripe

---

# Decision rules

- **Default verdict = pass** — if evidence is ambiguous, let it pass (avoid over-blocking normal products)
- Only `confidence = "high"` findings trigger reject; medium/low are recorded but do not reject
- Specify image_index (0-based) for each finding
- State the actual brand / ip_name (not just "there is a logo")

# Output JSON schema

{
  "verdict": "pass" | "reject",
  "image_issues": [
    {
      "image_index": 0,
      "issue": "big_brand_logo|registered_symbol|cartoon_ip|sports_team|celebrity|offensive|copycat_design",
      "subtype": "Trademark | Copyright | Design Patent | Trade Dress | Publicity Rights | Offensive | Counterfeit",
      "brand_or_ip": "<specific brand / IP name, e.g. Nike / Mickey Mouse / Stanley Cup>",
      "confidence": "high" | "medium" | "low",
      "detail": "<brief description, up to 20 words, of where / how it appears>"
    }
  ],
  "reasoning": "<overall judgment rationale, up to 30 words>"
}

Constraints:
- verdict=reject ⇔ at least one image_issues entry has confidence=high
- image_issues can be an empty array (pass)
- Do not invent findings; only report what is actually visible in the images
"""

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


def _build_user_prompt_en(
    product: ProductInfo,
    l1: L1Result,
    l3: L3Result | None,
) -> str:
    """English user prompt (for Gemini); same semantics as Chinese version."""
    bullets_txt = "\n".join(f"  - {b}" for b in product.bullet_points[:3]) or "  (none)"
    l3_hint = ""
    if l3 and l3.reason_text:
        l3_hint = f"\n# L3 text-stage note\nL3 reason: {l3.reason_text[:150]}"

    return f"""# Product context (aids visual judgment)

ASIN: {product.asin}
Title: {product.title[:200]}
Brand field (seller-declared): {product.brand or '(empty)'}
Walmart Product Type: {l1.walmart_product_type or '(unknown)'}
Walmart Category: {l1.walmart_category or '(empty)'}

Top 3 bullet points:
{bullets_txt}
{l3_hint}

# Task

Inspect the following {{img_count}} product images and output JSON per the 7-category system rubric.
- If the title explicitly claims authorization (® / ™ / Official / Licensed / Authorized), trust it and do NOT reject.
- If the seller's brand field matches the brand visible in the image, trust it (it's their own product, not infringement).
"""


def _vision_hit(issue: dict[str, Any], aggressive_offensive: bool = False) -> bool:
    """high confidence 才算 hit.

    aggressive_offensive=True 时, offensive 类 issue 即使 medium 也算 hit —
    针对 Costume/Halloween/Religious/Stickers 等高 offensive 风险 PT, 把阈值降到 medium.
    这样处理是因为 offensive 视觉特征 (纳粹符号/宗教嘲讽/血腥) 主流 VLM 常给 medium,
    default pass 会让大量真违规漏掉.
    """
    if not isinstance(issue, dict):
        return False
    conf = issue.get("confidence")
    if conf == "high":
        return True
    if aggressive_offensive and conf == "medium":
        issue_type = (issue.get("issue") or "").lower()
        if "offensive" in issue_type:
            return True
    return False


# Aggressive offensive 模式判定 — 从共享模块导入, 与 L3 混合路由共用一套逻辑
from .aggressive_offensive import should_aggressive_offensive as _should_aggressive_offensive  # noqa: E402


# Offensive aggressive 增强块 — 附加到 system prompt 尾部, 只在 aggressive=True 时加
AGGRESSIVE_OFFENSIVE_ADDON_EN = """

---

# AGGRESSIVE OFFENSIVE MODE (this product's category has elevated offensive-content risk)

Apply a STRICTER standard to offensive content (issue = "offensive"):

1. ANY visible hate symbol (Nazi swastika, SS runes, Confederate flag, KKK hood, white-supremacist "OK" hand or "88") → confidence = "high"
2. Religious mockery / blasphemy (Jesus / Muhammad / Buddha in clearly disrespectful pose, satanic symbols on religious product, pentagram on Christian cross etc.) → confidence = "high"
3. Racist imagery (blackface caricature, "slanted-eye" Asian stereotype, antisemitic nose/hook, etc.) → confidence = "high"
4. Sexual / nudity / explicit sexual innuendo on products NOT in adult category → confidence = "high"
5. Graphic violence / gore / dismemberment / realistic weapons-on-person imagery → confidence = "high"
6. Political attack: leaders depicted with blood, in cages, being shot, hanged, etc. (regardless of which political side) → confidence = "high"
7. Drug paraphernalia imagery on non-smoking product (marijuana leaves on kids product etc.) → confidence = "high"

IMPORTANT: a costume of a historically sensitive persona (Nazi officer, KKK member, terrorist stereotype, "Native American chief" headdress for party wear) is OFFENSIVE — high.
Generic skeleton / monster / zombie / pumpkin / witch / vampire for Halloween is NOT offensive — do not flag.
Religious jewelry / religious decor featuring ONE faith (plain cross, Om, Star of David) is NOT offensive — do not flag.

For all OTHER issue types (logos, cartoons, celebrities, copycat) keep the normal "high-confidence only" standard.
"""

AGGRESSIVE_OFFENSIVE_ADDON_ZH = """

---

# AGGRESSIVE OFFENSIVE 模式 (此产品类目 offensive 风险升高)

对 "offensive" 类违规用**更严标准**:

1. 任何仇恨符号 (卐纳粹 / SS / 南方邦联旗 / KKK 头罩 / 白人至上主义 "OK" 手势或 "88") → confidence = "high"
2. 宗教嘲讽 / 亵渎 (耶稣 / 穆罕默德 / 佛像明显不敬, 宗教产品上的撒旦符号, 十字架上的五芒星等) → confidence = "high"
3. 种族歧视图像 (黑脸 blackface / "眯眯眼" 亚裔刻板 / 反犹钩鼻等) → confidence = "high"
4. 非成人类目产品上的色情 / 裸露 / 明显性暗示 → confidence = "high"
5. 血腥暴力 / 肢解 / 真实武器指人 → confidence = "high"
6. 政治攻击: 领导人被血 / 关笼 / 射杀 / 绞刑形象 (不分党派) → confidence = "high"
7. 非吸烟类产品上出现大麻叶 / 吸食工具图像 (如儿童产品带大麻叶) → confidence = "high"

重要:
- Cosplay/装扮 里的历史敏感人物 (纳粹军官 / KKK / 恐怖分子刻板 / 派对用 "印第安酋长头饰") = OFFENSIVE high.
- 万圣节通用骷髅 / 怪物 / 僵尸 / 南瓜 / 巫婆 / 吸血鬼 = **不** offensive, 不要标.
- 单一宗教装饰 (纯十字架 / Om / 大卫星) = **不** offensive, 不要标.

其他 issue 类型 (logos / cartoons / celebrities / copycat) 仍保持 "仅 high confidence" 的正常标准.
"""


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

    # 本地下载 + base64 转 data URL (绕开火山服务端拉 Amazon 图超时的 bug)
    data_urls = await _urls_to_data_urls(urls)
    if not data_urls:
        log.warning("L4 %s 所有 %d 张图本地下载失败, 回落 pass", product.asin, len(urls))
        return L4Result(verdict="pass", image_issues=[], hits=[])

    # Offensive aggressive 判定 (根据 PT + category + title 切更严阈值)
    aggressive_offensive = _should_aggressive_offensive(l1, product)

    # Gemini 用英文 prompt (更准), 国产模型用中文 (中文训练好)
    if s.vision_provider == "gemini":
        system_prompt = SYSTEM_PROMPT_EN
        if aggressive_offensive:
            system_prompt = system_prompt + AGGRESSIVE_OFFENSIVE_ADDON_EN
        user_prompt = _build_user_prompt_en(product, l1, l3).replace("{img_count}", str(len(data_urls)))
    else:
        system_prompt = SYSTEM_PROMPT
        if aggressive_offensive:
            system_prompt = system_prompt + AGGRESSIVE_OFFENSIVE_ADDON_ZH
        user_prompt = _build_user_prompt(product, l1, l3).replace("{img_count}", str(len(data_urls)))

    if aggressive_offensive:
        log.info("L4 aggressive offensive mode ON for %s (pt=%r, cat=%r)",
                 product.asin, l1.walmart_product_type, l1.walmart_category)

    # 组装多模态消息: system + user(text + images)
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for u in data_urls:
        content.append({"type": "image_url", "image_url": {"url": u}})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    # 按 provider 路由 (qwen / volcengine)
    raw: dict[str, Any] = {}
    try:
        if s.vision_provider == "gemini":
            from integrations import gemini_client
            raw = await gemini_client.vision_json(
                image_urls=data_urls,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1024,
            )
        elif s.vision_provider == "volcengine":
            from integrations import volcengine_client
            raw = await volcengine_client.vision_json(
                image_urls=data_urls,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1024,
            )
        else:  # qwen (DashScope qwen-vl-plus)
            from integrations.llm_client import _client, _sem, _json_from_text
            async with _sem():
                resp = await _client().chat.completions.create(
                    model=s.llm_vision_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1024,
                )
            text = (resp.choices[0].message.content or "").strip()
            raw = _json_from_text(text)
    except Exception as e:  # noqa: BLE001
        log.warning("L4 vision 调用失败 %s (provider=%s): %s - 回落 pass",
                    product.asin, s.vision_provider, e)
        return L4Result(verdict="pass", image_issues=[], hits=[])

    if not raw or "_raw" in raw:
        log.warning("L4 vision JSON 解析失败 %s: raw=%r - 回落 pass", product.asin, str(raw)[:200])
        return L4Result(verdict="pass", image_issues=[], hits=[])

    issues = raw.get("image_issues") or []
    if not isinstance(issues, list):
        issues = []

    # 判定: 正常 high confidence 才算; aggressive offensive 模式下 offensive medium 也算
    high_hits = [i for i in issues if isinstance(i, dict) and _vision_hit(i, aggressive_offensive)]
    verdict = "reject" if high_hits else "pass"

    hits: list[RuleHit] = []
    if verdict == "reject":
        # reject 原因如果是 aggressive-escalated offensive, 在 detail 里标出
        aggressive_escalated = aggressive_offensive and any(
            (i.get("confidence") == "medium" and "offensive" in (i.get("issue") or "").lower())
            for i in high_hits
        )
        hits.append(RuleHit(
            stage="L4",
            rule_code="l4_vision_violation",
            penalty=-100,
            detail={
                "issues": high_hits,
                "all_issues": issues,
                "reasoning": raw.get("reasoning"),
                "images_scanned": len(data_urls),
                "aggressive_offensive": aggressive_offensive,
                "aggressive_escalated": aggressive_escalated,
            },
        ))

    return L4Result(
        verdict=verdict,
        image_issues=issues,
        hits=hits,
    )


__all__ = ["judge"]
