"""从 AuditOutcome 的所有 hits 中推导统一的 final_reason_category.

输出严格对齐 Walmart 37 条 Prohibited Product Policy 的 category_en,
用于合规报告给卖家时显示"按哪条 Walmart 政策被拒".

优先级 (先找到先用):
  1. L3 reason_category (LLM 语义判最准, 本身就是 Walmart 政策名)
  2. L4 image_issues (转成 Intellectual Property 或 Offensive Content)
  3. L2 forbidden_mega_cat.walmart_policy (yaml 定义)
  4. L2 cat_requires_cert_hard / soft (按 walmart_category 推)
  5. L2 trademark_live / title_desc_blacklist (映射 IP)
  6. L1 publication_pt_forbidden / excluded_category (映射对应政策)
  7. L0 phase0_forbidden_category.walmart_policy
  8. L0 phase0_brand_blacklist (IP)
  9. 基于 walmart_pt 关键词兜底 (pt_to_policy)

使用:
  from pipelines.reason_mapper import compute_final_reason
  final_cat = compute_final_reason(outcome)
"""
from __future__ import annotations

from typing import Any


# L3 reject reason_category → Walmart 37 条政策标准名
_L3_NORMALIZE = {
    'intellectual property':      'Intellectual Property',
    'offensive content':          'Offensive Content',
    'military & law enforcement': 'Military & Law Enforcement',
    'drugs & paraphernalia':      'Drugs & Paraphernalia',
    'cosmetic products':          'Cosmetic Products',
    'medical devices':            'Medical Devices',
    'baby products':              'Baby Products',
    "children's products":        "Children's Products",
    'pet products':               'Pet Products',
    'dietary supplements':        'Dietary Supplements',
    'alcohol':                    'Alcohol',
    'tobacco & vaping':           'Tobacco & Vaping',
    'food products':              'Food Products',
    'hazardous items':            'Hazardous Items',
    'electronics & rf':           'Electronics & RF',
    'textiles & apparel':         'Textiles & Apparel',
    'auto & motor vehicles':      'Auto & Motor Vehicles',
    'digital goods':              'Digital Goods',
    'general-use products':       'General-Use Products',
    'pfas chemicals':              'PFAS Chemicals',
    'brand_misuse':               'Intellectual Property',     # brand 伪装归到 IP
    'brand misuse':               'Intellectual Property',
}


def _normalize_l3_cat(cat: str | None) -> str | None:
    if not cat or cat.lower() in ('none', 'null', ''):
        return None
    return _L3_NORMALIZE.get(cat.strip().lower(), cat.strip().title())


# PT 关键词 → 政策 (兜底)
def _pt_to_policy(pt: str | None, title: str = '') -> str | None:
    """按 PT 名 (或 title 关键词) 推 Walmart 政策. 返回 None 表示无明确映射."""
    p = ((pt or '') + ' ' + (title or '')[:80]).lower()
    # 优先级从具体到宽泛
    if any(k in p for k in [
        'firearm', 'ammo', 'ammunition', 'crossbow', 'gun cleaning',
        'pistol', 'rifle', 'bb gun', 'pellet', 'airsoft', 'tactical knife',
        'pocket knife', 'throwing knife', 'switchblade', 'brass knuckle',
        'pepper spray', 'taser', 'stun gun',
    ]):
        return 'Military & Law Enforcement'
    if any(k in p for k in [
        'homeopathic', 'herbal', 'supplement', 'vitamin', 'drug test',
        'melatonin', 'prescription', 'otc drug', 'pharmaceutical',
    ]):
        return 'Dietary Supplements'
    if any(k in p for k in [
        'cosmetic', 'lipstick', 'makeup', 'nail polish', 'perfume',
        'fragrance', 'deodorant', 'tampon', 'essential oil',
        'body moisturizer', 'hand sanitizer', 'dry shampoo',
        'nail care', 'hair oil', 'teeth whitening',
    ]):
        return 'Cosmetic Products'
    if any(k in p for k in [
        'medical', 'thermometer', 'catheter', 'heating pad',
        'blood pressure', 'glucose meter', 'pulse oximeter',
        'nebulizer', 'cpap', 'syringe', 'diagnostic', 'surgical',
        'eye drop', 'spirometer', 'compression glove',
    ]):
        return 'Medical Devices'
    if any(k in p for k in [
        'pet food', 'dog food', 'cat food', 'animal feed', 'bird food',
        'pet treat', 'pet vitamin', 'pet supplement', 'animal supplement',
        'pet repellent',
    ]):
        return 'Pet Products'
    if any(k in p for k in ['alcohol', 'wine', 'beer', 'spirit', 'liquor', 'distill']):
        return 'Alcohol'
    if any(k in p for k in ['tobacco', 'cigarette', 'cigar', 'vape', 'e-cigarette', 'nicotine']):
        return 'Tobacco & Vaping'
    if any(k in p for k in ['baby', 'infant', 'toddler', 'crib', 'stroller', 'diaper']):
        return 'Baby Products'
    if any(k in p for k in ['apparel', 'clothing', 'shirt', 'pants', 'jeans', 'dress',
                             'sweater', 'jacket', 'underwear', 'bra', 'sock', 'footwear', 'shoe']):
        return 'Textiles & Apparel'
    if any(k in p for k in ['coin', 'currency', 'bullion', 'precious metal', 'jewelry', 'ring',
                             'necklace', 'bracelet']):
        return 'Jewelry/Precious Metals'
    return None


def compute_final_reason(
    outcome: Any,
    product: Any | None = None,
) -> str | None:
    """从 AuditOutcome 推导 Walmart 政策 category_en. 仅在 verdict=reject 时有意义.

    优先级 (从高到低):
      1. L0/L2 硬规则 detail.walmart_policy (类目禁售最直接对齐 Walmart 大类分类)
      2. L3 reason_category (LLM 语义判, 用于 L2 没命中大类的 case)
      3. L4 image_issues
      4. 兜底 (IP / PT 关键词 / cert)

    注: L2 硬规则优先于 L3, 因为 L2 forbidden_mega_cat 已直接映射到 Walmart
        37 条政策的 category_en. L3 可能把"大类禁售"的产品误判成"IP"(因为
        title 里常有知名品牌词), 导致对齐率退化.
    """
    if outcome is None or outcome.verdict != "reject":
        return None

    # 分 hard/soft hit:
    #   hard = forbidden_mega_cat / phase0_forbidden_category — 大类禁售精确对齐 Walmart 37 条政策
    #   soft = cert / brand / trademark 等 — 只标 L2 有命中, 实际原因要靠 L3 判
    HARD_RULE_CODES = {'forbidden_mega_cat', 'phase0_forbidden_category'}

    # (1) hard 规则 walmart_policy — 最优先
    for h in outcome.all_hits:
        if h.rule_code in HARD_RULE_CODES:
            policy = (h.detail or {}).get('walmart_policy')
            if policy:
                return policy

    # (2) L3 语义判 (硬规则没给时, L3 最准)
    if outcome.l3 and outcome.l3.verdict == 'reject':
        cat = _normalize_l3_cat(outcome.l3.reason_category)
        if cat:
            return cat

    # (1.5) 其他带 walmart_policy 的规则 (未来扩展用, 兜底放这)
    for h in outcome.all_hits:
        if h.rule_code not in HARD_RULE_CODES:
            policy = (h.detail or {}).get('walmart_policy')
            if policy:
                return policy

    # (3) L4 视觉
    if outcome.l4 and outcome.l4.verdict == 'reject':
        for issue in (outcome.l4.image_issues or []):
            if isinstance(issue, dict) and issue.get('confidence') == 'high':
                t = (issue.get('issue') or '').lower()
                if 'offensive' in t:
                    return 'Offensive Content'
                return 'Intellectual Property'

    # (4) 兜底
    hit_codes = {h.rule_code for h in outcome.all_hits}

    # 出版物 IP
    if 'publication_pt_forbidden' in hit_codes:
        return 'Intellectual Property'

    # L1 excluded_category / unmapped_amazon_path — 按 reason 推
    if outcome.l1 and (outcome.l1.excluded_category_reason or 'excluded_category' in hit_codes):
        reason = (outcome.l1.excluded_category_reason or '').lower()
        if '3c' in reason or '电子' in reason or 'electroni' in reason:
            return 'Electronics & RF'
        if '服饰' in reason or 'apparel' in reason or 'clothing' in reason:
            return 'Textiles & Apparel'
        if '汽配' in reason or 'automotive' in reason:
            return 'Auto & Motor Vehicles'
        if '带电' in reason or 'battery' in reason:
            return 'Hazardous Items'

    # PT 关键词兜底 (常用于 cert/禁售没命中大类但 PT 能推)
    pt = outcome.l1.walmart_product_type if outcome.l1 else None
    title = product.title if product else None
    by_pt = _pt_to_policy(pt, title or '')
    if by_pt:
        return by_pt

    # cat_requires_cert_* 按 walmart_category 推
    if outcome.l1:
        wc = (outcome.l1.walmart_category or '').lower()
        if {'cat_requires_cert_hard', 'cat_requires_cert_small_part',
            'cat_requires_cert_soft'} & hit_codes:
            if 'baby' in wc or 'children' in wc:
                return "Children's Products"
            if 'health' in wc or 'medical' in wc or 'personal care' in wc:
                return 'Medical Devices'
            if 'electronics' in wc:
                return 'Electronics & RF'
            return 'General-Use Products'

    # IP 兜底
    if 'trademark_live' in hit_codes or 'title_desc_blacklist' in hit_codes:
        return 'Intellectual Property'
    if 'phase0_brand_blacklist' in hit_codes:
        return 'Intellectual Property'

    return 'General-Use Products'


__all__ = ["compute_final_reason"]
