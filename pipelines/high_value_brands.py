"""高辨识度大牌清单 — 用于 L2 R6 "Compatible for <Brand>" 硬拦.

为什么需要单独维护:
  blacklist_brands 表有 36793 条 USPTO 注册品牌, 其中**大量通用英语词**
  (seat/kind/comfort/memory foam/fishing/slim/sportsman/players 等).
  R6 如果直接用整张表, 会把 "fits most fishing rods" / "suitable for players"
  这些合规描述误拦成 IP 侵权.

本清单只含:
  - 绝对可识别的大牌 (Dyson / Apple / Nike / Disney / LEGO / Samsung)
  - 即使有普通词义也因 R6 语法位置保证为品牌 (Apple / Stanley 在 "fits Apple" 上下文)

不含:
  - 歧义过大的单字母/两字母 (LV 单独易误判 — 由 R4 -15 兜底)
  - USPTO 的小众品牌 (由 R4 -15 处理)

维护方式:
  - 纯代码清单, 改一次即生效
  - 未来可扩 Licensed IP 清单 (Marvel/DC characters, sports teams 等)
"""
from __future__ import annotations


# 核心大牌 — 100% 辨识度, 无歧义
HIGH_VALUE_BRANDS: set[str] = {
    # === 3C / 电子 ===
    "apple", "iphone", "ipad", "ipod", "airpods", "airtag", "imac", "macbook",
    "samsung", "galaxy",
    "sony", "playstation", "ps4", "ps5", "xperia",
    "microsoft", "xbox", "surface",
    "google", "pixel", "chromecast",
    "nintendo", "switch",     # 注: Switch 在"Nintendo Switch"上下文才明确
    "intel", "amd", "nvidia", "geforce", "rtx",
    "logitech", "razer",
    "bose", "sonos", "beats", "sennheiser", "shure",
    "dyson",
    "garmin",
    "gopro",
    "oculus", "meta quest",
    "kindle",
    "huawei", "xiaomi", "oneplus",
    "tp-link", "netgear", "asus",
    "anker",

    # === 家电 / 厨房 ===
    "keurig", "nespresso", "breville", "delonghi",
    "roomba", "irobot", "shark", "dyson v6", "dyson v8", "dyson v10", "dyson v11",
    "sodastream",
    "kitchenaid", "cuisinart", "vitamix",
    "instant pot", "instapot",
    "ninja",                   # Ninja Foodi 等厨具大牌
    "yeti", "stanley",         # 杯具 (Stanley Cup tumbler, Yeti Rambler)
    "ember",

    # === 服装 / 鞋类 ===
    "nike", "adidas", "jordan", "air jordan", "puma", "under armour", "reebok",
    "converse", "vans", "crocs", "new balance",
    "supreme", "bape",
    "levi", "levis",
    "champion", "fila",
    "timberland", "ugg",

    # === 奢侈品 ===
    "louis vuitton", "lv monogram",
    "gucci", "chanel", "hermes", "hermès",
    "prada", "burberry", "dior",
    "cartier", "tiffany",
    "rolex", "omega", "patek philippe",
    "coach",

    # === 娱乐 / 影视 / 游戏 IP ===
    "disney", "mickey mouse", "disney princess", "frozen",
    "pixar",
    "marvel", "spider-man", "spiderman", "iron man", "captain america", "thor",
    "dc comics", "batman", "superman", "wonder woman",
    "star wars", "mandalorian", "jedi",
    "harry potter", "hogwarts",
    "pokemon", "pokémon", "pikachu",
    "hello kitty", "sanrio", "kuromi", "cinnamoroll", "my melody",
    "lego", "duplo",
    "hot wheels",
    "barbie",
    "sonic", "mario", "zelda", "luigi",
    "peppa pig", "paw patrol", "spongebob",
    "one piece", "luffy", "naruto", "dragon ball",
    "studio ghibli", "totoro",

    # === 运动联赛 / 高校 ===
    "nba", "nfl", "mlb", "nhl", "mls",
    "fifa", "uefa", "champions league",
    "ncaa",
    # 主要球队
    "lakers", "celtics", "warriors", "bulls", "heat", "knicks",
    "cowboys", "patriots", "giants", "packers", "eagles",
    "yankees", "red sox", "dodgers",

    # === 饮料 / 食品 ===
    "coca-cola", "coca cola", "cocacola", "coke",
    "pepsi",
    "starbucks",
    "mcdonald", "mcdonalds", "mcdonald's",
    "kfc",
    "kellogg", "kelloggs",
    "heinz",
    "red bull", "monster energy", "gatorade",

    # === 汽车 ===
    "ford", "chevrolet", "chevy", "toyota", "honda", "mazda", "nissan",
    "tesla", "model s", "model 3", "model y", "cybertruck",
    "bmw", "mercedes-benz", "mercedes", "audi", "porsche",
    "jeep", "dodge", "ram", "gmc",
    "polaris", "can-am",

    # === 乐器 / 音乐 ===
    "fender", "gibson", "taylor", "martin",
    "yamaha", "roland", "casio",
    "zildjian", "sabian",
    "drum workshop", "dw drums",
    "hercules",
    "shure", "sennheiser", "akg",

    # === 高尔夫 / 运动器材 ===
    "callaway", "titleist", "taylormade", "maxfli", "pinnacle",
    "wilson", "spalding",
    "rawlings",
    "trek", "specialized", "giant bicycles",

    # === 工具 / 户外 (注意 Stanley 已列) ===
    "dewalt", "makita", "milwaukee", "bosch",
    "black & decker", "black and decker", "blackdecker",
    "ryobi",
    "craftsman",
    "leatherman",
    "coleman",
    "columbia",
    "the north face", "north face", "patagonia",
    "osprey",

    # === 婴童 ===
    "pampers", "huggies", "luvs",
    "gerber",
    "fisher-price", "fisher price",
    "graco",
    "chicco",
    "britax",
    "maxi-cosi",

    # === 美妆 / 个护 ===
    "mac cosmetics", "mac lipstick",        # MAC 单独歧义大, 带修饰
    "clinique", "estee lauder", "estée lauder",
    "l'oreal", "loreal",
    "maybelline",
    "revlon",
    "chanel no 5", "chanel no5",
    "tom ford",
    "gucci guilty",
    "olay",
    "neutrogena",
    "dove",

    # === 其他 (常被仿制) ===
    "3m",
    "swiffer",
    "tide", "gain", "persil",
    "lysol",
    "post-it",
    "sharpie", "bic",
    "moleskine",
    "parker pen",
    "cross pen",
}


def is_high_value_brand(name: str) -> bool:
    """判断某个词是否在高辨识度大牌清单 (大小写无关)."""
    if not name:
        return False
    return name.strip().lower() in HIGH_VALUE_BRANDS


__all__ = ["HIGH_VALUE_BRANDS", "is_high_value_brand"]
