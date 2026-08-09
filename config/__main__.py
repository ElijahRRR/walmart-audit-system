"""让 `python -m config --check` 可用 (等价于 python -m config.settings --check)。

单独放 __main__ 是为了避开 runpy 的 "found in sys.modules" 警告 —
config/__init__.py 会先 import settings, 直接 -m config.settings 会双重加载。
"""
from .settings import _cli

raise SystemExit(_cli())
