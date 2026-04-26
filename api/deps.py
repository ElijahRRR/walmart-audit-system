"""FastAPI 依赖: API Key 校验 + (日后) DB pool 注入.

.env 配置:
  API_KEY=your-secret-here    # 不配则服务不认证, 仅本机回环使用
"""
from __future__ import annotations

import os

from fastapi import Header, HTTPException, status


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """校验 X-API-Key header.

    行为:
      - 未配置 API_KEY env → 不校验 (开发环境 / 本机内部调用)
      - 配置了 API_KEY → 必须请求头带一致的 key
    """
    expected = os.getenv("API_KEY")
    if not expected:
        return  # 未启用认证
    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid X-API-Key",
        )


__all__ = ["require_api_key"]
