"""
外部API客户端基础设施
统一管理外部数据源的可用性、mock fallback、错误处理和标准返回结构
"""
import asyncio
from typing import Any, Callable, Dict, Optional

import requests
from requests import exceptions as request_exceptions
from pydantic import BaseModel, Field

from app.config import settings


class ExternalAPIError(Exception):
    """外部API调用异常"""


class ExternalAPIResponse(BaseModel):
    """外部API标准返回结果"""

    success: bool = Field(..., description="外部API调用是否成功")
    data: Dict[str, Any] = Field(default_factory=dict, description="外部API返回数据")
    error: Optional[str] = Field(None, description="错误信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="调用元数据")

    def to_dict(self) -> Dict[str, Any]:
        """转换为普通字典"""
        return self.model_dump()


class ExternalDataSource(BaseModel):
    """外部数据源描述"""

    provider: str = Field(..., description="数据源提供方")
    source_type: str = Field(..., description="数据源类型")
    description: str = Field("", description="数据源说明")
    requires_api_key: bool = Field(True, description="是否需要API Key")


class ExternalAPIClient:
    """外部API客户端基类"""

    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        retry_times: Optional[int] = None,
        mock_enabled: Optional[bool] = None,
        request_func: Optional[Callable[..., Any]] = None,
    ):
        """
        初始化外部API客户端

        Args:
            name: API提供方名称
            api_key: API Key
            timeout: 请求超时时间
            retry_times: 重试次数
            mock_enabled: 是否允许mock fallback
            request_func: 可注入的请求函数，便于测试
        """
        self.name = name
        self.api_key = api_key or ""
        self.timeout = timeout if timeout is not None else settings.EXTERNAL_API_TIMEOUT
        self.retry_times = retry_times if retry_times is not None else settings.EXTERNAL_API_RETRY_TIMES
        self.mock_enabled = mock_enabled if mock_enabled is not None else settings.EXTERNAL_API_MOCK_ENABLED
        self.request_func = request_func or requests.request

    def available(self) -> bool:
        """判断当前客户端是否具备真实API调用条件"""
        return bool(self.api_key)

    async def request_json(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        mock_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """请求JSON数据，支持无Key时mock fallback"""
        if not self.available():
            if self.mock_enabled:
                return self.mock_response(mock_data=mock_data, reason="api_key_missing")
            return self.error_response(
                "API Key未配置，且mock fallback未启用。",
                mock=False,
                metadata={
                    "api_status": "unavailable",
                    "execution_mode": "unavailable",
                    "error_type": "api_key_missing",
                },
            )

        last_error = None
        last_error_type = None
        attempts = max(self.retry_times, 0) + 1
        for attempt in range(1, attempts + 1):
            try:
                response = await asyncio.to_thread(
                    self.request_func,
                    method,
                    url,
                    params=params,
                    headers=headers,
                    json=json_body,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    data = {"items": data}
                return self.success_response(
                    data=data,
                    mock=False,
                    metadata={
                        "url": url,
                        "method": method.upper(),
                        "attempt": attempt,
                        "attempt_count": attempt,
                        "retry_count": attempt - 1,
                        "api_status": "success",
                        "execution_mode": "real_api",
                    },
                )
            except Exception as exc:  # pragma: no cover - 网络错误分支由注入请求函数测试
                last_error = str(exc)
                last_error_type = self._classify_exception(exc)

        if self.mock_enabled:
            return self.mock_response(
                mock_data=mock_data,
                reason=last_error_type or last_error or "request_failed",
                metadata={
                    "api_status": "degraded",
                    "error_type": last_error_type or "request_failed",
                    "upstream_error": last_error,
                    "attempt_count": attempts,
                    "retry_count": max(attempts - 1, 0),
                },
            )
        return self.error_response(
            last_error or "外部API请求失败。",
            mock=False,
            metadata={
                "api_status": "failed",
                "execution_mode": "real_api_failed",
                "error_type": last_error_type or "request_failed",
                "attempt_count": attempts,
                "retry_count": max(attempts - 1, 0),
            },
        )

    def mock_response(
        self,
        mock_data: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构建mock fallback响应"""
        merged_metadata = dict(metadata or {})
        merged_metadata.update({
            "mock_reason": reason or "mock_enabled",
            "fallback_reason": reason or "mock_enabled",
            "fallback_used": True,
            "api_status": merged_metadata.get("api_status") or "degraded",
            "execution_mode": "mock_fallback",
        })
        return self.success_response(
            data=mock_data or {},
            mock=True,
            metadata=merged_metadata,
        )

    def success_response(
        self,
        data: Optional[Dict[str, Any]] = None,
        mock: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构建成功响应"""
        return ExternalAPIResponse(
            success=True,
            data=data or {},
            error=None,
            metadata=self._build_metadata(mock=mock, metadata=metadata),
        ).to_dict()

    def error_response(
        self,
        error: str,
        mock: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构建失败响应"""
        return ExternalAPIResponse(
            success=False,
            data={},
            error=error,
            metadata=self._build_metadata(mock=mock, metadata=metadata),
        ).to_dict()

    def _build_metadata(self, mock: bool, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """构建外部API元数据"""
        merged = dict(metadata or {})
        merged.setdefault("provider", self.name)
        merged.setdefault("mock", mock)
        merged.setdefault("source", "external_api")
        merged.setdefault("api_status", "degraded" if mock else "success")
        merged.setdefault("execution_mode", "mock_fallback" if mock else "real_api")
        merged.setdefault("fallback_used", bool(mock))
        return merged

    def _classify_exception(self, exc: Exception) -> str:
        """将requests异常映射为稳定错误类型，避免上层依赖原始异常文本。"""
        if isinstance(exc, request_exceptions.Timeout):
            return "timeout"
        if isinstance(exc, request_exceptions.HTTPError):
            return "http_error"
        if isinstance(exc, request_exceptions.ConnectionError):
            return "network_error"
        if isinstance(exc, ValueError):
            return "invalid_json"
        return "request_failed"
