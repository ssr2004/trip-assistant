"""
LLM客户端封装
提供OpenAI-compatible模型调用和无API Key降级能力
"""
import time
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from app.config import Settings, get_settings
from core.llm.schemas import LLMRequest, LLMResponse


class LLMClient:
    """OpenAI-compatible LLM客户端"""

    def __init__(self, settings: Optional[Settings] = None, openai_client: Any = None):
        """
        初始化LLM客户端

        Args:
            settings: 应用配置，默认使用全局配置
            openai_client: 可注入的OpenAI兼容客户端，主要用于测试
        """
        self.settings = settings or get_settings()
        self._client = openai_client or self._build_client()

    @property
    def available(self) -> bool:
        """LLM是否具备真实调用条件"""
        return bool(self.settings.LLM_API_KEY and self.settings.LLM_BASE_URL)

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """
        调用聊天模型

        Args:
            request: 标准LLM请求

        Returns:
            标准LLM响应
        """
        model = request.model or self.settings.LLM_MODEL
        started_at = time.perf_counter()
        temperature = request.temperature
        if temperature is None:
            temperature = self.settings.LLM_TEMPERATURE

        request_metadata = self._sanitize_request_metadata(request.metadata)
        metadata = {
            **self._build_metadata(model=model, fallback=not self.available),
            **request_metadata,
        }

        if not self.available:
            return LLMResponse(
                success=False,
                content="",
                error="LLM_API_KEY未配置，已使用规则逻辑降级。",
                metadata={**metadata, "error_type": "missing_api_key", "duration_ms": self._elapsed_ms(started_at)},
            )

        if not request.messages:
            return LLMResponse(
                success=False,
                content="",
                error="LLM请求缺少messages。",
                metadata={
                    **self._build_metadata(model=model),
                    **request_metadata,
                    "error_type": "invalid_request",
                    "duration_ms": self._elapsed_ms(started_at),
                },
            )

        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [message.model_dump() for message in request.messages],
                "temperature": temperature,
            }
            if request.response_format == "json_object":
                kwargs["response_format"] = {"type": "json_object"}

            completion = await self._client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content or ""
            usage_metadata = self._extract_usage_metadata(completion)

            return LLMResponse(
                success=True,
                content=content,
                error=None,
                metadata={
                    **self._build_metadata(model=model),
                    **request_metadata,
                    "response_format": request.response_format,
                    "execution_mode": "llm",
                    "duration_ms": self._elapsed_ms(started_at),
                    **usage_metadata,
                },
            )
        except Exception as exc:  # pragma: no cover - 具体异常类型由SDK决定
            return LLMResponse(
                success=False,
                content="",
                error=self._sanitize_error(exc),
                metadata={
                    **self._build_metadata(model=model),
                    **request_metadata,
                    "error_type": self._classify_error(exc),
                    "duration_ms": self._elapsed_ms(started_at),
                },
            )

    def _build_client(self) -> Optional[AsyncOpenAI]:
        """构建OpenAI兼容客户端"""
        if not self.available:
            return None
        return AsyncOpenAI(
            api_key=self.settings.LLM_API_KEY,
            base_url=self.settings.LLM_BASE_URL,
        )

    def _build_metadata(self, model: str, fallback: bool = False) -> Dict[str, Any]:
        """构建不包含密钥的元信息"""
        return {
            "provider": self.settings.LLM_PROVIDER,
            "model": model,
            "base_url": self.settings.LLM_BASE_URL,
            "fallback": fallback,
            "execution_mode": "rule_fallback" if fallback else "llm",
        }

    def _sanitize_request_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only trace-safe request metadata."""
        allowed_keys = {
            "prompt_id",
            "prompt_name",
            "prompt_version",
            "prompt_purpose",
            "fallback_for",
            "repair_for",
            "repair_failure_type",
            "quality_gate",
        }
        return {key: value for key, value in (metadata or {}).items() if key in allowed_keys}

    def _elapsed_ms(self, started_at: float) -> int:
        """Return non-negative elapsed milliseconds for runtime observability."""
        return max(round((time.perf_counter() - started_at) * 1000), 0)

    def _extract_usage_metadata(self, completion: Any) -> Dict[str, int]:
        """Extract token usage when the provider returns OpenAI-compatible usage."""
        usage = getattr(completion, "usage", None)
        if usage is None and isinstance(completion, dict):
            usage = completion.get("usage")
        if not usage:
            return {}

        def read_int(field: str) -> Optional[int]:
            value = getattr(usage, field, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(field)
            return int(value) if isinstance(value, (int, float)) else None

        metadata: Dict[str, int] = {}
        for source, target in [
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ]:
            value = read_int(source)
            if value is not None:
                metadata[target] = value
        return metadata

    def _sanitize_error(self, exc: Exception) -> str:
        """清理错误信息，避免泄露API Key"""
        message = str(exc)
        api_key = self.settings.LLM_API_KEY
        if api_key:
            message = message.replace(api_key, "***")
        return message or "LLM调用失败。"

    def _classify_error(self, exc: Exception) -> str:
        """将SDK异常归类为稳定错误类型，供日志和Trace展示。"""
        name = exc.__class__.__name__.lower()
        message = str(exc).lower()
        if "timeout" in name or "timeout" in message or "timed out" in message:
            return "timeout"
        if "ratelimit" in name or "rate_limit" in name or "rate limit" in message or "429" in message:
            return "rate_limit"
        if "authentication" in name or "permission" in name or "unauthorized" in message or "401" in message:
            return "authentication"
        if "connection" in name or "connect" in message or "network" in message:
            return "connection"
        if "badrequest" in name or "invalid" in message or "400" in message:
            return "invalid_request"
        return "provider_error"
