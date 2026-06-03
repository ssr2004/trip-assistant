"""
LLM客户端封装
提供OpenAI-compatible模型调用和无API Key降级能力
"""
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
        temperature = request.temperature
        if temperature is None:
            temperature = self.settings.LLM_TEMPERATURE

        metadata = self._build_metadata(model=model, fallback=not self.available)

        if not self.available:
            return LLMResponse(
                success=False,
                content="",
                error="LLM_API_KEY未配置，已使用规则逻辑降级。",
                metadata=metadata,
            )

        if not request.messages:
            return LLMResponse(
                success=False,
                content="",
                error="LLM请求缺少messages。",
                metadata=self._build_metadata(model=model),
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

            return LLMResponse(
                success=True,
                content=content,
                error=None,
                metadata={
                    **self._build_metadata(model=model),
                    "response_format": request.response_format,
                },
            )
        except Exception as exc:  # pragma: no cover - 具体异常类型由SDK决定
            return LLMResponse(
                success=False,
                content="",
                error=self._sanitize_error(exc),
                metadata=self._build_metadata(model=model),
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
        }

    def _sanitize_error(self, exc: Exception) -> str:
        """清理错误信息，避免泄露API Key"""
        message = str(exc)
        api_key = self.settings.LLM_API_KEY
        if api_key:
            message = message.replace(api_key, "***")
        return message or "LLM调用失败。"
