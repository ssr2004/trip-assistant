"""嵌入模型管理。

提供 OpenAI-compatible 文本向量化、内存缓存，以及无 Key/调用失败时的确定性本地降级。
"""
from __future__ import annotations

import hashlib
import math
from typing import Callable, Dict, List, Optional

from app.config import settings


class EmbeddingManager:
    """嵌入模型管理器"""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        openai_client=None,
        openai_client_factory: Optional[Callable[..., object]] = None,
        fallback_dimension: int = 384,
    ):
        """初始化嵌入模型"""
        self.provider = provider or settings.EMBEDDING_PROVIDER
        self.model = model or settings.EMBEDDING_MODEL
        self.api_key = api_key if api_key is not None else settings.EMBEDDING_API_KEY
        self.base_url = base_url or settings.EMBEDDING_BASE_URL
        self.openai_client = openai_client
        self.openai_client_factory = openai_client_factory
        self.fallback_dimension = fallback_dimension
        self.last_backend = "uninitialized"
        self.last_error: Optional[str] = None
        self._cache: Dict[str, List[float]] = {}

    @property
    def available(self) -> bool:
        """是否具备调用真实 embedding 服务的最小配置。"""
        return bool(self.api_key and self.provider in {"openai", "zhipu"})

    def embed(self, text: str) -> List[float]:
        """
        获取文本嵌入向量

        Args:
            text: 输入文本

        Returns:
            嵌入向量
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        normalized_texts = [str(text or "") for text in texts]
        missing = [text for text in normalized_texts if self._cache_key(text) not in self._cache]

        if missing:
            embeddings = self._embed_missing(missing)
            for text, embedding in zip(missing, embeddings):
                self._cache[self._cache_key(text)] = embedding

        return [self._cache[self._cache_key(text)] for text in normalized_texts]

    def _embed_missing(self, texts: List[str]) -> List[List[float]]:
        """向真实服务请求缺失向量，失败时确定性降级。"""
        if self.provider == "openai" and self.api_key:
            try:
                self.last_backend = "openai_compatible"
                return self._openai_embed_batch(texts)
            except Exception as exc:
                self._record_error(exc)
        elif self.provider == "zhipu" and self.api_key:
            try:
                self.last_backend = "zhipu"
                return [self._zhipu_embed(text) for text in texts]
            except Exception as exc:
                self._record_error(exc)

        self.last_backend = "deterministic_fallback"
        return [self._fallback_embed(text) for text in texts]

    def _openai_embed_batch(self, texts: List[str]) -> List[List[float]]:
        """OpenAI-compatible 批量嵌入，兼容百炼 DashScope compatible-mode。"""
        client = self._get_openai_client()
        response = client.embeddings.create(
            model=self.model,
            input=texts[0] if len(texts) == 1 else texts,
        )
        embeddings = [item.embedding for item in response.data]
        if len(embeddings) != len(texts):
            raise RuntimeError("embedding provider returned an unexpected number of vectors")
        return embeddings

    def _get_openai_client(self):
        """延迟创建 OpenAI-compatible 客户端，便于测试替换。"""
        if self.openai_client is not None:
            return self.openai_client
        if self.openai_client_factory is not None:
            self.openai_client = self.openai_client_factory(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            return self.openai_client

        import openai

        self.openai_client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        return self.openai_client

    def _openai_embed(self, text: str) -> List[float]:
        """OpenAI嵌入"""
        return self._openai_embed_batch([text])[0]

    def _zhipu_embed(self, text: str) -> List[float]:
        """智谱AI嵌入"""
        from zhipuai import ZhipuAI

        client = ZhipuAI(api_key=self.api_key)
        response = client.embeddings.create(
            model=self.model or "embedding-3",
            input=text,
        )
        return response.data[0].embedding

    def _fallback_embed(self, text: str) -> List[float]:
        """确定性降级向量，避免无 Key 测试和本地开发出现随机检索结果。"""
        vector = [0.0 for _ in range(self.fallback_dimension)]
        tokens = self._fallback_tokens(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.fallback_dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [round(value / norm, 8) for value in vector]

    def _fallback_tokens(self, text: str) -> List[str]:
        """为中文和英文都生成稳定的轻量 token。"""
        cleaned = "".join(char.lower() for char in str(text or "") if char.strip())
        tokens = []
        for index, char in enumerate(cleaned):
            tokens.append(char)
            if index + 1 < len(cleaned):
                tokens.append(cleaned[index:index + 2])
        return tokens

    def _cache_key(self, text: str) -> str:
        """缓存 Key 不包含真实 API Key。"""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.provider}:{self.model}:{self.base_url}:{digest}"

    def _record_error(self, exc: Exception) -> None:
        """记录脱敏错误，供调试和测试断言使用。"""
        message = str(exc)
        if self.api_key:
            message = message.replace(self.api_key, "***")
        self.last_error = message
