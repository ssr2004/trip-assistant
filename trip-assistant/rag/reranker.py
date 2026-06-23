"""重排序器（可插拔）。

检索阶段用 BM25+向量混合做粗排（高召回），重排阶段用更重的模型做精排（高精度）。
重排是 RAG 提升精度的最高性价比手段（cross-encoder rerank）。

- ``KeywordReranker``：基于 token 重叠的轻量重排（CJK 感知），默认，无需模型。
- ``CrossEncoderReranker``：基于 cross-encoder（如 BGE-reranker-v2-m3，多语言/中文友好）
  的语义重排；首次使用时延迟加载 sentence-transformers 模型，下载失败或依赖缺失时
  优雅降级到 KeywordReranker，保证链路不中断。
- ``Reranker``：``KeywordReranker`` 的别名，向后兼容旧调用方。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from rag.bm25 import tokenize as bm25_tokenize

logger = logging.getLogger(__name__)


class BaseReranker:
    """重排器基类。"""

    name = "base"

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        if not documents:
            return []
        scored = [{**doc, "rerank_score": self._score(query, doc)} for doc in documents]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]

    def _score(self, query: str, document: Dict) -> float:
        raise NotImplementedError


class KeywordReranker(BaseReranker):
    """基于 token 重叠的轻量重排（CJK 感知）。

    原 Reranker 用 ``split()`` 分词，对中文整句视为一个 token 导致重排失效；
    这里改用与 BM25 一致的 CJK 字+bigram 分词。
    """

    name = "keyword"

    def _score(self, query: str, document: Dict) -> float:
        query_tokens = set(bm25_tokenize(query))
        if not query_tokens:
            return 0.0
        doc_tokens = set(bm25_tokenize(self._doc_text(document)))
        overlap = len(query_tokens & doc_tokens)
        return overlap / len(query_tokens)

    @staticmethod
    def _doc_text(document: Dict) -> str:
        return "\n".join([
            str(document.get("title") or ""),
            str(document.get("section") or ""),
            str(document.get("content") or ""),
        ])


class CrossEncoderReranker(BaseReranker):
    """基于 cross-encoder 模型的语义重排。"""

    name = "cross_encoder"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        fallback: Optional[BaseReranker] = None,
    ):
        self.model_name = model_name
        self.fallback = fallback or KeywordReranker()
        self._model = None
        self._load_attempted = False

    def _ensure_model(self):
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        except Exception as exc:
            logger.warning(
                "CrossEncoder 模型加载失败，降级到 %s 重排：%s",
                self.fallback.name,
                exc,
            )
            self._model = None

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        if not documents:
            return []
        self._ensure_model()
        if self._model is None:
            return self.fallback.rerank(query, documents, top_k)
        pairs = [(query, KeywordReranker._doc_text(doc)) for doc in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: float(x[1]), reverse=True)
        return [
            {**doc, "rerank_score": float(score), "reranker": self.name}
            for doc, score in ranked[:top_k]
        ]

    def _score(self, query: str, document: Dict) -> float:
        return self.fallback._score(query, document)


class DashScopeReranker(BaseReranker):
    """基于阿里云百炼 DashScope rerank API 的语义重排。

    复用百炼 API Key（与 embedding 的 EMBEDDING_API_KEY 同账号同 key），
    无需下载本地模型，按次调用。默认 qwen3-rerank（gte-rerank 已于 2026-05 下线）。
    未配置 key 或调用失败时优雅降级到 fallback（默认 KeywordReranker）。
    """

    name = "dashscope_rerank"
    DEFAULT_MODEL = "qwen3-rerank"
    DEFAULT_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        fallback: Optional[BaseReranker] = None,
        instruct: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.api_key = api_key or _resolve_dashscope_api_key()
        self.model = model or self.DEFAULT_MODEL
        self.base_url = base_url or self.DEFAULT_URL
        self.fallback = fallback or KeywordReranker()
        self.instruct = instruct
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        if not documents:
            return []
        if not self.available:
            return self.fallback.rerank(query, documents, top_k)
        try:
            import httpx

            payload = {
                "model": self.model,
                "query": query,
                "documents": [KeywordReranker._doc_text(doc) for doc in documents],
                "top_n": top_k,
                "return_documents": False,
            }
            if self.instruct:
                payload["instruct"] = self.instruct
            response = httpx.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            results = (response.json() or {}).get("results", [])
            ordered: List[Dict] = []
            for item in results:
                index = item.get("index")
                if index is None or index < 0 or index >= len(documents):
                    continue
                ordered.append({
                    **documents[index],
                    "rerank_score": float(item.get("relevance_score", 0.0)),
                    "reranker": self.name,
                })
            return ordered[:top_k] if ordered else self.fallback.rerank(query, documents, top_k)
        except Exception as exc:
            logger.warning("DashScope rerank 失败，降级到 %s：%s", self.fallback.name, exc)
            return self.fallback.rerank(query, documents, top_k)

    def _score(self, query: str, document: Dict) -> float:
        return self.fallback._score(query, document)


def _resolve_dashscope_api_key() -> str:
    """复用百炼 key：优先专用 RERANK_API_KEY，否则沿用 EMBEDDING_API_KEY（同账号）。"""
    try:
        from app.config import settings

        return (
            getattr(settings, "RERANK_API_KEY", "")
            or getattr(settings, "EMBEDDING_API_KEY", "")
            or getattr(settings, "DASHSCOPE_API_KEY", "")
        )
    except Exception:
        return ""


class Reranker(KeywordReranker):
    """默认重排器（KeywordReranker 别名，向后兼容）。"""
    pass
