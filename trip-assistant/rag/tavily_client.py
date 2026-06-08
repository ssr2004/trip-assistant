"""Tavily Search API client for dynamic travel guide enrichment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests

from app.config import settings


@dataclass
class TavilySearchResult:
    title: str
    url: str
    content: str = ""
    raw_content: str = ""
    score: float = 0.0


class TavilyClient:
    """Small HTTP client around Tavily Search API."""

    SEARCH_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        request_func: Optional[Callable[..., requests.Response]] = None,
        timeout: Optional[int] = None,
    ):
        self.api_key = settings.TAVILY_API_KEY if api_key is None else api_key
        self.request_func = request_func or requests.post
        self.timeout = timeout or settings.EXTERNAL_API_TIMEOUT

    @property
    def available(self) -> bool:
        return bool(settings.TAVILY_SEARCH_ENABLED and self.api_key)

    def search(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        search_depth: Optional[str] = None,
        include_raw_content: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if not settings.TAVILY_SEARCH_ENABLED:
            return self._error("tavily_disabled", "Tavily搜索增强已关闭。")
        if not self.api_key:
            return self._error("missing_api_key", "Tavily API Key未配置。")
        if not query:
            return self._error("missing_query", "Tavily搜索需要query。")

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results or settings.TAVILY_MAX_RESULTS,
            "search_depth": search_depth or settings.TAVILY_SEARCH_DEPTH,
            "include_raw_content": settings.TAVILY_INCLUDE_RAW_CONTENT if include_raw_content is None else include_raw_content,
        }
        domains = include_domains if include_domains is not None else self._configured_domains()
        if domains:
            payload["include_domains"] = domains

        try:
            response = self.request_func(self.SEARCH_URL, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return self._error("request_failed", self._sanitize_error(str(exc)))

        results = [self._normalize_result(item) for item in data.get("results", []) if isinstance(item, dict)]
        return {
            "success": True,
            "data": {
                "query": query,
                "results": [result.__dict__ for result in results],
                "answer": data.get("answer"),
            },
            "error": None,
            "metadata": {
                "provider": "tavily",
                "source": "tavily_search",
                "mock": False,
                "result_count": len(results),
                "include_domains": domains,
            },
        }

    def _normalize_result(self, item: Dict[str, Any]) -> TavilySearchResult:
        return TavilySearchResult(
            title=str(item.get("title") or ""),
            url=str(item.get("url") or ""),
            content=str(item.get("content") or ""),
            raw_content=str(item.get("raw_content") or ""),
            score=float(item.get("score") or 0.0),
        )

    def _configured_domains(self) -> List[str]:
        return [
            domain.strip()
            for domain in (settings.TAVILY_INCLUDE_DOMAINS or "").split(",")
            if domain.strip()
        ]

    def _error(self, error_type: str, message: str) -> Dict[str, Any]:
        return {
            "success": False,
            "data": {"results": []},
            "error": message,
            "metadata": {
                "provider": "tavily",
                "source": "tavily_search",
                "mock": False,
                "error_type": error_type,
                "api_status": "unavailable" if error_type in {"missing_api_key", "tavily_disabled"} else "failed",
            },
        }

    def _sanitize_error(self, message: str) -> str:
        if self.api_key:
            message = message.replace(self.api_key, "***")
        return message
