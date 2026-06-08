"""Self-growing guide RAG powered by Tavily search."""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings
from rag.embeddings import EmbeddingManager
from rag.local_retriever import LocalMarkdownRetriever
from rag.tavily_client import TavilyClient


@dataclass
class GeneratedGuideResult:
    generated: bool
    path: Optional[Path] = None
    source_count: int = 0
    chunks_added: int = 0
    chunks_deduplicated: int = 0
    chunks_filtered_short: int = 0
    chunks_filtered_low_quality: int = 0
    source: str = "tavily_search"
    error: Optional[str] = None
    reused_existing: bool = False

    @property
    def notes_count(self) -> int:
        """Backward-compatible alias for older metadata callers."""
        return self.source_count


class TavilyDynamicGuideGenerator:
    """Search, clean, dedupe, persist, and reuse destination guide chunks."""

    def __init__(
        self,
        tavily_client: Optional[TavilyClient] = None,
        output_dir: Optional[Path] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        ttl_days: Optional[int] = None,
        min_chunk_length: Optional[int] = None,
        dedupe_threshold: Optional[float] = None,
        max_results: Optional[int] = None,
    ):
        project_dir = Path(__file__).resolve().parents[1]
        self.tavily_client = tavily_client or TavilyClient()
        self.output_dir = output_dir or project_dir / "rag" / "documents" / "guides" / "generated"
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.ttl_days = ttl_days if ttl_days is not None else settings.TAVILY_GUIDE_TTL_DAYS
        self.min_chunk_length = min_chunk_length if min_chunk_length is not None else settings.RAG_MIN_CHUNK_LENGTH
        self.dedupe_threshold = dedupe_threshold if dedupe_threshold is not None else settings.RAG_CHUNK_DEDUPE_THRESHOLD
        self.max_results = max_results if max_results is not None else settings.TAVILY_MAX_RESULTS

    def ensure_destination_guide(
        self,
        destination: str,
        query: str = "",
        preferences: Optional[List[str]] = None,
        existing_documents: Optional[List[Dict]] = None,
    ) -> GeneratedGuideResult:
        if not destination:
            return GeneratedGuideResult(generated=False, error="missing_destination")

        target_path = self._target_path(destination)
        if self._is_reusable_guide(target_path, destination):
            return GeneratedGuideResult(generated=False, path=target_path, reused_existing=True)

        search_query = query or self._build_query(destination, preferences)
        search_result = self._search_with_quality_fallback(destination, search_query)
        if not search_result["success"]:
            metadata = search_result.get("metadata", {}) or {}
            return GeneratedGuideResult(
                generated=False,
                path=target_path,
                error=metadata.get("error_type") or search_result.get("error") or "tavily_failed",
            )

        raw_results = search_result["results"]
        chunks = search_result["chunks"]
        filtered_short = search_result["filtered_short"]
        filtered_low_quality = search_result["filtered_low_quality"]
        kept_chunks, deduped = self._dedupe_chunks(chunks, existing_documents or [])
        if not kept_chunks:
            return GeneratedGuideResult(
                generated=False,
                path=target_path,
                source_count=len(raw_results),
                chunks_filtered_short=filtered_short,
                chunks_filtered_low_quality=filtered_low_quality,
                chunks_deduplicated=deduped,
                error="no_new_chunks",
            )

        markdown = self._build_markdown(destination, search_query, raw_results, kept_chunks)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target_path.write_text(markdown, encoding="utf-8")
        return GeneratedGuideResult(
            generated=True,
            path=target_path,
            source_count=len(raw_results),
            chunks_added=len(kept_chunks),
            chunks_deduplicated=deduped,
            chunks_filtered_short=filtered_short,
            chunks_filtered_low_quality=filtered_low_quality,
        )

    def _search_with_quality_fallback(self, destination: str, search_query: str) -> Dict:
        attempts = [
            {
                "query": search_query,
                "include_domains": None,
            }
        ]
        if settings.TAVILY_INCLUDE_DOMAINS:
            attempts.append({
                "query": f"{destination} 旅游攻略 行程 景点 美食 住宿 交通",
                "include_domains": [],
            })

        last_error = None
        for attempt in attempts:
            search_kwargs = {"max_results": self.max_results}
            if attempt["include_domains"] is not None:
                search_kwargs["include_domains"] = attempt["include_domains"]
            search_result = self.tavily_client.search(attempt["query"], **search_kwargs)
            if not search_result.get("success"):
                last_error = search_result
                continue

            raw_results = search_result.get("data", {}).get("results", [])
            chunks, filtered_short, filtered_low_quality = self._build_candidate_chunks(destination, raw_results)
            if chunks:
                return {
                    "success": True,
                    "results": raw_results,
                    "chunks": chunks,
                    "filtered_short": filtered_short,
                    "filtered_low_quality": filtered_low_quality,
                }
            last_error = {
                "success": False,
                "error": "no_quality_chunks",
                "metadata": {"error_type": "no_quality_chunks"},
                "results": raw_results,
                "filtered_short": filtered_short,
                "filtered_low_quality": filtered_low_quality,
            }

        return last_error or {
            "success": False,
            "error": "tavily_failed",
            "metadata": {"error_type": "tavily_failed"},
        }

    def _build_query(self, destination: str, preferences: Optional[List[str]]) -> str:
        preference_text = " ".join([str(item) for item in preferences or [] if item])
        domain_hint = "小红书" if "xiaohongshu.com" in settings.TAVILY_INCLUDE_DOMAINS else "旅行攻略"
        return f"{destination} 旅游攻略 {domain_hint} {preference_text}".strip()

    def _target_path(self, destination: str) -> Path:
        slug = self._slug(destination)
        digest = hashlib.md5(destination.encode("utf-8")).hexdigest()[:8]
        return self.output_dir / f"{slug}_{digest}_tavily_generated.md"

    def target_path_for(self, destination: str) -> Path:
        """Return the canonical generated guide path for a destination."""
        return self._target_path(destination)

    def _slug(self, destination: str) -> str:
        aliases = {
            "烟台": "yantai",
            "杭州": "hangzhou",
            "郑州": "zhengzhou",
            "太原": "taiyuan",
            "威海": "weihai",
            "青岛": "qingdao",
            "厦门": "xiamen",
            "成都": "chengdu",
            "三亚": "sanya",
        }
        if destination in aliases:
            return aliases[destination]
        ascii_text = re.sub(r"[^a-zA-Z0-9]+", "-", destination).strip("-").lower()
        return ascii_text or "destination"

    def _is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        ttl_seconds = max(int(self.ttl_days or 0), 1) * 86400
        return time.time() - path.stat().st_mtime <= ttl_seconds

    def _is_reusable_guide(self, path: Path, destination: str) -> bool:
        if not self._is_fresh(path):
            return False
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return False
        if "## 攻略片段" not in content:
            return False
        guide_text = content.split("## 攻略片段", 1)[-1]
        return self._has_destination_signal(destination, guide_text) and not self._looks_like_site_shell(guide_text)

    def _build_candidate_chunks(self, destination: str, raw_results: List[Dict]) -> tuple[List[Dict], int, int]:
        chunks = []
        filtered_short = 0
        filtered_low_quality = 0
        for index, result in enumerate(raw_results, start=1):
            content = self._clean_content(result.get("raw_content") or result.get("content") or "")
            title = self._clean_content(result.get("title") or f"{destination}旅行攻略来源{index}")
            summary = self._summarize_content(destination, title, content)
            if len(summary) < self.min_chunk_length:
                filtered_short += 1
                continue
            if not self._is_quality_guide_summary(destination, summary):
                filtered_low_quality += 1
                continue
            chunks.append({
                "title": title,
                "url": result.get("url") or "",
                "score": float(result.get("score") or 0.0),
                "content": summary,
                "source_index": index,
            })
        return chunks, filtered_short, filtered_low_quality

    def _clean_content(self, value: str) -> str:
        text = re.sub(r"<[^>]+>", " ", str(value or ""))
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _summarize_content(self, destination: str, title: str, content: str) -> str:
        text = f"{title}。{content}".strip("。")
        sentences = re.split(r"(?<=[。！？!?])\s*", text)
        selected = []
        keywords = [destination, destination.rstrip("市"), "攻略", "路线", "景点", "美食", "住宿", "交通", "避坑", "行程"]
        for sentence in sentences:
            if not sentence:
                continue
            if any(keyword and keyword in sentence for keyword in keywords):
                selected.append(sentence.strip())
            if len("。".join(selected)) >= 600:
                break
        if not selected:
            selected = sentences[:4]
        return "。".join([item.strip("。") for item in selected if item]).strip()[:1200]

    def _is_quality_guide_summary(self, destination: str, summary: str) -> bool:
        if not self._has_destination_signal(destination, summary):
            return False
        if self._looks_like_site_shell(summary):
            return False
        travel_terms = ["攻略", "路线", "景点", "美食", "住宿", "交通", "行程", "避坑", "门票", "打卡"]
        return any(term in summary for term in travel_terms)

    def _has_destination_signal(self, destination: str, text: str) -> bool:
        target = str(destination or "").strip()
        target_short = target.rstrip("市")
        return bool(target and (target in text or (target_short and target_short in text)))

    def _looks_like_site_shell(self, text: str) -> bool:
        shell_markers = [
            "增值电信业务经营许可证",
            "互联网药品信息服务资格证书",
            "违法不良信息举报",
            "沪公网安备",
            "网信算备",
            "自营经营者信息",
            "© 2014-2026 行吟信息科技",
        ]
        return sum(1 for marker in shell_markers if marker in text) >= 2

    def _dedupe_chunks(self, chunks: List[Dict], existing_documents: List[Dict]) -> tuple[List[Dict], int]:
        if not chunks:
            return [], 0
        existing_chunks = LocalMarkdownRetriever(enable_vector=False).split_documents(existing_documents, max_lines=8)
        existing_texts = [chunk.get("content", "") for chunk in existing_chunks if chunk.get("content")]
        if not existing_texts:
            return chunks, 0

        kept = []
        deduped = 0
        for chunk in chunks:
            if self._is_duplicate(chunk["content"], existing_texts + [item["content"] for item in kept]):
                deduped += 1
            else:
                kept.append(chunk)
        return kept, deduped

    def _is_duplicate(self, text: str, existing_texts: List[str]) -> bool:
        if not text or not existing_texts:
            return False
        embeddings = self.embedding_manager.embed_batch([text, *existing_texts])
        target = embeddings[0]
        for existing in embeddings[1:]:
            if self._cosine_similarity(target, existing) >= self.dedupe_threshold:
                return True
        return False

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        import math

        dot = sum(left * right for left, right in zip(vec1, vec2))
        norm1 = math.sqrt(sum(value * value for value in vec1))
        norm2 = math.sqrt(sum(value * value for value in vec2))
        if not norm1 or not norm2:
            return 0.0
        return dot / (norm1 * norm2)

    def _build_markdown(self, destination: str, query: str, raw_results: List[Dict], chunks: List[Dict]) -> str:
        now = time.strftime("%Y-%m-%d")
        lines = [
            f"# {destination}Tavily搜索攻略摘要",
            "",
            "> 自动生成说明：本文档由 Tavily 搜索结果提炼生成，用于自增长RAG；只保存摘要和结构化片段，不保存大段网页全文。",
            "",
            "## 元数据",
            f"- 目的地：{destination}",
            f"- 查询词：{query}",
            f"- 生成日期：{now}",
            f"- 搜索结果数：{len(raw_results)}",
            f"- 入库切片数：{len(chunks)}",
            "- 数据边界：搜索结果按 Tavily 相关性返回，内容需结合实时POI、交通和天气工具校验。",
            "",
            "## 搜索来源",
        ]
        for index, result in enumerate(raw_results[: self.max_results], start=1):
            title = self._escape_md(result.get("title") or f"来源{index}")
            url = result.get("url") or ""
            score = result.get("score") or 0
            lines.append(f"{index}. [{title}]({url}) - Tavily相关性：{score}")
        lines.append("")
        for chunk in chunks:
            lines.extend([
                f"## 攻略片段 {chunk['source_index']}：{chunk['title']}",
                f"- 来源链接：{chunk['url']}",
                f"- Tavily相关性：{chunk['score']}",
                "",
                chunk["content"],
                "",
            ])
        return "\n".join(lines).strip() + "\n"

    def _escape_md(self, value: str) -> str:
        return str(value or "").replace("[", "【").replace("]", "】")
