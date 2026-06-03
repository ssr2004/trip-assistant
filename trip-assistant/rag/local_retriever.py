"""
本地Markdown知识检索器
提供政策、攻略等本地文档的加载、关键词检索和结构化片段构建
"""
from pathlib import Path
from typing import Dict, List, Optional

from models.rag import RetrievedChunk


class LocalMarkdownRetriever:
    """本地Markdown文档检索器"""

    def load_documents(
        self,
        directory: Path,
        document_type: str,
        base_dir: Optional[Path] = None,
    ) -> List[Dict]:
        """加载Markdown文档并提取基础元数据"""
        documents = []
        if not directory.exists():
            return documents

        base_dir = base_dir or directory.parent
        for filepath in sorted(directory.glob("*.md")):
            content = filepath.read_text(encoding="utf-8")
            documents.append({
                "title": self._extract_title(content, filepath),
                "content": content,
                "source": self._relative_source(filepath, base_dir),
                "type": document_type,
            })
        return documents

    def search(
        self,
        query: str,
        documents: List[Dict],
        terms: Optional[List[str]] = None,
        top_k: int = 3,
        max_excerpt_lines: int = 6,
    ) -> List[Dict]:
        """基于关键词命中返回结构化检索片段"""
        if not documents:
            return []

        query_terms = self._dedupe_terms(terms or self._fallback_terms(query))
        if not query_terms:
            return []

        scored_results = []
        for document in documents:
            content = document.get("content", "")
            matched_terms = [term for term in query_terms if term in content]
            score = self._score(matched_terms, query_terms)

            if score <= 0 and query:
                matched_terms = self._weak_character_matches(query, content)
                score = min(len(matched_terms) * 0.02, 0.2) if matched_terms else 0.0

            if score <= 0:
                continue

            excerpt = self._build_excerpt(content, matched_terms, max_excerpt_lines)
            scored_results.append(RetrievedChunk(
                content=excerpt,
                source=document.get("source", "本地文档"),
                type=document.get("type", "document"),
                score=round(score, 4),
                title=document.get("title"),
                matched_terms=matched_terms,
                excerpt=excerpt,
            ).model_dump())

        scored_results.sort(key=lambda item: item["score"], reverse=True)
        return scored_results[:top_k]

    def _extract_title(self, content: str, filepath: Path) -> str:
        """从Markdown标题或文件名提取文档标题"""
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                if title:
                    return title
        return filepath.stem

    def _relative_source(self, filepath: Path, base_dir: Path) -> str:
        """生成相对来源路径，方便最终回复展示"""
        try:
            return str(filepath.relative_to(base_dir)).replace("\\", "/")
        except ValueError:
            return str(filepath).replace("\\", "/")

    def _dedupe_terms(self, terms: List[str]) -> List[str]:
        """清洗并去重关键词"""
        cleaned_terms = []
        for term in terms:
            text = str(term).strip() if term else ""
            if text and text not in cleaned_terms:
                cleaned_terms.append(text)
        return cleaned_terms

    def _fallback_terms(self, query: str) -> List[str]:
        """无明确关键词时按短语切分生成兜底检索词"""
        if not query:
            return []
        return [query[index:index + 2] for index in range(0, len(query), 2)]

    def _score(self, matched_terms: List[str], query_terms: List[str]) -> float:
        """按命中关键词占比计算匹配分数"""
        if not query_terms:
            return 0.0
        return len(matched_terms) / len(query_terms)

    def _weak_character_matches(self, query: str, content: str) -> List[str]:
        """关键词未命中时用少量字符弱匹配兜底"""
        matches = []
        for char in query:
            if char.strip() and char in content and char not in matches:
                matches.append(char)
            if len(matches) >= 5:
                break
        return matches

    def _build_excerpt(self, content: str, terms: List[str], max_lines: int) -> str:
        """构建包含命中词的引用片段"""
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return ""

        matched_lines = [line for line in lines if any(term in line for term in terms)]
        excerpt_lines = matched_lines[:max_lines] if matched_lines else lines[:max_lines]
        return "\n".join(excerpt_lines)
