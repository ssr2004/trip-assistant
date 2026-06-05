"""
本地Markdown知识检索器
提供政策、攻略等本地文档的加载、分块、混合检索和结构化片段构建
"""
import math
from pathlib import Path
from typing import Dict, List, Optional

from models.rag import RetrievedChunk
from rag.embeddings import EmbeddingManager


class LocalMarkdownRetriever:
    """本地Markdown文档检索器"""

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        enable_vector: bool = True,
        keyword_weight: float = 0.7,
        vector_weight: float = 0.3,
        vector_min_score: float = 0.08,
    ):
        """初始化检索器。

        默认启用向量检索；无真实 Key 时 EmbeddingManager 会使用确定性本地降级。
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.enable_vector = enable_vector
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.vector_min_score = vector_min_score

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
                "document_id": filepath.stem,
                "title": self._extract_title(content, filepath),
                "content": content,
                "source": self._relative_source(filepath, base_dir),
                "type": document_type,
            })
        return documents

    def split_document(self, document: Dict, max_lines: int = 8) -> List[Dict]:
        """按Markdown章节和固定行数切分单篇文档"""
        content = document.get("content", "")
        lines = [line.rstrip() for line in content.splitlines()]
        sections = self._split_sections(lines, document.get("title"))
        chunks = []

        for section_title, section_lines in sections:
            clean_lines = [line.strip() for line in section_lines if line.strip()]
            if not clean_lines:
                continue

            for start in range(0, len(clean_lines), max_lines):
                chunk_lines = clean_lines[start:start + max_lines]
                chunk_index = len(chunks)
                document_id = document.get("document_id") or self._document_id_from_source(document.get("source", "document"))
                chunks.append({
                    "chunk_id": f"{document.get('type', 'document')}-{document_id}-{chunk_index}",
                    "document_id": document_id,
                    "title": document.get("title"),
                    "section": section_title,
                    "content": "\n".join(chunk_lines),
                    "source": document.get("source", "本地文档"),
                    "type": document.get("type", "document"),
                    "chunk_index": chunk_index,
                })

        return chunks

    def split_documents(self, documents: List[Dict], max_lines: int = 8) -> List[Dict]:
        """切分多篇Markdown文档"""
        chunks = []
        for document in documents:
            chunks.extend(self.split_document(document, max_lines=max_lines))
        return chunks

    def search(
        self,
        query: str,
        documents: List[Dict],
        terms: Optional[List[str]] = None,
        top_k: int = 3,
        max_excerpt_lines: int = 6,
    ) -> List[Dict]:
        """基于关键词和向量相似度返回结构化检索片段"""
        if not documents:
            return []

        query_terms = self._dedupe_terms(self._fallback_terms(query) if terms is None else terms)
        if not query_terms and not query:
            return []

        chunks = self._ensure_chunks(documents, max_excerpt_lines)
        vector_scores = self._vector_scores(query, chunks) if self.enable_vector and query else {}
        scored_results = []
        for chunk in chunks:
            content = chunk.get("content", "")
            searchable_content = self._searchable_content(chunk)
            matched_terms = [term for term in query_terms if term in searchable_content]
            keyword_score = self._score(matched_terms, query_terms)

            if keyword_score <= 0 and query:
                matched_terms = self._weak_character_matches(query, searchable_content)
                keyword_score = min(len(matched_terms) * 0.02, 0.2) if matched_terms else 0.0

            vector_score = vector_scores.get(chunk.get("chunk_id", ""), 0.0)
            score = self._hybrid_score(keyword_score, vector_score)
            if keyword_score <= 0 and vector_score < self.vector_min_score:
                continue

            excerpt = self._build_excerpt(content, matched_terms, max_excerpt_lines)
            scored_results.append(RetrievedChunk(
                content=excerpt,
                source=chunk.get("source", "本地文档"),
                type=chunk.get("type", "document"),
                score=round(score, 4),
                title=chunk.get("title"),
                matched_terms=matched_terms,
                excerpt=excerpt,
                chunk_id=chunk.get("chunk_id", ""),
                document_id=chunk.get("document_id", ""),
                section=chunk.get("section"),
                chunk_index=chunk.get("chunk_index", 0),
                keyword_score=round(keyword_score, 4),
                vector_score=round(vector_score, 4),
                retrieval_strategy=self._retrieval_strategy(keyword_score, vector_score),
            ).model_dump())

        scored_results.sort(key=lambda item: (item["score"], -item.get("chunk_index", 0)), reverse=True)
        return scored_results[:top_k]

    def _ensure_chunks(self, documents: List[Dict], max_lines: int) -> List[Dict]:
        """兼容文档级和chunk级输入"""
        if not documents:
            return []
        if all("chunk_id" in item for item in documents):
            return documents
        return self.split_documents(documents, max_lines=max_lines)

    def _split_sections(self, lines: List[str], document_title: Optional[str]) -> List[tuple[Optional[str], List[str]]]:
        """按Markdown二级及以下标题切分章节"""
        sections = []
        current_section = document_title
        current_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("#"):
                heading = stripped.lstrip("#").strip()
                if heading == document_title and not current_lines:
                    continue
                if current_lines:
                    sections.append((current_section, current_lines))
                    current_lines = []
                current_section = heading or current_section
                current_lines.append(stripped)
                continue

            current_lines.append(stripped)

        if current_lines:
            sections.append((current_section, current_lines))
        return sections

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

    def _document_id_from_source(self, source: str) -> str:
        """从来源路径生成文档标识"""
        return Path(source).stem or "document"

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

    def _vector_scores(self, query: str, chunks: List[Dict]) -> Dict[str, float]:
        """计算查询与chunk之间的向量相似度。"""
        if not chunks:
            return {}

        texts = [query] + [self._searchable_content(chunk) for chunk in chunks]
        embeddings = self.embedding_manager.embed_batch(texts)
        query_embedding = embeddings[0]
        scores = {}
        for chunk, chunk_embedding in zip(chunks, embeddings[1:]):
            raw_score = self._cosine_similarity(query_embedding, chunk_embedding)
            scores[chunk.get("chunk_id", "")] = max(0.0, raw_score)
        return scores

    def _hybrid_score(self, keyword_score: float, vector_score: float) -> float:
        """融合关键词和向量分数，保持关键词命中优先。"""
        if keyword_score <= 0:
            return vector_score * self.vector_weight
        return keyword_score * self.keyword_weight + vector_score * self.vector_weight

    def _retrieval_strategy(self, keyword_score: float, vector_score: float) -> str:
        """标记检索来源，便于评测和调试。"""
        if keyword_score > 0 and vector_score >= self.vector_min_score:
            return "hybrid"
        if keyword_score > 0:
            return "keyword"
        return "vector"

    def _searchable_content(self, chunk: Dict) -> str:
        """构建可检索文本。"""
        return "\n".join([
            str(chunk.get("title") or ""),
            str(chunk.get("section") or ""),
            str(chunk.get("content") or ""),
        ])

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度，兼容空向量和维度异常。"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(left * right for left, right in zip(vec1, vec2))
        norm1 = math.sqrt(sum(value * value for value in vec1))
        norm2 = math.sqrt(sum(value * value for value in vec2))
        if not norm1 or not norm2:
            return 0.0
        return dot / (norm1 * norm2)

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
