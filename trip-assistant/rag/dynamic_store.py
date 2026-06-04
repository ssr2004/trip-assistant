"""
动态RAG文档存储
用于接收外部API产生的RAG文档，并提供轻量关键词检索能力
"""
from typing import Dict, List, Optional

from rag.local_retriever import LocalMarkdownRetriever


class DynamicRAGStore:
    """动态RAG文档存储"""

    def __init__(self, retriever: Optional[LocalMarkdownRetriever] = None):
        """初始化动态文档存储"""
        self.retriever = retriever or LocalMarkdownRetriever()
        self._documents: Dict[str, Dict] = {}

    def add_documents(self, documents: List[Dict]) -> None:
        """添加动态RAG文档，按source/document_id去重"""
        for document in documents or []:
            if not isinstance(document, dict):
                continue
            normalized = self._normalize_document(document)
            if not normalized:
                continue
            self._documents[normalized["document_id"]] = normalized

    def list_documents(self) -> List[Dict]:
        """列出已入库的动态RAG文档"""
        return list(self._documents.values())

    def search(self, query: str, top_k: int = 3, terms: Optional[List[str]] = None) -> List[Dict]:
        """检索动态RAG文档"""
        return self.retriever.search(
            query=query,
            documents=self.list_documents(),
            terms=terms,
            top_k=top_k,
            max_excerpt_lines=6,
        )

    def clear(self) -> None:
        """清空动态RAG文档"""
        self._documents.clear()

    def _normalize_document(self, document: Dict) -> Optional[Dict]:
        """规范化外部RAG文档，补齐检索所需字段"""
        title = document.get("title")
        content = document.get("content")
        source = document.get("source")
        if not title or not content or not source:
            return None

        metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
        document_id = document.get("document_id") or metadata.get("record_id") or source
        normalized = dict(document)
        normalized["document_id"] = str(document_id)
        normalized["title"] = str(title)
        normalized["content"] = str(content)
        normalized["source"] = str(source)
        normalized["type"] = str(document.get("type") or metadata.get("source_type") or "external")
        normalized["metadata"] = metadata
        return normalized
