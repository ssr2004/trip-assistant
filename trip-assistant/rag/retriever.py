"""
RAG检索器
实现本地知识库混合检索策略
"""
from pathlib import Path
from typing import Dict, List

from rag.embeddings import EmbeddingManager
from rag.local_retriever import LocalMarkdownRetriever
from rag.reranker import Reranker
from rag.vector_store import InMemoryVectorStore


def _persistent_store_path(namespace: str) -> Path:
    """持久化向量库的存储路径，按命名空间隔离不同检索器，避免并发写冲突。"""
    project_dir = Path(__file__).resolve().parents[1]
    return project_dir / "data" / "vector_store" / f"{namespace}_index"


def _build_persistent_store(embedding_manager: EmbeddingManager, namespace: str) -> InMemoryVectorStore:
    """构建持久化向量库；embedding 后端变化时按 model_signature 自动失效重建。"""
    signature = f"{embedding_manager.provider}:{embedding_manager.model}"
    return InMemoryVectorStore(
        persist_path=_persistent_store_path(namespace),
        model_signature=signature,
    )


class RAGRetriever:
    """RAG检索器"""

    def __init__(self, embedding_manager: EmbeddingManager = None):
        """初始化检索器"""
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.vector_store = _build_persistent_store(self.embedding_manager, namespace="local")
        self.local_retriever = LocalMarkdownRetriever(
            embedding_manager=self.embedding_manager,
            vector_store=self.vector_store,
        )
        self.reranker = Reranker()
        self.documents = []
        self._load_documents()

    def _load_documents(self):
        """加载文档"""
        project_dir = Path(__file__).resolve().parents[1]
        documents_path = project_dir / "rag" / "documents"
        self.documents = []
        self.documents.extend(self.local_retriever.load_documents(
            documents_path / "policies",
            "policy",
            project_dir,
        ))
        self.documents.extend(self.local_retriever.load_documents(
            documents_path / "guides",
            "guide",
            project_dir,
        ))

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            相关文档列表
        """
        if not self.documents:
            return []

        results = self.local_retriever.search(query=query, documents=self.documents, top_k=top_k * 2)
        return self.reranker.rerank(query, results, top_k)

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """兼容旧接口的向量检索入口。"""
        return self.local_retriever.search(query=query, documents=self.documents, terms=[], top_k=top_k)

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """兼容旧接口的关键词检索入口。"""
        retriever = LocalMarkdownRetriever(
            embedding_manager=self.embedding_manager,
            enable_vector=False,
        )
        return retriever.search(query=query, documents=self.documents, top_k=top_k)

    def _merge_results(self, vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        """合并检索结果"""
        merged = {}

        for result in vector_results:
            key = result["source"]
            if key not in merged:
                merged[key] = result
            else:
                merged[key]["score"] = max(merged[key]["score"], result["score"])

        for result in bm25_results:
            key = result["source"]
            if key not in merged:
                merged[key] = result
            else:
                merged[key]["score"] = max(merged[key]["score"], result["score"])

        return list(merged.values())

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """兼容旧接口的余弦相似度计算。"""
        return self.local_retriever._cosine_similarity(vec1, vec2)
