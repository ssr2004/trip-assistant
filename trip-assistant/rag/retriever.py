"""
RAG检索器
实现混合检索策略
"""
from typing import Dict, List, Optional
import os

from rag.embeddings import EmbeddingManager
from rag.reranker import Reranker


class RAGRetriever:
    """RAG检索器"""

    def __init__(self):
        """初始化检索器"""
        self.embedding_manager = EmbeddingManager()
        self.reranker = Reranker()
        self.documents = []
        self._load_documents()

    def _load_documents(self):
        """加载文档"""
        documents_path = "rag/documents"

        # 加载政策文档
        policies_path = os.path.join(documents_path, "policies")
        if os.path.exists(policies_path):
            for filename in os.listdir(policies_path):
                if filename.endswith(".md"):
                    filepath = os.path.join(policies_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.documents.append({
                            "content": content,
                            "source": filepath,
                            "type": "policy"
                        })

        # 加载攻略文档
        guides_path = os.path.join(documents_path, "guides")
        if os.path.exists(guides_path):
            for filename in os.listdir(guides_path):
                if filename.endswith(".md"):
                    filepath = os.path.join(guides_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.documents.append({
                            "content": content,
                            "source": filepath,
                            "type": "guide"
                        })

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

        # 向量检索
        vector_results = self._vector_search(query, top_k * 2)

        # BM25检索
        bm25_results = self._bm25_search(query, top_k * 2)

        # 合并结果
        merged_results = self._merge_results(vector_results, bm25_results)

        # 重排序
        reranked_results = self.reranker.rerank(query, merged_results, top_k)

        return reranked_results

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """向量检索"""
        # 计算查询向量
        query_embedding = self.embedding_manager.embed(query)

        # 计算相似度
        results = []
        for doc in self.documents:
            doc_embedding = self.embedding_manager.embed(doc["content"])
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            results.append({
                **doc,
                "score": similarity
            })

        # 排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """BM25检索"""
        # 简化版BM25
        query_words = set(query.split())
        results = []

        for doc in self.documents:
            doc_words = set(doc["content"].split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                results.append({
                    **doc,
                    "score": overlap / len(query_words)
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

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
        """计算余弦相似度"""
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
