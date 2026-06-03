"""
重排序器
对检索结果进行重排序
"""
from typing import Dict, List


class Reranker:
    """重排序器"""

    def __init__(self):
        """初始化重排序器"""
        pass

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回数量

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        # 计算相关性分数
        scored_docs = []
        for doc in documents:
            score = self._calculate_relevance(query, doc)
            scored_docs.append({
                **doc,
                "rerank_score": score
            })

        # 按分数排序
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored_docs[:top_k]

    def _calculate_relevance(self, query: str, document: Dict) -> float:
        """
        计算相关性分数

        Args:
            query: 查询文本
            document: 文档

        Returns:
            相关性分数
        """
        # 简单的关键词匹配
        query_words = set(query.lower().split())
        doc_words = set(document.get("content", "").lower().split())

        # 计算重叠比例
        if not query_words:
            return 0.0

        overlap = len(query_words & doc_words)
        return overlap / len(query_words)
