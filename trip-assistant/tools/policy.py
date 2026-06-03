"""
政策检索工具
基于本地政策文档提供退改签、取消等政策查询能力
"""
from pathlib import Path
from typing import Dict, List

from tools.registry import BaseTool


class PolicyTool(BaseTool):
    """政策检索工具"""

    @property
    def name(self) -> str:
        return "retrieve_policy"

    @property
    def description(self) -> str:
        return "检索旅行政策文档"

    async def execute(self, query: str = "", **kwargs) -> Dict:
        """
        检索政策文档

        Args:
            query: 用户政策问题

        Returns:
            标准化政策检索结果
        """
        documents = self._load_documents()
        results = self._search_documents(query, documents)
        answer = self._build_answer(query, results)

        return {
            "success": True,
            "data": {
                "query": query,
                "answer": answer,
                "sources": results,
            },
            "error": None,
            "metadata": {
                "source": "local_policy_documents",
                "tool": self.name,
            },
        }

    def _load_documents(self) -> List[Dict]:
        """加载本地政策文档"""
        documents_dir = Path(__file__).resolve().parents[1] / "rag" / "documents" / "policies"
        documents = []

        if not documents_dir.exists():
            return documents

        for filepath in sorted(documents_dir.glob("*.md")):
            documents.append({
                "content": filepath.read_text(encoding="utf-8"),
                "source": str(filepath.relative_to(Path(__file__).resolve().parents[1])),
                "type": "policy",
            })
        return documents

    def _search_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """基于关键词重叠检索政策文档"""
        if not documents:
            return []

        query_terms = self._extract_terms(query)
        scored_results = []

        for document in documents:
            content = document["content"]
            score = sum(1 for term in query_terms if term and term in content)
            if score == 0 and query:
                score = 1 if any(char in content for char in query) else 0
            scored_results.append({
                "content": self._build_excerpt(content, query_terms),
                "source": document["source"],
                "type": document["type"],
                "score": score,
            })

        scored_results.sort(key=lambda item: item["score"], reverse=True)
        return scored_results[:3]

    def _extract_terms(self, query: str) -> List[str]:
        """提取政策查询关键词"""
        terms = []
        keywords = ["退票", "改签", "取消", "酒店", "机票", "航班", "政策", "手续费", "退款"]
        for keyword in keywords:
            if keyword in query:
                terms.append(keyword)
        if not terms and query:
            terms.extend([query[index:index + 2] for index in range(0, len(query), 2)])
        return terms

    def _build_excerpt(self, content: str, query_terms: List[str]) -> str:
        """构建文档摘要片段"""
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        matched_lines = [line for line in lines if any(term in line for term in query_terms)]
        excerpt_lines = matched_lines[:6] if matched_lines else lines[:6]
        return "\n".join(excerpt_lines)

    def _build_answer(self, query: str, results: List[Dict]) -> str:
        """根据检索结果构建政策回答"""
        if not results:
            return "暂未检索到相关政策文档，建议补充更具体的问题。"

        best_excerpt = results[0]["content"]
        return f"根据本地政策文档，关于“{query}”可参考以下内容：\n{best_excerpt}"
