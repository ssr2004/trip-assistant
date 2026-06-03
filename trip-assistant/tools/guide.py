"""
旅行攻略检索工具
基于本地攻略文档提供目的地玩法、景点和行程建议
"""
from pathlib import Path
from typing import Dict, List

from tools.registry import BaseTool


class GuideTool(BaseTool):
    """旅行攻略检索工具"""

    @property
    def name(self) -> str:
        return "retrieve_guide"

    @property
    def description(self) -> str:
        return "检索目的地旅行攻略"

    async def execute(self, query: str = "", destination: str = None, **kwargs) -> Dict:
        """
        检索旅行攻略

        Args:
            query: 攻略查询问题
            destination: 目的地

        Returns:
            标准化攻略检索结果
        """
        documents = self._load_documents()
        results = self._search_documents(query, destination, documents)
        answer = self._build_answer(query, destination, results)

        return {
            "success": True,
            "data": {
                "query": query,
                "destination": destination,
                "answer": answer,
                "sources": results,
            },
            "error": None,
            "metadata": {
                "source": "local_guide_documents",
                "tool": self.name,
            },
        }

    def _load_documents(self) -> List[Dict]:
        """加载本地攻略文档"""
        documents_dir = Path(__file__).resolve().parents[1] / "rag" / "documents" / "guides"
        documents = []

        if not documents_dir.exists():
            return documents

        for filepath in sorted(documents_dir.glob("*.md")):
            documents.append({
                "content": filepath.read_text(encoding="utf-8"),
                "source": str(filepath.relative_to(Path(__file__).resolve().parents[1])),
                "type": "guide",
            })
        return documents

    def _search_documents(self, query: str, destination: str, documents: List[Dict]) -> List[Dict]:
        """基于目的地和关键词检索攻略文档"""
        if not documents:
            return []

        terms = self._extract_terms(query, destination)
        scored_results = []

        for document in documents:
            content = document["content"]
            score = sum(1 for term in terms if term and term in content)
            scored_results.append({
                "content": self._build_excerpt(content, terms),
                "source": document["source"],
                "type": document["type"],
                "score": score,
            })

        scored_results.sort(key=lambda item: item["score"], reverse=True)
        return scored_results[:3]

    def _extract_terms(self, query: str, destination: str) -> List[str]:
        """提取攻略查询关键词"""
        terms = []
        if destination:
            terms.append(destination)

        keywords = [
            "景点", "攻略", "三天", "3天", "路线", "美食", "酒店", "交通", "西湖",
            "灵隐寺", "西溪", "宋城", "预算", "注意事项", "行程",
        ]
        for keyword in keywords:
            if keyword in query:
                terms.append(keyword)
        return terms or [query]

    def _build_excerpt(self, content: str, terms: List[str]) -> str:
        """构建攻略摘要片段"""
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        matched_lines = [line for line in lines if any(term in line for term in terms)]
        excerpt_lines = matched_lines[:8] if matched_lines else lines[:8]
        return "\n".join(excerpt_lines)

    def _build_answer(self, query: str, destination: str, results: List[Dict]) -> str:
        """根据检索结果构建攻略回答"""
        if not results:
            target = destination or "目的地"
            return f"暂未检索到{target}的本地攻略文档，后续可接入更多攻略数据。"

        target = destination or "相关目的地"
        return f"根据本地攻略文档，为您检索到{target}相关建议：\n{results[0]['content']}"
