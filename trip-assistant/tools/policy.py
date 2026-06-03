"""
政策检索工具
基于本地政策文档提供退改签、取消等政策查询能力
"""
from pathlib import Path
from typing import Dict, List

from rag.local_retriever import LocalMarkdownRetriever
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

        return self.success_result(
            data={
                "query": query,
                "answer": answer,
                "sources": results,
            },
            metadata={"source": "local_policy_documents"},
        )

    def _load_documents(self) -> List[Dict]:
        """加载本地政策文档"""
        project_dir = Path(__file__).resolve().parents[1]
        documents_dir = project_dir / "rag" / "documents" / "policies"
        return LocalMarkdownRetriever().load_documents(
            directory=documents_dir,
            document_type="policy",
            base_dir=project_dir,
        )

    def _search_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """基于关键词重叠检索政策文档chunk"""
        return LocalMarkdownRetriever().search(
            query=query,
            documents=documents,
            terms=self._extract_terms(query),
            top_k=3,
            max_excerpt_lines=6,
        )

    def _extract_terms(self, query: str) -> List[str]:
        """提取政策查询关键词"""
        terms = []
        keywords = [
            "退票", "退", "退房", "改签", "取消", "酒店", "机票", "航班", "延误",
            "政策", "手续费", "退款", "门票", "景点", "保险", "突发", "疾病", "赔付",
        ]
        for keyword in keywords:
            if keyword in query:
                terms.append(keyword)
        if not terms and query:
            terms.extend([query[index:index + 2] for index in range(0, len(query), 2)])
        return terms

    def _build_answer(self, query: str, results: List[Dict]) -> str:
        """根据检索结果构建政策回答"""
        if not results:
            return "暂未检索到相关政策文档，建议补充更具体的问题。"

        best_result = results[0]
        title = best_result.get("title") or "本地政策文档"
        section = best_result.get("section")
        reference = f"《{title}》"
        if section and section != title:
            reference = f"《{title}》的“{section}”部分"
        best_excerpt = best_result.get("excerpt") or best_result.get("content", "")
        return f"根据本地政策文档{reference}，关于“{query}”可参考以下内容：\n{best_excerpt}"
