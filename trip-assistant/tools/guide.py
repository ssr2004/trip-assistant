"""
旅行攻略检索工具
基于本地攻略文档提供目的地玩法、景点和行程建议
"""
from pathlib import Path
from typing import Dict, List

from rag.local_retriever import LocalMarkdownRetriever
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

        return self.success_result(
            data={
                "query": query,
                "destination": destination,
                "answer": answer,
                "sources": results,
            },
            metadata={"source": "local_guide_documents"},
        )

    def _load_documents(self) -> List[Dict]:
        """加载本地攻略文档"""
        project_dir = Path(__file__).resolve().parents[1]
        documents_dir = project_dir / "rag" / "documents" / "guides"
        return LocalMarkdownRetriever().load_documents(
            directory=documents_dir,
            document_type="guide",
            base_dir=project_dir,
        )

    def _search_documents(self, query: str, destination: str, documents: List[Dict]) -> List[Dict]:
        """基于目的地和关键词检索攻略文档chunk"""
        return LocalMarkdownRetriever().search(
            query=query,
            documents=documents,
            terms=self._extract_terms(query, destination),
            top_k=3,
            max_excerpt_lines=8,
        )

    def _extract_terms(self, query: str, destination: str) -> List[str]:
        """提取攻略查询关键词"""
        terms = []
        if destination:
            terms.append(destination)

        keywords = [
            "杭州", "成都", "厦门", "三亚", "景点", "攻略", "三天", "3天", "路线", "美食",
            "小吃", "火锅", "酒店", "住宿", "交通", "西湖", "灵隐寺", "西溪", "宋城",
            "预算", "注意事项", "行程", "海边", "亲子", "情侣", "拍照", "自然风光", "人文历史",
            "放松", "度假", "沙滩", "鼓浪屿", "亚龙湾", "春熙路", "宽窄巷子",
        ]
        for keyword in keywords:
            if keyword in query:
                terms.append(keyword)
        return terms or [query]

    def _build_answer(self, query: str, destination: str, results: List[Dict]) -> str:
        """根据检索结果构建攻略回答"""
        if not results:
            target = destination or "目的地"
            return f"暂未检索到{target}的本地攻略文档，后续可接入更多攻略数据。"

        target = destination or "相关目的地"
        best_result = results[0]
        title = best_result.get("title") or "本地攻略文档"
        section = best_result.get("section")
        reference = f"《{title}》"
        if section and section != title:
            reference = f"《{title}》的“{section}”部分"
        excerpt = best_result.get("excerpt") or best_result.get("content", "")
        return f"根据本地攻略文档{reference}，为您检索到{target}相关建议：\n{excerpt}"
