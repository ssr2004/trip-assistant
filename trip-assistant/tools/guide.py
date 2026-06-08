"""
旅行攻略检索工具
基于本地攻略文档提供目的地玩法、景点和行程建议
"""
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings
from rag.local_retriever import LocalMarkdownRetriever
from rag.dynamic_guide_generator import TavilyDynamicGuideGenerator
from tools.registry import BaseTool


class GuideTool(BaseTool):
    """旅行攻略检索工具"""

    def __init__(
        self,
        retriever: Optional[LocalMarkdownRetriever] = None,
        guide_generator: Optional[TavilyDynamicGuideGenerator] = None,
        documents_dir: Optional[Path] = None,
        auto_generate: Optional[bool] = None,
    ):
        self.retriever = retriever or LocalMarkdownRetriever()
        self.guide_generator = guide_generator or TavilyDynamicGuideGenerator()
        self.documents_dir = documents_dir
        self.auto_generate = settings.TAVILY_SEARCH_ENABLED if auto_generate is None else auto_generate

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
        destination_documents = self._filter_destination_documents(destination, documents)
        initial_destination_document_count = len(destination_documents)
        results = self._search_documents(query, destination, destination_documents)
        generation_metadata = self._maybe_grow_guide_knowledge(
            destination=destination,
            query=query,
            destination_documents=destination_documents,
            initial_results=results,
        )
        if generation_metadata.get("knowledge_chunks_added", 0) > 0 or generation_metadata.get("reused_existing"):
            documents = self._load_documents()
            destination_documents = self._filter_destination_documents(destination, documents)
            results = self._search_documents(query, destination, destination_documents)
            generation_metadata["rag_second_pass_hit"] = self._is_rag_hit(results, destination)
            generation_metadata["rag_second_pass_top_score"] = self._top_score(results)
        answer = self._build_answer(query, destination, results)
        planning_insights = self._build_planning_insights(destination, results)

        return self.success_result(
            data={
                "query": query,
                "destination": destination,
                "answer": answer,
                "sources": results,
                "planning_insights": planning_insights,
            },
            metadata={
                "source": "local_guide_documents",
                "destination_filter_applied": bool(destination),
                "initial_destination_document_count": initial_destination_document_count,
                "destination_document_count": len(destination_documents),
                **generation_metadata,
            },
        )

    def _load_documents(self) -> List[Dict]:
        """加载本地攻略文档"""
        project_dir = Path(__file__).resolve().parents[1]
        documents_dir = self.documents_dir or project_dir / "rag" / "documents" / "guides"
        return self.retriever.load_documents(
            directory=documents_dir,
            document_type="guide",
            base_dir=project_dir,
        )

    def _maybe_grow_guide_knowledge(
        self,
        destination: str,
        query: str,
        destination_documents: List[Dict],
        initial_results: List[Dict],
    ) -> Dict:
        """Trigger Tavily search when initial RAG retrieval is low-confidence."""
        initial_hit = self._is_rag_hit(initial_results, destination)
        base_metadata = {
            "rag_initial_hit": initial_hit,
            "rag_initial_top_score": self._top_score(initial_results),
            "web_search_attempted": False,
            "web_search_source": None,
            "web_search_result_count": 0,
            "knowledge_chunks_added": 0,
            "knowledge_chunks_deduplicated": 0,
            "knowledge_chunks_filtered_short": 0,
            "knowledge_chunks_filtered_low_quality": 0,
            "rag_second_pass_hit": initial_hit,
            "rag_second_pass_top_score": self._top_score(initial_results),
        }
        if not destination or not self.auto_generate:
            return base_metadata
        if initial_hit and not self._needs_canonical_generated_doc(destination, initial_results):
            return base_metadata

        result = self.guide_generator.ensure_destination_guide(
            destination=destination,
            query=query,
            existing_documents=[] if initial_hit else destination_documents,
        )
        metadata = {
            **base_metadata,
            "web_search_attempted": True,
            "web_search_source": result.source,
            "web_search_result_count": result.source_count,
            "knowledge_chunks_added": result.chunks_added,
            "knowledge_chunks_deduplicated": result.chunks_deduplicated,
            "knowledge_chunks_filtered_short": result.chunks_filtered_short,
            "knowledge_chunks_filtered_low_quality": result.chunks_filtered_low_quality,
            "knowledge_growth_error": result.error,
            "generated_doc_path": str(result.path).replace("\\", "/") if result.path else None,
            "reused_existing": bool(result.reused_existing),
        }
        return metadata

    def _needs_canonical_generated_doc(self, destination: str, results: List[Dict]) -> bool:
        if not results:
            return False
        target_path_getter = getattr(self.guide_generator, "target_path_for", None)
        if not callable(target_path_getter):
            return False
        canonical_path = target_path_getter(destination)
        if canonical_path.exists():
            return False
        top_source = str(results[0].get("source") or "").replace("\\", "/")
        if "generated/" not in top_source or "tavily_generated" not in top_source:
            return False
        return canonical_path.name not in top_source

    def _search_documents(self, query: str, destination: str, documents: List[Dict]) -> List[Dict]:
        """基于目的地和关键词检索攻略文档chunk"""
        if destination and not documents:
            return []
        results = self.retriever.search(
            query=query,
            documents=documents,
            terms=self._extract_terms(query, destination),
            top_k=8,
            max_excerpt_lines=8,
        )
        return self._prioritize_guide_results(results)[:3]

    def _prioritize_guide_results(self, results: List[Dict]) -> List[Dict]:
        """Prefer real guide chunks over generated-document metadata/source sections."""
        non_noise = [result for result in results if not self._is_generated_noise_section(result)]
        guide_chunks = [result for result in non_noise if self._is_guide_chunk(result)]
        if guide_chunks:
            remaining = [result for result in non_noise if result not in guide_chunks]
            return self._dedupe_results_by_content(guide_chunks + remaining)
        return self._dedupe_results_by_content(non_noise)

    def _dedupe_results_by_content(self, results: List[Dict]) -> List[Dict]:
        deduped = []
        seen = set()
        for result in results:
            text = str(result.get("excerpt") or result.get("content") or "").strip()
            key = " ".join(text.split())[:240] or str(result.get("chunk_id") or "")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(result)
        return deduped

    def _is_guide_chunk(self, result: Dict) -> bool:
        section = str(result.get("section") or "")
        return "攻略片段" in section

    def _is_generated_noise_section(self, result: Dict) -> bool:
        source = str(result.get("source") or "")
        if "generated/" not in source or "tavily_generated" not in source:
            return False
        section = str(result.get("section") or "")
        title = str(result.get("title") or "")
        noise_sections = {"元数据", "搜索来源", title}
        if section in noise_sections:
            return True
        text = "\n".join([
            str(result.get("content") or ""),
            str(result.get("excerpt") or ""),
        ])
        shell_markers = [
            "增值电信业务经营许可证",
            "互联网药品信息服务资格证书",
            "违法不良信息举报",
            "沪公网安备",
            "网信算备",
            "自营经营者信息",
            "© 2014-2026 行吟信息科技",
        ]
        return sum(1 for marker in shell_markers if marker in text) >= 2

    def _filter_destination_documents(self, destination: str, documents: List[Dict]) -> List[Dict]:
        """目的地强过滤，避免烟台问题检索到杭州攻略。"""
        if not destination:
            return documents
        target = str(destination).strip()
        if not target:
            return documents
        target_short = target.rstrip("市")
        matched = []
        for document in documents:
            searchable = "\n".join([
                str(document.get("title") or ""),
                str(document.get("source") or ""),
                str(document.get("content") or ""),
            ])
            if target in searchable or target_short in searchable:
                matched.append(document)
        return matched

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

    def _is_rag_hit(self, results: List[Dict], destination: str) -> bool:
        if not results:
            return False
        top_score = self._top_score(results)
        if top_score < settings.RAG_MIN_HIT_SCORE:
            return False
        if destination and not self._result_matches_destination(results[0], destination):
            return False
        content = results[0].get("content") or results[0].get("excerpt") or ""
        return len(str(content)) >= settings.RAG_MIN_CHUNK_LENGTH

    def _top_score(self, results: List[Dict]) -> float:
        if not results:
            return 0.0
        try:
            return float(results[0].get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _result_matches_destination(self, result: Dict, destination: str) -> bool:
        target = str(destination or "").strip()
        target_short = target.rstrip("市")
        searchable = "\n".join([
            str(result.get("title") or ""),
            str(result.get("source") or ""),
            str(result.get("section") or ""),
            str(result.get("content") or ""),
            str(result.get("excerpt") or ""),
        ])
        return bool(target and (target in searchable or target_short in searchable))

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
        excerpt = self._merge_excerpts(results)
        return f"根据本地攻略文档{reference}，为您检索到{target}相关建议：\n{excerpt}"

    def _build_planning_insights(self, destination: str, results: List[Dict]) -> Dict:
        """Convert retrieved guide chunks into compact planning signals for itinerary generation."""
        combined_text = "\n".join([
            str(result.get("excerpt") or result.get("content") or "")
            for result in results
        ])
        clean_text = self._clean_planning_text(combined_text)
        highlights = self._extract_highlights(destination, clean_text, results)
        route_hints = self._extract_sentences(clean_text, ["路线", "行程", "第一天", "第二天", "第三天", "Day", "天"], limit=3)
        food_hints = [
            hint for hint in self._extract_sentences(clean_text, ["美食", "小吃", "夜市", "餐", "吃"], limit=3)
            if "住宿主要" not in hint and "酒店" not in hint
        ]
        lodging_hints = self._extract_sentences(clean_text, ["住宿", "酒店", "住", "商圈", "广场", "交通便利"], limit=2)
        caution_hints = self._extract_sentences(clean_text, ["避坑", "注意", "提前", "门票", "预约", "排队", "不要", "建议"], limit=3)
        source_briefs = self._build_source_briefs(results)
        decision_basis = []
        if highlights:
            decision_basis.append(f"优先围绕{self._join_items(highlights[:4])}组织游览顺序")
        if lodging_hints:
            decision_basis.append("住宿区域参考攻略中提到的交通便利和餐饮集中区域")
        if caution_hints:
            decision_basis.append("将预约、门票和排队等提醒放入注意事项")

        return {
            "destination": destination,
            "highlights": highlights,
            "route_hints": route_hints,
            "food_hints": food_hints,
            "lodging_hints": lodging_hints,
            "caution_hints": caution_hints,
            "decision_basis": decision_basis,
            "source_briefs": source_briefs,
            "source_count": len(results),
        }

    def _clean_planning_text(self, text: str) -> str:
        import re

        text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", str(text or ""))
        text = re.sub(r"\[[^\]]+\.(?:jpg|png|jpeg|gif|webp)\]", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"//\S+", " ", text)
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"(?i)(微信|VX|v信|电话|联系|免费咨询|导游|小西|客服)[^。！？\n]{0,80}", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_highlights(self, destination: str, text: str, results: List[Dict]) -> List[str]:
        candidates = []
        known_places = {
            "郑州": ["河南博物院", "二七广场", "二七纪念塔", "少林寺", "嵩山", "只有河南", "中原福塔", "郑州博物馆", "国贸360", "中原万达"],
            "杭州": ["西湖", "灵隐寺", "西溪湿地", "宋城", "南宋御街", "河坊街"],
            "烟台": ["养马岛", "烟台山", "所城里", "蓬莱阁", "海昌渔人码头", "东炮台"],
            "威海": ["刘公岛", "火炬八街", "威海公园", "那香海", "猫头山", "国际海水浴场"],
        }
        for place in known_places.get(destination or "", []):
            if place in text:
                candidates.append(place)

        for token in self._extract_place_like_tokens(text):
            if token and token not in candidates:
                candidates.append(token)

        return candidates[:8]

    def _extract_place_like_tokens(self, text: str) -> List[str]:
        import re

        suffixes = "博物院|博物馆|公园|广场|纪念塔|寺|山|湿地|古城|老街|步行街|夜市|景区|码头|海水浴场|福塔|万达"
        pattern = rf"[\u4e00-\u9fa5A-Za-z0-9]{{2,10}}(?:{suffixes})"
        blocked = {"攻略片段", "旅游攻略", "自由行路线", "河南郑州", "本地攻略", "搜索摘要"}
        tokens = []
        for match in re.finditer(pattern, str(text or "")):
            token = match.group(0).strip()
            if (
                token in blocked
                or token.startswith(("可以", "如果", "喜欢"))
                or any(word in token for word in ["攻略", "摘要", "片段"])
            ):
                continue
            if token not in tokens:
                tokens.append(token)
        return tokens

    def _extract_sentences(self, text: str, keywords: List[str], limit: int) -> List[str]:
        import re

        sentences = re.split(r"(?<=[。！？!?])\s*|[；;]\s*", str(text or ""))
        selected = []
        blocked_markers = [
            "攻略片段",
            "自己去",
            "导游",
            "小西",
            "免费咨询",
            "强制购物",
            "全程下来",
            "我们当时",
            "图片",
            ".jpg",
            ".png",
            "联系方式",
            "可以找",
            "问题都可以找",
            "门票交通住宿等等方面",
        ]
        for sentence in sentences:
            clean = sentence.strip(" -#\n\t")
            if len(clean) < 8 or len(clean) > 160:
                continue
            if any(marker in clean for marker in blocked_markers):
                continue
            if any(keyword in clean for keyword in keywords):
                if clean not in selected:
                    selected.append(clean)
            if len(selected) >= limit:
                break
        return selected

    def _build_source_briefs(self, results: List[Dict]) -> List[Dict]:
        briefs = []
        seen = set()
        for result in results[:3]:
            title = result.get("title") or "本地攻略文档"
            section = result.get("section")
            if section and ("攻略片段" in str(section) or "自己去" in str(section)):
                section = None
            source = result.get("source")
            key = (title, section, source)
            if key in seen:
                continue
            seen.add(key)
            briefs.append({
                "title": title,
                "section": section,
                "source": source,
            })
        return briefs

    def _join_items(self, items: List[str]) -> str:
        return "、".join([str(item) for item in items if item])

    def _merge_excerpts(self, results: List[Dict], max_sections: int = 2) -> str:
        """Merge a few distinct sections so generated guides expose useful facts, not only source metadata."""
        excerpts = []
        seen_sections = set()
        for result in results:
            section_key = result.get("section") or result.get("chunk_id")
            if section_key in seen_sections:
                continue
            seen_sections.add(section_key)
            text = result.get("excerpt") or result.get("content") or ""
            if text and text not in excerpts:
                excerpts.append(text)
            if len(excerpts) >= max_sections:
                break
        return "\n".join(excerpts)
