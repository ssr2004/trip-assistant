"""
行程生成工具
根据目的地、天数、预算和偏好生成基础旅行行程
"""
import json
from typing import Dict, List, Optional

from core.llm import LLMClient, LLMMessage, LLMRequest
from core.llm.json_repair import parse_llm_json_object
from core.llm.prompts import ITINERARY_GENERATION_SYSTEM_PROMPT
from models.itinerary import LLMItineraryPlan
from tools.registry import BaseTool


class ItineraryTool(BaseTool):
    """行程生成工具"""

    def __init__(self, llm_client: Optional[LLMClient] = None, llm_enabled: bool = False):
        """
        初始化行程工具

        Args:
            llm_client: LLM客户端，默认使用配置创建
            llm_enabled: 是否启用LLM行程生成
        """
        self.llm_client = llm_client or LLMClient()
        self.llm_enabled = llm_enabled

    @property
    def name(self) -> str:
        return "generate_itinerary"

    @property
    def description(self) -> str:
        return "生成结构化旅行行程"

    async def execute(
        self,
        origin: str = None,
        destination: str = None,
        duration: int = 3,
        budget: float = None,
        travelers: int = None,
        preferences: List[str] = None,
        context: Dict = None,
        **kwargs,
    ) -> Dict:
        """
        生成基础行程

        Args:
            origin: 出发地
            destination: 目的地
            duration: 旅行天数
            budget: 预算
            travelers: 出行人数
            preferences: 用户偏好
            context: 前置任务注入的航班、酒店、景点和攻略结果

        Returns:
            标准化行程生成结果
        """
        destination = destination or "目的地"
        duration = duration or 3
        preferences = preferences or []
        context = context or {}
        context_summary = self._build_context_summary(context)
        budget_summary = self._build_budget_summary(duration, budget, travelers)

        llm_plan = None
        if self._should_use_llm():
            llm_plan = await self._llm_generate(
                origin=origin,
                destination=destination,
                duration=duration,
                budget=budget,
                travelers=travelers,
                preferences=preferences,
                context=context,
            )

        if llm_plan:
            return self._build_result(
                origin=origin,
                destination=destination,
                duration=duration,
                budget=budget,
                travelers=travelers,
                preferences=preferences,
                itinerary=[day.model_dump() for day in llm_plan.itinerary],
                budget_summary={
                    **budget_summary,
                    "budget_note": llm_plan.budget_tips or budget_summary["budget_note"],
                },
                summary=llm_plan.summary or self._build_summary(origin, destination, duration, budget),
                generation_mode="llm",
                metadata_source="llm_itinerary_generator",
                context_summary=context_summary,
            )

        itinerary = self._build_itinerary(destination, duration, preferences)
        return self._build_result(
            origin=origin,
            destination=destination,
            duration=duration,
            budget=budget,
            travelers=travelers,
            preferences=preferences,
            itinerary=itinerary,
            budget_summary=budget_summary,
            summary=self._build_summary(origin, destination, duration, budget),
            generation_mode="template",
            metadata_source="template_itinerary_generator",
            context_summary=context_summary,
        )

    def _should_use_llm(self) -> bool:
        """判断是否使用LLM行程生成"""
        return self.llm_enabled and self.llm_client.available

    async def _llm_generate(
        self,
        origin: str,
        destination: str,
        duration: int,
        budget: float,
        travelers: int,
        preferences: List[str],
        context: Dict = None,
    ) -> Optional[LLMItineraryPlan]:
        """调用LLM生成行程，失败时返回None"""
        context_text = self._build_llm_context_text(context or {})
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=ITINERARY_GENERATION_SYSTEM_PROMPT),
                LLMMessage(
                    role="user",
                    content=(
                        "请根据以下旅行需求生成结构化行程JSON。\n"
                        f"出发地：{origin or '未知'}\n"
                        f"目的地：{destination}\n"
                        f"旅行天数：{duration}\n"
                        f"预算：{budget if budget is not None else '未知'}\n"
                        f"出行人数：{travelers if travelers is not None else '未知'}\n"
                        f"用户偏好：{', '.join(preferences) if preferences else '无'}\n"
                        f"可参考的前置工具结果：{context_text}\n"
                        "如果前置工具结果中包含航班、酒店、景点或攻略，请优先参考这些结果生成安排。\n"
                        "只返回JSON，字段必须包含 itinerary、summary、budget_tips。\n"
                        "itinerary 的长度必须等于旅行天数，每天必须包含 day、title、activities、notes。"
                    ),
                ),
            ],
            response_format="json_object",
            metadata={"fallback_for": "itinerary_generation"},
        )
        response = await self.llm_client.chat(request)
        if not response.success:
            return None

        parsed_json = parse_llm_json_object(response.content)
        if not parsed_json:
            return None

        try:
            plan = LLMItineraryPlan.model_validate(parsed_json)
        except Exception:
            return None

        if len(plan.itinerary) != duration:
            return None
        if any(day.day != index for index, day in enumerate(plan.itinerary, start=1)):
            return None
        return plan

    def _build_result(
        self,
        origin: str,
        destination: str,
        duration: int,
        budget: float,
        travelers: int,
        preferences: List[str],
        itinerary: List[Dict],
        budget_summary: Dict,
        summary: str,
        generation_mode: str,
        metadata_source: str,
        context_summary: Dict,
    ) -> Dict:
        """构建标准化行程工具结果"""
        return self.success_result(
            data={
                "origin": origin,
                "destination": destination,
                "duration": duration,
                "budget": budget,
                "travelers": travelers,
                "preferences": preferences,
                "itinerary": itinerary,
                "budget_summary": budget_summary,
                "summary": summary,
                "generation_mode": generation_mode,
                "context_summary": context_summary,
            },
            metadata={"source": metadata_source},
        )

    def _build_context_summary(self, context: Dict) -> Dict:
        """构建前置工具结果摘要，用于验证依赖注入和后续调试"""
        flights = context.get("flights") if isinstance(context, dict) else []
        hotels = context.get("hotels") if isinstance(context, dict) else []
        attractions = context.get("attractions") if isinstance(context, dict) else []
        guide = context.get("guide") if isinstance(context, dict) else None
        errors = context.get("errors") if isinstance(context, dict) else {}

        return {
            "flight_count": len(flights) if isinstance(flights, list) else 0,
            "hotel_count": len(hotels) if isinstance(hotels, list) else 0,
            "attraction_count": len(attractions) if isinstance(attractions, list) else 0,
            "has_guide": bool(guide),
            "dependency_error_count": len(errors) if isinstance(errors, dict) else 0,
        }

    def _build_llm_context_text(self, context: Dict) -> str:
        """构建传给LLM的紧凑上下文文本"""
        if not context:
            return "无"

        flights = context.get("flights") if isinstance(context.get("flights"), list) else []
        hotels = context.get("hotels") if isinstance(context.get("hotels"), list) else []
        attractions = context.get("attractions") if isinstance(context.get("attractions"), list) else []
        errors = context.get("errors") if isinstance(context.get("errors"), dict) else {}

        compact_context = {
            "flights": flights[:3],
            "hotels": hotels[:3],
            "attractions": attractions[:5],
            "guide": context.get("guide"),
            "errors": errors,
        }
        if not any(compact_context.values()):
            return "无"
        return json.dumps(compact_context, ensure_ascii=False)

    def _build_itinerary(self, destination: str, duration: int, preferences: List[str]) -> List[Dict]:
        """构建每日行程"""
        if destination == "杭州":
            base_days = [
                {
                    "day": 1,
                    "title": "抵达杭州与西湖初体验",
                    "activities": ["抵达杭州", "办理酒店入住", "游览西湖", "湖滨步行街晚餐"],
                    "notes": "第一天以适应节奏为主，不安排过高强度行程。",
                },
                {
                    "day": 2,
                    "title": "灵隐寺与西溪湿地",
                    "activities": ["灵隐寺", "飞来峰", "西溪国家湿地公园", "品尝杭帮菜"],
                    "notes": "兼顾历史文化和自然风光，适合完整游玩日。",
                },
                {
                    "day": 3,
                    "title": "宋城或城市休闲后返程",
                    "activities": ["宋城", "杭州特色伴手礼", "整理返程"],
                    "notes": "根据返程时间选择宋城演出或轻松购物。",
                },
            ]
        else:
            base_days = [
                {
                    "day": 1,
                    "title": f"抵达{destination}与城市初体验",
                    "activities": [f"抵达{destination}", "办理酒店入住", "游览城市核心景区", "品尝当地特色美食"],
                    "notes": "第一天以抵达和轻松游览为主。",
                },
                {
                    "day": 2,
                    "title": f"{destination}深度游览",
                    "activities": ["热门景点游览", "当地文化体验", "特色餐饮", "夜间休闲"],
                    "notes": "安排较完整的景点和文化体验。",
                },
                {
                    "day": 3,
                    "title": f"{destination}休闲与返程",
                    "activities": ["轻松游览", "购买伴手礼", "整理返程"],
                    "notes": "最后一天降低强度，预留返程时间。",
                },
            ]

        if "慢节奏" in preferences:
            for day in base_days:
                day["notes"] += " 已根据慢节奏偏好减少高强度安排。"

        if duration <= len(base_days):
            return base_days[:duration]

        itinerary = base_days[:]
        for day_index in range(len(base_days) + 1, duration + 1):
            itinerary.append({
                "day": day_index,
                "title": f"{destination}自由探索日",
                "activities": ["根据兴趣选择小众景点", "体验当地美食", "休闲散步"],
                "notes": "该日作为弹性安排，可根据天气和体力调整。",
            })
        return itinerary

    def _build_budget_summary(self, duration: int, budget: float, travelers: int) -> Dict:
        """构建预算摘要"""
        travelers = travelers or 1
        estimated_hotel = max(duration - 1, 1) * 600
        estimated_food = duration * 180 * travelers
        estimated_local_transport = duration * 80 * travelers
        estimated_attractions = duration * 120 * travelers
        estimated_total_without_flight = (
            estimated_hotel + estimated_food + estimated_local_transport + estimated_attractions
        )

        return {
            "input_budget": budget,
            "travelers": travelers,
            "estimated_hotel": estimated_hotel,
            "estimated_food": estimated_food,
            "estimated_local_transport": estimated_local_transport,
            "estimated_attractions": estimated_attractions,
            "estimated_total_without_flight": estimated_total_without_flight,
            "budget_note": self._build_budget_note(budget, estimated_total_without_flight),
        }

    def _build_budget_note(self, budget: float, estimated_total: float) -> str:
        """生成预算说明"""
        if budget is None:
            return "用户未提供明确预算，当前为基础估算，后续可结合航班和酒店价格细化。"
        if budget >= estimated_total:
            return "预算基本可以覆盖当地住宿、餐饮、交通和景点开销，机票费用需另行结合航班结果计算。"
        return "预算可能偏紧，建议优先选择经济型酒店、免费景点和公共交通。"

    def _build_summary(self, origin: str, destination: str, duration: int, budget: float) -> str:
        """生成行程摘要"""
        route = f"从{origin}出发前往{destination}" if origin else f"前往{destination}"
        budget_text = f"，预算约{budget:g}元" if budget else ""
        return f"已为您生成{route}的{duration}天旅行行程{budget_text}。"
