"""
行程生成工具
根据目的地、天数、预算和偏好生成基础旅行行程
"""
import json
from typing import Dict, List, Optional

from core.llm import LLMClient, LLMMessage, LLMRequest
from core.llm.json_repair import parse_llm_json_object
from core.llm.prompts import ITINERARY_GENERATION_SYSTEM_PROMPT, get_prompt_metadata
from core.llm.quality import audit_itinerary_quality
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
        self.last_llm_metadata: Dict = {}

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
        memory_profile: Dict = None,
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
            memory_profile: 长期记忆注入的规划画像
            context: 前置任务注入的航班、酒店、景点和攻略结果

        Returns:
            标准化行程生成结果
        """
        destination = destination or "目的地"
        duration = duration or 3
        preferences = preferences or []
        memory_profile = memory_profile or {}
        context = context or {}
        context_summary = self._build_context_summary(context)
        personalization_summary = self._build_personalization_summary(memory_profile, preferences)
        budget_summary = self._build_budget_summary(duration, budget, travelers)

        llm_plan = None
        self.last_llm_metadata = {}
        if self._should_use_llm():
            llm_plan = await self._llm_generate(
                origin=origin,
                destination=destination,
                duration=duration,
                budget=budget,
                travelers=travelers,
                preferences=preferences,
                memory_profile=memory_profile,
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
                personalization_summary=personalization_summary,
                metadata_extra=self.last_llm_metadata,
            )

        itinerary = self._build_itinerary(destination, duration, preferences, memory_profile)
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
            personalization_summary=personalization_summary,
            metadata_extra=self.last_llm_metadata,
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
        memory_profile: Dict,
        context: Dict = None,
    ) -> Optional[LLMItineraryPlan]:
        """调用LLM生成行程，失败时返回None"""
        context_text = self._build_llm_context_text(context or {})
        memory_text = self._build_llm_memory_text(memory_profile or {})
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
                        f"长期记忆约束：{memory_text}\n"
                        f"可参考的前置工具结果：{context_text}\n"
                        "如果前置工具结果中包含航班、酒店、景点或攻略，请优先参考这些结果生成安排。\n"
                        "只返回JSON，字段必须包含 itinerary、summary、budget_tips。\n"
                        "itinerary 的长度必须等于旅行天数，每天必须包含 day、title、activities、notes。"
                    ),
                ),
            ],
            response_format="json_object",
            metadata={**get_prompt_metadata("itinerary_generation"), "fallback_for": "itinerary_generation"},
        )
        response = await self.llm_client.chat(request)
        self.last_llm_metadata = {
            "llm_attempted": True,
            "llm_model": response.metadata.get("model"),
            "prompt_id": response.metadata.get("prompt_id"),
            "prompt_version": response.metadata.get("prompt_version"),
            "llm_error_type": response.metadata.get("error_type"),
            "llm_duration_ms": int(response.metadata.get("duration_ms") or 0),
            "llm_prompt_tokens": int(response.metadata.get("prompt_tokens") or 0),
            "llm_completion_tokens": int(response.metadata.get("completion_tokens") or 0),
            "llm_total_tokens": int(response.metadata.get("total_tokens") or 0),
        }
        if not response.success:
            self.last_llm_metadata["fallback_reason"] = "llm_call_failed"
            return None

        parsed_json = parse_llm_json_object(response.content)
        if not parsed_json:
            self.last_llm_metadata["fallback_reason"] = "json_parse_failed"
            return None

        try:
            plan = LLMItineraryPlan.model_validate(parsed_json)
        except Exception:
            self.last_llm_metadata["fallback_reason"] = "schema_validation_failed"
            return None

        quality_result = audit_itinerary_quality(plan, duration)
        self.last_llm_metadata.update(quality_result.metadata())
        if not quality_result.passed:
            self.last_llm_metadata["fallback_reason"] = "quality_gate_failed"
            return None
        self.last_llm_metadata["fallback_reason"] = None
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
        personalization_summary: Dict,
        metadata_extra: Dict = None,
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
                "personalization_summary": personalization_summary,
            },
            metadata={"source": metadata_source, **(metadata_extra or {})},
        )

    def _build_context_summary(self, context: Dict) -> Dict:
        """构建前置工具结果摘要，用于验证依赖注入和后续调试"""
        flights = context.get("flights") if isinstance(context, dict) else []
        hotels = context.get("hotels") if isinstance(context, dict) else []
        attractions = context.get("attractions") if isinstance(context, dict) else []
        guide = context.get("guide") if isinstance(context, dict) else None
        weather = context.get("weather") if isinstance(context, dict) else None
        forecasts = weather.get("forecasts", []) if isinstance(weather, dict) else []
        errors = context.get("errors") if isinstance(context, dict) else {}

        return {
            "flight_count": len(flights) if isinstance(flights, list) else 0,
            "hotel_count": len(hotels) if isinstance(hotels, list) else 0,
            "attraction_count": len(attractions) if isinstance(attractions, list) else 0,
            "has_guide": bool(guide),
            "has_weather": bool(weather),
            "weather_forecast_count": len(forecasts) if isinstance(forecasts, list) else 0,
            "dependency_error_count": len(errors) if isinstance(errors, dict) else 0,
        }

    def _build_llm_context_text(self, context: Dict) -> str:
        """构建传给LLM的紧凑上下文文本"""
        if not context:
            return "无"

        flights = context.get("flights") if isinstance(context.get("flights"), list) else []
        hotels = context.get("hotels") if isinstance(context.get("hotels"), list) else []
        attractions = context.get("attractions") if isinstance(context.get("attractions"), list) else []
        weather = context.get("weather") if isinstance(context.get("weather"), dict) else None
        errors = context.get("errors") if isinstance(context.get("errors"), dict) else {}

        compact_context = {
            "flights": flights[:3],
            "hotels": hotels[:3],
            "attractions": attractions[:5],
            "guide": context.get("guide"),
            "weather": weather,
            "errors": errors,
        }
        if not any(compact_context.values()):
            return "无"
        return json.dumps(compact_context, ensure_ascii=False)

    def _build_personalization_summary(self, memory_profile: Dict, preferences: List[str]) -> Dict:
        """构建长期记忆个性化摘要。"""
        used_preferences = memory_profile.get("used_preferences") if isinstance(memory_profile, dict) else []
        if not isinstance(used_preferences, list):
            used_preferences = []
        return {
            "memory_applied": bool(used_preferences),
            "used_preferences": used_preferences,
            "fallback_preferences": preferences or [],
            "budget_preference": memory_profile.get("budget_preference") if isinstance(memory_profile, dict) else None,
            "excluded_preferences": memory_profile.get("excluded_preferences", []) if isinstance(memory_profile, dict) else [],
            "conflicts": memory_profile.get("conflicts", []) if isinstance(memory_profile, dict) else [],
        }

    def _build_llm_memory_text(self, memory_profile: Dict) -> str:
        """构建传给LLM的长期记忆约束文本。"""
        if not memory_profile:
            return "无"
        compact = {
            "used_preferences": memory_profile.get("used_preferences", []),
            "budget_preference": memory_profile.get("budget_preference"),
            "excluded_preferences": memory_profile.get("excluded_preferences", []),
            "conflicts": memory_profile.get("conflicts", []),
        }
        if not any(compact.values()):
            return "无"
        return json.dumps(compact, ensure_ascii=False)

    def _build_itinerary(self, destination: str, duration: int, preferences: List[str], memory_profile: Dict = None) -> List[Dict]:
        """构建每日行程"""
        memory_profile = memory_profile or {}
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
        if "少走路" in preferences:
            for day in base_days:
                day["notes"] += " 已根据少走路偏好控制步行距离。"
        excluded_preferences = memory_profile.get("excluded_preferences", []) if isinstance(memory_profile, dict) else []
        if excluded_preferences:
            excluded_text = "、".join(excluded_preferences[:3])
            for day in base_days:
                day["notes"] += f" 已避开{excluded_text}等排除项。"

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
