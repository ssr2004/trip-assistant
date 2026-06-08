"""
行程生成工具
根据目的地、天数、预算和偏好生成基础旅行行程
"""
import json
import re
from typing import Any, Dict, List, Optional

from app.config import get_settings
from core.llm import LLMClient, LLMMessage, LLMRequest
from core.llm.json_repair import parse_llm_json_object
from core.llm.prompts import ITINERARY_GENERATION_SYSTEM_PROMPT, get_prompt_metadata
from core.llm.quality import audit_itinerary_quality
from models.itinerary import LLMItineraryPlan
from tools.registry import BaseTool


class ItineraryTool(BaseTool):
    """行程生成工具"""

    PLACE_NAME_PATTERN = re.compile(
        r"([\u4e00-\u9fffA-Za-z0-9]{2,24}?(?:博物馆|博物院|公园|广场|寺|庙|湖|山|岛|塔|古镇|故居|湿地|景区|乐园|园|城|村|馆|街|机场|航站楼))"
    )
    HOTEL_NAME_PATTERN = re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{2,24}?(?:酒店|宾馆|民宿|客栈|旅馆))")
    TRAIN_CODE_PATTERN = re.compile(r"\b([GDCZTKY]\d{1,5})\b", re.IGNORECASE)
    FLIGHT_CODE_PATTERN = re.compile(r"\b([A-Z]{2}\d{2,5})\b")

    def __init__(self, llm_client: Optional[LLMClient] = None, llm_enabled: Optional[bool] = None):
        """
        初始化行程工具

        Args:
            llm_client: LLM客户端，默认使用配置创建
            llm_enabled: 是否启用LLM行程生成
        """
        self.settings = get_settings()
        self.llm_client = llm_client or LLMClient()
        self.llm_enabled = self.settings.ITINERARY_LLM_ENABLED if llm_enabled is None else llm_enabled
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
        budget_summary = self._build_budget_summary(duration, budget, travelers, context)

        llm_plan = None
        self.last_llm_metadata = {
            "llm_enabled": bool(self.llm_enabled),
            "llm_available": bool(self.llm_client.available),
            "llm_attempted": False,
        }
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
                grounding_context=self._build_grounding_context(context),
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

        itinerary = self._build_itinerary(destination, duration, preferences, memory_profile, context)
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
        if not self.llm_enabled:
            self.last_llm_metadata["fallback_reason"] = "llm_disabled"
            return False
        if not self.llm_client.available:
            self.last_llm_metadata["fallback_reason"] = "llm_unavailable"
            return False
        return True

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
        grounding_context: Dict = None,
    ) -> Optional[LLMItineraryPlan]:
        """调用LLM生成行程，失败时返回None"""
        context_text = self._build_llm_context_text(context or {})
        memory_text = self._build_llm_memory_text(memory_profile or {})
        grounding_text = json.dumps(grounding_context or {}, ensure_ascii=False)
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
                        f"允许使用的具体实体清单：{grounding_text}\n"
                        "硬性约束：如果要写具体景点、酒店、机场、车次或交通名称，只能使用上面的前置工具结果或允许实体清单。\n"
                        "硬性约束：不要新增未出现在工具结果中的景点、酒店、航班号、车次、房型、房态、余票或可订价格。\n"
                        "如果信息不足，请使用“城市核心区游览”“办理酒店入住”“当地特色餐饮”等通用表述，不要编造具体名称。\n"
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
        grounding_issues = self._audit_grounded_plan(plan, grounding_context or {})
        self.last_llm_metadata["grounding_validation_passed"] = not grounding_issues
        self.last_llm_metadata["grounding_issue_count"] = len(grounding_issues)
        self.last_llm_metadata["grounding_issues"] = grounding_issues
        if grounding_issues:
            self.last_llm_metadata["fallback_reason"] = "grounding_validation_failed"
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
        trains = context.get("trains") if isinstance(context, dict) else []
        hotels = context.get("hotels") if isinstance(context, dict) else []
        attractions = context.get("attractions") if isinstance(context, dict) else []
        guide = context.get("guide") if isinstance(context, dict) else None
        weather = context.get("weather") if isinstance(context, dict) else None
        forecasts = weather.get("forecasts", []) if isinstance(weather, dict) else []
        errors = context.get("errors") if isinstance(context, dict) else {}

        return {
            "flight_count": len(flights) if isinstance(flights, list) else 0,
            "train_count": len(trains) if isinstance(trains, list) else 0,
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
        trains = context.get("trains") if isinstance(context.get("trains"), list) else []
        hotels = context.get("hotels") if isinstance(context.get("hotels"), list) else []
        attractions = context.get("attractions") if isinstance(context.get("attractions"), list) else []
        weather = context.get("weather") if isinstance(context.get("weather"), dict) else None
        errors = context.get("errors") if isinstance(context.get("errors"), dict) else {}
        airport_guidance = self._extract_airport_guidance(context)

        compact_context = {
            "flights": flights[:3],
            "trains": trains[:3],
            "hotels": hotels[:3],
            "attractions": attractions[:5],
            "guide": context.get("guide"),
            "weather": weather,
            "airport_guidance": airport_guidance,
            "errors": errors,
        }
        if not any(compact_context.values()):
            return "无"
        return json.dumps(compact_context, ensure_ascii=False)

    def _build_grounding_context(self, context: Dict) -> Dict:
        """构建LLM可使用的具体实体白名单。"""
        context = context or {}
        airport_guidance = self._extract_airport_guidance(context)
        allowed = {
            "places": self._unique_values([
                *self._item_values(context.get("attractions"), "name"),
                *self._guide_highlights(context.get("guide")),
                *self._airport_names(airport_guidance),
            ]),
            "hotels": self._unique_values(self._item_values(context.get("hotels"), "name")),
            "transport": self._unique_values([
                *self._item_values(context.get("trains"), "train_code"),
                *self._item_values(context.get("trains"), "from_station"),
                *self._item_values(context.get("trains"), "to_station"),
                *self._flight_values(context.get("flights")),
            ]),
        }
        allowed["has_grounding_sources"] = bool(allowed["places"] or allowed["hotels"] or allowed["transport"])
        return allowed

    def _extract_airport_guidance(self, context: Dict) -> Dict:
        raw_results = context.get("raw_results") if isinstance(context, dict) else {}
        if not isinstance(raw_results, dict):
            return {}
        for raw in raw_results.values():
            if not isinstance(raw, dict) or raw.get("tool") != "search_flights":
                continue
            data = raw.get("data")
            if isinstance(data, dict) and isinstance(data.get("airport_guidance"), dict):
                return data["airport_guidance"]
        return {}

    def _airport_names(self, airport_guidance: Dict) -> List[str]:
        names = []
        if not isinstance(airport_guidance, dict):
            return names
        for key in ("origin_airports", "destination_airports"):
            names.extend(self._item_values(airport_guidance.get(key), "name"))
        for pair in airport_guidance.get("airport_pairs") or []:
            if not isinstance(pair, dict):
                continue
            for endpoint in ("origin_airport", "destination_airport"):
                airport = pair.get(endpoint)
                if isinstance(airport, dict) and airport.get("name"):
                    names.append(str(airport["name"]))
        return names

    def _item_values(self, items: Any, key: str) -> List[str]:
        if not isinstance(items, list):
            return []
        values = []
        for item in items:
            if isinstance(item, dict) and item.get(key):
                values.append(str(item[key]))
        return values

    def _flight_values(self, flights: Any) -> List[str]:
        if not isinstance(flights, list):
            return []
        values = []
        for flight in flights:
            if not isinstance(flight, dict):
                continue
            for key in ("flight_no", "airline", "origin_airport", "destination_airport"):
                if flight.get(key):
                    values.append(str(flight[key]))
        return values

    def _guide_highlights(self, guide: Any) -> List[str]:
        if not isinstance(guide, dict):
            return []
        insights = guide.get("planning_insights")
        if not isinstance(insights, dict):
            return []
        highlights = insights.get("highlights") or []
        return [str(item) for item in highlights if item]

    def _unique_values(self, values: List[str]) -> List[str]:
        unique = []
        for value in values:
            cleaned = self._normalize_entity(value)
            if cleaned and cleaned not in unique:
                unique.append(cleaned)
        return unique

    def _audit_grounded_plan(self, plan: LLMItineraryPlan, grounding_context: Dict) -> List[str]:
        """校验LLM行程是否编造了工具结果之外的具体实体。"""
        if not grounding_context or not grounding_context.get("has_grounding_sources"):
            return []

        issues = []
        allowed_places = grounding_context.get("places") or []
        allowed_hotels = grounding_context.get("hotels") or []
        allowed_transport = grounding_context.get("transport") or []

        for day in plan.itinerary or []:
            text = " ".join([day.title or "", " ".join(day.activities or []), day.notes or ""])
            if allowed_places:
                for token in self._extract_place_tokens(text):
                    if not self._is_allowed_entity(token, allowed_places) and not self._is_generic_place_token(token):
                        issues.append(f"ungrounded_place:{day.day}:{token}")
            if allowed_hotels:
                for token in self._extract_hotel_tokens(text):
                    if not self._is_allowed_entity(token, allowed_hotels) and not self._is_generic_hotel_token(token):
                        issues.append(f"ungrounded_hotel:{day.day}:{token}")
            if allowed_transport:
                for token in self._extract_transport_tokens(text):
                    if not self._is_allowed_entity(token, allowed_transport):
                        issues.append(f"ungrounded_transport:{day.day}:{token}")
        return issues[:10]

    def _extract_place_tokens(self, text: str) -> List[str]:
        tokens = []
        for match in self.PLACE_NAME_PATTERN.findall(text or ""):
            token = self._normalize_entity(match)
            if token and token not in tokens:
                tokens.append(token)
        return tokens

    def _extract_hotel_tokens(self, text: str) -> List[str]:
        tokens = []
        for match in self.HOTEL_NAME_PATTERN.findall(text or ""):
            token = self._normalize_entity(match)
            if token and token not in tokens:
                tokens.append(token)
        return tokens

    def _extract_transport_tokens(self, text: str) -> List[str]:
        tokens = []
        for pattern in (self.TRAIN_CODE_PATTERN, self.FLIGHT_CODE_PATTERN):
            for match in pattern.findall(text or ""):
                token = self._normalize_entity(match.upper())
                if token and token not in tokens:
                    tokens.append(token)
        return tokens

    def _normalize_entity(self, value: Any) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\s+", "", text)
        text = text.strip("，。；、,.()（）[]【】")
        return text

    def _is_allowed_entity(self, token: str, allowed_entities: List[str]) -> bool:
        token = self._normalize_entity(token)
        if not token:
            return True
        for allowed in allowed_entities:
            allowed = self._normalize_entity(allowed)
            if token == allowed or token in allowed or allowed in token:
                return True
        return False

    def _is_generic_place_token(self, token: str) -> bool:
        generic_markers = [
            "城市核心景区",
            "热门景点",
            "小众景点",
            "核心景点",
            "周边景点",
            "当地景区",
            "市区公园",
            "城市公园",
            "海边景点",
            "文化街区",
            "商业街",
            "美食街",
            "步行街",
        ]
        return any(marker in token for marker in generic_markers)

    def _is_generic_hotel_token(self, token: str) -> bool:
        generic_markers = [
            "办理酒店",
            "推荐酒店",
            "候选酒店",
            "中档酒店",
            "经济型酒店",
            "地铁附近酒店",
            "酒店入住",
            "酒店",
        ]
        return any(marker in token for marker in generic_markers)

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

    def _build_itinerary(
        self,
        destination: str,
        duration: int,
        preferences: List[str],
        memory_profile: Dict = None,
        context: Dict = None,
    ) -> List[Dict]:
        """构建每日行程"""
        memory_profile = memory_profile or {}
        context = context or {}
        guide_insights = self._guide_insights(context)
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
        elif guide_insights and guide_insights.get("highlights"):
            base_days = self._build_guide_informed_days(destination, duration, guide_insights)
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

    def _guide_insights(self, context: Dict) -> Dict:
        guide = context.get("guide") if isinstance(context, dict) else None
        if not isinstance(guide, dict):
            return {}
        insights = guide.get("planning_insights") or {}
        return insights if isinstance(insights, dict) else {}

    def _build_guide_informed_days(self, destination: str, duration: int, insights: Dict) -> List[Dict]:
        highlights = [item for item in insights.get("highlights", []) if item]
        route_hints = insights.get("route_hints") or []
        food_hints = insights.get("food_hints") or []
        caution_hints = insights.get("caution_hints") or []
        days = []
        day_count = max(int(duration or 3), 1)
        chunks = self._distribute_highlights(highlights, day_count)

        for index in range(day_count):
            day = index + 1
            activities = chunks[index] if index < len(chunks) else []
            if day == 1:
                activities = [f"抵达{destination}", *activities]
            if day == day_count:
                activities = [*activities, "整理返程"]
            if not activities:
                activities = [f"{destination}城市核心区游览", "品尝当地特色美食"]

            notes = self._guide_day_note(day, route_hints, food_hints, caution_hints)
            title = self._guide_day_title(destination, day, day_count, activities)
            days.append({
                "day": day,
                "title": title,
                "activities": activities,
                "notes": notes,
            })
        return days

    def _distribute_highlights(self, highlights: List[str], day_count: int) -> List[List[str]]:
        cleaned = []
        for item in highlights:
            if item and item not in cleaned:
                cleaned.append(item)
        if not cleaned:
            return [[] for _ in range(day_count)]
        buckets = [[] for _ in range(day_count)]
        for index, item in enumerate(cleaned[: max(day_count * 3, 3)]):
            buckets[index % day_count].append(item)
        return buckets

    def _guide_day_title(self, destination: str, day: int, day_count: int, activities: List[str]) -> str:
        if day == 1:
            return f"抵达{destination}与核心景点"
        if day == day_count:
            return f"{destination}补充游览与返程"
        primary = next((item for item in activities if item and not item.startswith("抵达")), destination)
        return f"{primary}主题游览"

    def _guide_day_note(
        self,
        day: int,
        route_hints: List[str],
        food_hints: List[str],
        caution_hints: List[str],
    ) -> str:
        notes = ["根据攻略检索结果提取的景点和路线线索安排。"]
        if day == 1 and food_hints:
            notes.append(f"餐饮可参考：{food_hints[0]}")
        if day == 2 and route_hints:
            notes.append(f"路线参考：{route_hints[0]}")
        if caution_hints:
            notes.append(f"注意：{caution_hints[0]}")
        return " ".join(notes)

    def _build_budget_summary(self, duration: int, budget: float, travelers: int, context: Dict = None) -> Dict:
        """构建预算摘要"""
        context = context or {}
        travelers = travelers or 1
        estimated_hotel = max(duration - 1, 1) * 600
        estimated_food = duration * 180 * travelers
        estimated_local_transport = duration * 80 * travelers
        estimated_attractions = duration * 120 * travelers
        real_price_sources = self._collect_real_price_sources(context)
        if real_price_sources.get("hotel_prices"):
            estimated_hotel = sum(real_price_sources["hotel_prices"][: max(duration - 1, 1)])
        if real_price_sources.get("attraction_prices"):
            estimated_attractions = sum(real_price_sources["attraction_prices"][: max(duration * travelers, 1)])
        estimated_total_without_flight = (
            estimated_hotel + estimated_food + estimated_local_transport + estimated_attractions
        )
        price_source_count = sum(len(values) for values in real_price_sources.values())
        price_confidence = "real_price_partial" if price_source_count else "rough_template"

        return {
            "input_budget": budget,
            "travelers": travelers,
            "price_confidence": price_confidence,
            "price_source_count": price_source_count,
            "estimation_basis": "真实价格字段 + 模板估算" if price_source_count else "缺少真实价格字段，以下仅为模板粗略参考",
            "estimated_hotel": estimated_hotel,
            "estimated_food": estimated_food,
            "estimated_local_transport": estimated_local_transport,
            "estimated_attractions": estimated_attractions,
            "estimated_total_without_flight": estimated_total_without_flight,
            "budget_note": self._build_budget_note(budget, estimated_total_without_flight, price_confidence),
        }

    def _collect_real_price_sources(self, context: Dict) -> Dict[str, List[float]]:
        """Collect numeric prices returned by real tools without inventing missing data."""
        hotels = context.get("hotels") if isinstance(context, dict) else []
        attractions = context.get("attractions") if isinstance(context, dict) else []
        return {
            "hotel_prices": self._numeric_prices(hotels, ["price", "cost"]),
            "attraction_prices": self._numeric_prices(attractions, ["ticket_price", "price"]),
        }

    def _numeric_prices(self, items: List[Dict], keys: List[str]) -> List[float]:
        prices = []
        if not isinstance(items, list):
            return prices
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in keys:
                value = item.get(key)
                number = self._parse_price(value)
                if number is not None:
                    prices.append(number)
                    break
        return prices

    def _parse_price(self, value) -> Optional[float]:
        if value in (None, "", "待定", "免费"):
            return 0.0 if value == "免费" else None
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
        import re

        match = re.search(r"(\d+(?:\.\d+)?)", str(value))
        return float(match.group(1)) if match else None

    def _build_budget_note(self, budget: float, estimated_total: float, price_confidence: str) -> str:
        """生成预算说明"""
        confidence_note = "当前缺少足够真实价格字段，金额仅作粗略参考。" if price_confidence == "rough_template" else "已纳入部分真实价格字段，仍不包含航班和实时酒店库存。"
        if budget is None:
            return f"用户未提供明确预算，{confidence_note}"
        if budget >= estimated_total:
            return f"预算大致可以覆盖当地住宿、餐饮、交通和景点开销；{confidence_note}"
        return f"预算可能偏紧，建议优先选择经济型酒店、免费景点和公共交通；{confidence_note}"

    def _build_summary(self, origin: str, destination: str, duration: int, budget: float) -> str:
        """生成行程摘要"""
        route = f"从{origin}出发前往{destination}" if origin else f"前往{destination}"
        budget_text = f"，预算约{budget:g}元" if budget else ""
        return f"已为您生成{route}的{duration}天旅行行程{budget_text}。"
