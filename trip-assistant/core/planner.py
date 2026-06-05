"""
智能任务规划器
根据结构化意图、实体和缺失槽位生成可执行任务计划
"""
from typing import Any, Dict, List, Optional
import json

from core.llm import LLMClient, LLMMessage, LLMRequest
from core.llm.json_repair import parse_llm_json_object
from core.llm.prompts import PLANNER_FALLBACK_SYSTEM_PROMPT
from models.task import PlanningTask, TaskPlan


class TaskPlanner:
    """任务规划器"""

    ALLOWED_TASK_TYPES = {"ask_user", "tool_call", "recommend_destination", "generate_itinerary", "dynamic_rag_query", "revise_itinerary"}
    ALLOWED_TOOLS = {
        "search_flights",
        "search_hotels",
        "search_attractions",
        "retrieve_policy",
        "retrieve_guide",
        "generate_itinerary",
        "get_weather_forecast",
    }

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        llm_planner_enabled: Optional[bool] = None,
        llm_planner_mode: Optional[str] = None,
        llm_planner_complexity_threshold: int = 3,
    ):
        """
        初始化规划器

        Args:
            llm_client: LLM客户端，默认使用配置创建
            llm_planner_enabled: 是否启用LLM规划fallback
        """
        self.llm_client = llm_client or LLMClient()
        if llm_planner_mode is None:
            llm_planner_mode = "off" if llm_planner_enabled is False else "auto"
        self.llm_planner_mode = self._normalize_planner_mode(llm_planner_mode)
        self.llm_planner_enabled = self.llm_planner_mode != "off"
        self.llm_planner_complexity_threshold = max(int(llm_planner_complexity_threshold or 1), 1)
        self.last_plan_metadata: Dict = self._base_plan_metadata()

    def plan(self, intent: Dict, context: Dict) -> List[Dict]:
        """
        根据意图规划任务（同步模板规划）

        Args:
            intent: 意图信息
            context: 上下文信息，包括用户原始输入、RAG上下文、记忆上下文等

        Returns:
            任务列表
        """
        if not intent:
            self.last_plan_metadata = {**self._base_plan_metadata(), "planner_mode": "template", "skip_reason": "empty_intent"}
            return []

        plan = self._template_plan(intent, context)
        route_metadata = self._llm_plan_route(intent, context, plan)
        self.last_plan_metadata = {
            **self._base_plan_metadata(),
            **route_metadata,
            "planner_mode": "template",
            "llm_planner_attempted": False,
            "llm_planner_adopted": False,
            "skip_reason": "sync_template_plan",
            "template_task_count": len(plan.tasks),
        }
        return [task.model_dump() for task in plan.tasks]

    async def plan_async(self, intent: Dict, context: Dict) -> List[Dict]:
        """
        根据意图异步规划任务，模板优先，复杂场景可尝试LLM fallback

        Args:
            intent: 意图信息
            context: 上下文信息

        Returns:
            任务列表
        """
        if not intent:
            self.last_plan_metadata = {**self._base_plan_metadata(), "planner_mode": "template", "skip_reason": "empty_intent"}
            return []

        template_plan = self._template_plan(intent, context)
        skip_reason = self._llm_plan_skip_reason(intent, context, template_plan)
        route_metadata = self._llm_plan_route(intent, context, template_plan)
        if skip_reason:
            self.last_plan_metadata = {
                **self._base_plan_metadata(),
                **route_metadata,
                "planner_mode": "template",
                "llm_planner_attempted": False,
                "llm_planner_adopted": False,
                "skip_reason": skip_reason,
                "template_task_count": len(template_plan.tasks),
            }
            return [task.model_dump() for task in template_plan.tasks]

        llm_plan = await self._llm_plan(intent, context, template_plan)
        if llm_plan:
            self.last_plan_metadata = {
                **self.last_plan_metadata,
                "planner_mode": "llm",
                "llm_planner_adopted": True,
                "llm_task_count": len(llm_plan.tasks),
                "template_task_count": len(template_plan.tasks),
            }
            return [task.model_dump() for task in llm_plan.tasks]
        self.last_plan_metadata = {
            **self.last_plan_metadata,
            "planner_mode": "template",
            "llm_planner_adopted": False,
            "template_task_count": len(template_plan.tasks),
        }
        return [task.model_dump() for task in template_plan.tasks]

    def _template_plan(self, intent: Dict, context: Dict) -> TaskPlan:
        """基于模板和规则生成任务计划"""
        intent_type = intent.get("intent", "general_chat")
        entities = dict(intent.get("entities", {}) or {})
        entities["preferences"] = self._merge_preferences(
            entities.get("preferences", []) or [],
            context.get("memory", {}).get("preferences", {}) or context.get("memory", {}).get("user_preferences", {}),
        )
        missing_slots = intent.get("missing_slots", []) or []
        user_query = context.get("query", "")

        if intent_type == "travel_plan" and "destination" in missing_slots:
            return self._build_travel_plan(intent_type, entities, user_query)

        if missing_slots:
            return self._build_followup_plan(intent_type, entities, missing_slots, intent)

        if intent_type == "travel_plan":
            return self._build_travel_plan(intent_type, entities, user_query)

        if intent_type == "flight_search":
            return self._build_single_tool_plan(
                intent_type=intent_type,
                tool="search_flights",
                name="搜索航班",
                params={
                    "origin": entities.get("origin"),
                    "destination": entities.get("destination"),
                    "date": entities.get("departure_date"),
                },
                reason="用户需要查询两个城市之间的航班信息。",
            )

        if intent_type == "hotel_search":
            return self._build_single_tool_plan(
                intent_type=intent_type,
                tool="search_hotels",
                name="搜索酒店",
                params={
                    "location": entities.get("destination"),
                    "checkin_date": entities.get("departure_date"),
                    "checkout_date": entities.get("return_date"),
                    "budget": entities.get("budget"),
                    "preferences": entities.get("preferences", []),
                },
                reason="用户需要查询目的地酒店信息。",
            )

        if intent_type == "attraction_search":
            return self._build_single_tool_plan(
                intent_type=intent_type,
                tool="search_attractions",
                name="搜索景点",
                params={
                    "location": entities.get("destination"),
                    "keywords": entities.get("preferences", []),
                },
                reason="用户需要查询目的地景点或玩法信息。",
            )

        if intent_type == "policy_query":
            return self._build_single_tool_plan(
                intent_type=intent_type,
                tool="retrieve_policy",
                name="检索政策文档",
                params={"query": user_query},
                reason="用户询问退改签、取消或旅行政策，需要从政策文档中检索答案。",
            )

        if intent_type == "guide_query":
            return self._build_single_tool_plan(
                intent_type=intent_type,
                tool="retrieve_guide",
                name="检索旅行攻略",
                params={"query": user_query, "destination": entities.get("destination")},
                reason="用户询问目的地玩法、攻略、美食或路线，需要从攻略文档中检索答案。",
            )

        if intent_type == "dynamic_knowledge_query":
            return self._build_dynamic_rag_plan(intent_type, user_query)

        if intent_type == "itinerary_revision":
            return self._build_itinerary_revision_plan(intent_type, user_query)

        if intent_type == "weather_query":
            return self._build_single_tool_plan(
                intent_type=intent_type,
                tool="get_weather_forecast",
                name="查询天气预报",
                params={"city": entities.get("destination"), "days": entities.get("duration") or 3},
                reason="用户询问目的地天气，需要查询天气预报并给出旅行建议。",
            )

        return TaskPlan(
            intent=intent_type,
            tasks=[],
            need_user_input=False,
            summary="当前输入暂不需要规划工具任务。",
        )

    def _merge_preferences(self, current_preferences: List[str], memory_preferences: Dict) -> List[str]:
        """合并当前输入偏好和长期记忆偏好，当前输入优先保留顺序"""
        merged = []
        for preference in current_preferences or []:
            if preference and preference not in merged:
                merged.append(preference)

        for preference in self._flatten_memory_preferences(memory_preferences):
            if preference and preference not in merged:
                merged.append(preference)
        return merged

    def _flatten_memory_preferences(self, memory_preferences: Dict) -> List[str]:
        """将长期记忆偏好结构拉平成任务规划可用的偏好列表"""
        if not isinstance(memory_preferences, dict):
            return []

        preference_fields = [
            "travel_styles",
            "hotel_preferences",
            "transport_preferences",
            "attraction_preferences",
            "food_preferences",
            "raw_preferences",
        ]
        flattened = []
        for field in preference_fields:
            values = memory_preferences.get(field, [])
            if not isinstance(values, list):
                continue
            for value in values:
                if value and value not in flattened:
                    flattened.append(value)

        budget_preference = memory_preferences.get("budget_preference")
        if budget_preference and budget_preference not in flattened:
            flattened.append(budget_preference)
        return flattened

    def _build_followup_plan(
        self,
        intent_type: str,
        entities: Dict,
        missing_slots: List[str],
        intent: Dict,
    ) -> TaskPlan:
        """构建追问任务计划"""
        question = intent.get("followup_question") or self._fallback_question(missing_slots, entities)
        task = PlanningTask(
            task_id="ask_user_1",
            task_type="ask_user",
            name="补充缺失信息",
            priority=1,
            params={
                "missing_slots": missing_slots,
                "question": question,
            },
            reason="当前旅行需求缺少关键信息，继续调用工具前需要先向用户追问。",
        )
        return TaskPlan(
            intent=intent_type,
            tasks=[task],
            need_user_input=True,
            summary="需要用户补充关键信息后再继续规划。",
        )

    def _build_travel_plan(self, intent_type: str, entities: Dict, user_query: str) -> TaskPlan:
        """构建完整旅行规划任务"""
        destination = entities.get("destination")
        origin = entities.get("origin")
        duration = entities.get("duration") or 3
        budget = entities.get("budget")
        preferences = entities.get("preferences", [])

        if not destination:
            task = PlanningTask(
                task_id="recommend_destination_1",
                task_type="recommend_destination",
                name="推荐旅行目的地",
                priority=1,
                params={
                    "origin": origin,
                    "duration": duration,
                    "budget": budget,
                    "preferences": preferences,
                    "query": user_query,
                },
                reason="用户没有明确目的地，需要先根据预算、时间和偏好推荐候选城市。",
            )
            return TaskPlan(
                intent=intent_type,
                tasks=[task],
                need_user_input=False,
                summary="用户目的地不明确，先推荐候选目的地。",
            )

        tasks = [
            PlanningTask(
                task_id="search_flights_1",
                task_type="tool_call",
                name="搜索航班",
                priority=1,
                tool="search_flights",
                params={
                    "origin": origin,
                    "destination": destination,
                    "date": entities.get("departure_date"),
                    "budget": budget,
                    "preferences": preferences,
                },
                reason="完整旅行规划需要先获取出发地到目的地的交通方案。",
            ),
            PlanningTask(
                task_id="search_hotels_1",
                task_type="tool_call",
                name="搜索酒店",
                priority=2,
                tool="search_hotels",
                params={
                    "location": destination,
                    "checkin_date": entities.get("departure_date"),
                    "checkout_date": entities.get("return_date"),
                    "budget": budget,
                    "preferences": preferences,
                },
                reason="完整旅行规划需要结合预算和偏好推荐住宿。",
            ),
            PlanningTask(
                task_id="search_attractions_1",
                task_type="tool_call",
                name="搜索景点",
                priority=3,
                tool="search_attractions",
                params={
                    "location": destination,
                    "keywords": preferences,
                    "duration": duration,
                },
                reason="完整旅行规划需要根据目的地和偏好筛选可游玩的景点。",
            ),
            PlanningTask(
                task_id="retrieve_guide_1",
                task_type="tool_call",
                name="检索旅行攻略",
                priority=4,
                tool="retrieve_guide",
                params={
                    "query": self._build_guide_query(destination, duration, preferences),
                    "destination": destination,
                },
                reason="需要从攻略文档中获取目的地玩法、交通和注意事项。",
            ),
            PlanningTask(
                task_id="generate_itinerary_1",
                task_type="generate_itinerary",
                name="生成旅行行程",
                priority=5,
                tool="generate_itinerary",
                params={
                    "origin": origin,
                    "destination": destination,
                    "duration": duration,
                    "budget": budget,
                    "travelers": entities.get("travelers"),
                    "preferences": preferences,
                },
                depends_on=[
                    "search_flights_1",
                    "search_hotels_1",
                    "search_attractions_1",
                    "retrieve_guide_1",
                ],
                reason="需要综合航班、酒店、景点、攻略和偏好生成最终行程。",
            ),
        ]

        # Replace the static itinerary task with a dependency-aware version.
        tasks = [task for task in tasks if task.task_id != "generate_itinerary_1"]

        weather_aware = self._has_weather_constraint(user_query, preferences)
        if weather_aware:
            weather_task = PlanningTask(
                task_id="get_weather_forecast_1",
                task_type="tool_call",
                name="查询天气预报",
                priority=5,
                tool="get_weather_forecast",
                params={
                    "city": destination,
                    "days": duration,
                    "departure_date": entities.get("departure_date"),
                    "preferences": preferences,
                },
                reason="用户提到天气、雨天或备选路线约束，完整行程生成前需要先获取目的地天气信息。",
            )
            tasks.append(weather_task)

        itinerary_dependencies = [
            "search_flights_1",
            "search_hotels_1",
            "search_attractions_1",
            "retrieve_guide_1",
        ]
        if any(task.task_id == "get_weather_forecast_1" for task in tasks):
            itinerary_dependencies.append("get_weather_forecast_1")
        itinerary_priority = max(task.priority for task in tasks) + 1
        tasks.append(
            PlanningTask(
                task_id="generate_itinerary_1",
                task_type="generate_itinerary",
                name="鐢熸垚鏃呰琛岀▼",
                priority=itinerary_priority,
                tool="generate_itinerary",
                params={
                    "origin": origin,
                    "destination": destination,
                    "duration": duration,
                    "budget": budget,
                    "travelers": entities.get("travelers"),
                    "preferences": preferences,
                    "weather_aware": weather_aware,
                },
                depends_on=itinerary_dependencies,
                reason="闇€瑕佺患鍚堣埅鐝€侀厭搴椼€佹櫙鐐广€佹敾鐣ュ拰鍋忓ソ鐢熸垚鏈€缁堣绋嬨€?",
            )
        )
        return TaskPlan(
            intent=intent_type,
            tasks=tasks,
            need_user_input=False,
            summary="目的地明确，规划航班、酒店、景点、攻略检索和行程生成任务。",
        )

    def _has_weather_constraint(self, user_query: str, preferences: List[str]) -> bool:
        """Return whether itinerary planning should explicitly fetch weather context."""
        text = " ".join([user_query or "", *[str(preference) for preference in preferences or []]])
        weather_keywords = [
            "天气",
            "下雨",
            "雨天",
            "阴雨",
            "备选",
            "室内",
            "weather",
            "rain",
            "backup",
        ]
        return any(keyword in text for keyword in weather_keywords)

    def _build_dynamic_rag_plan(self, intent_type: str, user_query: str) -> TaskPlan:
        """构建动态外部知识检索任务"""
        task = PlanningTask(
            task_id="dynamic_rag_query_1",
            task_type="dynamic_rag_query",
            name="检索动态外部知识",
            priority=1,
            params={"query": user_query},
            reason="用户基于刚才推荐的外部景点继续追问，需要从会话动态RAG文档中检索答案。",
        )
        return TaskPlan(
            intent=intent_type,
            tasks=[task],
            need_user_input=False,
            summary="规划任务：检索动态外部知识。",
        )

    def _build_itinerary_revision_plan(self, intent_type: str, user_query: str) -> TaskPlan:
        """构建行程修订任务"""
        task = PlanningTask(
            task_id="revise_itinerary_1",
            task_type="revise_itinerary",
            name="修订旅行行程",
            priority=1,
            params={"query": user_query},
            reason="用户基于上一轮行程提出调整，需要读取会话行程上下文并修订每日安排。",
        )
        return TaskPlan(
            intent=intent_type,
            tasks=[task],
            need_user_input=False,
            summary="规划任务：修订旅行行程。",
        )

    def _build_single_tool_plan(
        self,
        intent_type: str,
        tool: str,
        name: str,
        params: Dict,
        reason: str,
    ) -> TaskPlan:
        """构建单工具调用任务计划"""
        task = PlanningTask(
            task_id=f"{tool}_1",
            task_type="tool_call",
            name=name,
            priority=1,
            tool=tool,
            params=params,
            reason=reason,
        )
        return TaskPlan(
            intent=intent_type,
            tasks=[task],
            need_user_input=False,
            summary=f"规划任务：{name}。",
        )

    def _build_guide_query(self, destination: str, duration: int, preferences: List[str]) -> str:
        """构建攻略检索查询"""
        preference_text = " ".join(preferences) if preferences else ""
        return f"{destination}{duration}天旅行攻略 {preference_text}".strip()

    def _fallback_question(self, missing_slots: List[str], entities: Dict) -> str:
        """缺失追问兜底问题"""
        first_slot = missing_slots[0] if missing_slots else "信息"
        destination = entities.get("destination") or "目的地"
        origin = entities.get("origin") or "出发地"
        questions = {
            "origin": f"请问您准备从哪个城市出发去{destination}？",
            "destination": "请问您计划去哪个城市旅行？如果还没确定，我也可以先帮您推荐目的地。",
            "departure_date": f"请问您计划什么时候从{origin}出发？",
            "duration": f"请问您计划在{destination}玩几天？",
        }
        return questions.get(first_slot, "请补充一下旅行需求中的关键信息。")

    def _should_use_llm_plan(self, intent: Dict, context: Dict, template_plan: TaskPlan) -> bool:
        """判断是否尝试LLM规划"""
        return self._llm_plan_skip_reason(intent, context, template_plan) is None

    def _llm_plan_skip_reason(self, intent: Dict, context: Dict, template_plan: TaskPlan) -> Optional[str]:
        """Return why LLM planning should be skipped, or None when it should run."""
        return self._llm_plan_route(intent, context, template_plan).get("skip_reason")

    def _llm_plan_route(self, intent: Dict, context: Dict, template_plan: TaskPlan) -> Dict[str, Any]:
        """Return sanitized auto-routing metadata for LLM task planning."""
        signals = self._planning_complexity_signals(intent, context)
        score = len(signals)
        metadata: Dict[str, Any] = {
            "planner_mode_config": self.llm_planner_mode,
            "llm_planner_enabled": self.llm_planner_enabled,
            "llm_planner_available": bool(self.llm_client.available),
            "llm_planner_auto_route": self.llm_planner_mode == "auto",
            "llm_planner_complexity_score": score,
            "llm_planner_complexity_threshold": self.llm_planner_complexity_threshold,
            "llm_planner_complexity_signals": signals,
            "llm_planner_route_decision": "skip",
            "skip_reason": None,
        }
        if self.llm_planner_mode == "off":
            return {**metadata, "skip_reason": "mode_off"}
        if not self.llm_client.available:
            return {**metadata, "skip_reason": "llm_unavailable"}
        if template_plan.need_user_input:
            return {**metadata, "skip_reason": "need_user_input"}
        if intent.get("intent") not in {"travel_plan", "itinerary_revision"}:
            return {**metadata, "skip_reason": "unsupported_intent"}
        if self.llm_planner_mode == "always":
            return {**metadata, "llm_planner_route_decision": "attempt_llm"}
        if score < self.llm_planner_complexity_threshold:
            return {**metadata, "skip_reason": "not_complex_enough"}
        return {**metadata, "llm_planner_route_decision": "attempt_llm"}

    def _planning_complexity_signals(self, intent: Dict, context: Dict) -> List[str]:
        """Detect product-level complexity signals without exposing planner modes to users."""
        query = context.get("query", "") or ""
        entities = intent.get("entities", {}) or {}
        preferences = entities.get("preferences", []) or []
        confidence = intent.get("confidence", 1.0)
        keyword_groups = {
            "multi_objective": ["同时", "兼顾", "既要", "又要", "平衡", "权衡"],
            "pace_constraint": ["不要太累", "别太累", "轻松", "慢节奏", "老人", "孩子", "亲子"],
            "route_constraint": ["路线", "顺路", "少走路", "地铁", "交通", "距离"],
            "weather_contingency": ["天气", "下雨", "雨天", "备选", "如果"],
            "budget_constraint": ["预算", "人均", "费用", "价格", "以内", "控制"],
            "accommodation_constraint": ["住宿", "酒店", "民宿", "附近", "商圈"],
        }
        signals: List[str] = []
        if len(query.strip()) >= 36:
            signals.append("long_query")
        if len(preferences) >= 3:
            signals.append("multiple_preferences")
        if entities.get("budget"):
            signals.append("budget_entity")
        if entities.get("travelers"):
            signals.append("traveler_entity")
        if isinstance(confidence, (int, float)) and confidence < 0.65:
            signals.append("low_confidence")
        for signal, keywords in keyword_groups.items():
            if any(keyword in query for keyword in keywords):
                signals.append(signal)
        return signals

    def _normalize_planner_mode(self, mode: str) -> str:
        """Normalize external config into an internal routing mode."""
        normalized = (mode or "auto").strip().lower()
        aliases = {
            "enabled": "auto",
            "true": "auto",
            "on": "auto",
            "disabled": "off",
            "false": "off",
            "manual": "always",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"auto", "off", "always"}:
            return "auto"
        return normalized

    def _base_plan_metadata(self) -> Dict:
        """Build default sanitized planner metadata for trace observability."""
        return {
            "planner_mode": "template",
            "planner_mode_config": self.llm_planner_mode,
            "llm_planner_enabled": bool(self.llm_planner_enabled),
            "llm_planner_available": bool(self.llm_client.available),
            "llm_planner_auto_route": self.llm_planner_mode == "auto",
            "llm_planner_complexity_score": 0,
            "llm_planner_complexity_threshold": self.llm_planner_complexity_threshold,
            "llm_planner_complexity_signals": [],
            "llm_planner_route_decision": "skip",
            "llm_planner_attempted": False,
            "llm_planner_adopted": False,
        }

    async def _llm_plan(self, intent: Dict, context: Dict, template_plan: TaskPlan) -> Optional[TaskPlan]:
        """调用LLM生成任务计划，失败时返回None"""
        self.last_plan_metadata = {
            **self._base_plan_metadata(),
            **self._llm_plan_route(intent, context, template_plan),
            "planner_mode": "template",
            "llm_planner_attempted": True,
            "llm_planner_adopted": False,
        }
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=PLANNER_FALLBACK_SYSTEM_PROMPT),
                LLMMessage(
                    role="user",
                    content=(
                        "请基于下面的意图和上下文生成任务计划JSON。\n"
                        f"用户输入：{context.get('query', '')}\n"
                        f"结构化意图：{json.dumps(intent, ensure_ascii=False)}\n"
                        f"模板规划参考：{template_plan.model_dump_json() }"
                    ),
                ),
            ],
            response_format="json_object",
            metadata={"fallback_for": "task_plan"},
        )
        response = await self.llm_client.chat(request)
        self.last_plan_metadata = {
            **self.last_plan_metadata,
            "llm_planner_model": response.metadata.get("model"),
            "llm_planner_error_type": response.metadata.get("error_type"),
            "llm_planner_duration_ms": int(response.metadata.get("duration_ms") or 0),
            "llm_planner_prompt_tokens": int(response.metadata.get("prompt_tokens") or 0),
            "llm_planner_completion_tokens": int(response.metadata.get("completion_tokens") or 0),
            "llm_planner_total_tokens": int(response.metadata.get("total_tokens") or 0),
        }
        if not response.success:
            self.last_plan_metadata = {
                **self.last_plan_metadata,
                "fallback_reason": "llm_call_failed",
            }
            return None

        parsed_json = parse_llm_json_object(response.content)
        if not parsed_json:
            self.last_plan_metadata = {
                **self.last_plan_metadata,
                "fallback_reason": "json_parse_failed",
            }
            return None

        try:
            plan = TaskPlan.model_validate(parsed_json)
        except Exception:
            self.last_plan_metadata = {
                **self.last_plan_metadata,
                "fallback_reason": "schema_validation_failed",
            }
            return None

        if not self._validate_llm_plan(plan):
            self.last_plan_metadata = {
                **self.last_plan_metadata,
                "fallback_reason": "unsafe_plan",
            }
            return None
        self.last_plan_metadata = {
            **self.last_plan_metadata,
            "fallback_reason": None,
        }
        return plan

    def _validate_llm_plan(self, plan: TaskPlan) -> bool:
        """校验LLM规划结果是否安全可执行"""
        task_ids = set()
        for task in plan.tasks:
            if not task.task_id or task.task_id in task_ids:
                return False
            task_ids.add(task.task_id)

            if task.task_type not in self.ALLOWED_TASK_TYPES:
                return False

            if task.priority is None:
                return False

            if task.task_type in {"tool_call", "generate_itinerary"}:
                if task.tool not in self.ALLOWED_TOOLS:
                    return False

            if task.task_type in {"ask_user", "recommend_destination"} and task.tool:
                return False

            for dependency in task.depends_on:
                if dependency not in task_ids:
                    return False

        return True
