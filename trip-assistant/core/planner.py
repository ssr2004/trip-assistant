"""
智能任务规划器
根据结构化意图、实体和缺失槽位生成可执行任务计划
"""
from typing import Dict, List

from models.task import PlanningTask, TaskPlan


class TaskPlanner:
    """任务规划器"""

    def __init__(self):
        """初始化规划器"""
        self.llm_planner_enabled = False

    def plan(self, intent: Dict, context: Dict) -> List[Dict]:
        """
        根据意图规划任务

        Args:
            intent: 意图信息
            context: 上下文信息，包括用户原始输入、RAG上下文、记忆上下文等

        Returns:
            任务列表
        """
        if not intent:
            return []

        if self.llm_planner_enabled:
            return self._llm_plan(intent, context)

        plan = self._template_plan(intent, context)
        return [task.model_dump() for task in plan.tasks]

    def _template_plan(self, intent: Dict, context: Dict) -> TaskPlan:
        """基于模板和规则生成任务计划"""
        intent_type = intent.get("intent", "general_chat")
        entities = intent.get("entities", {}) or {}
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

        return TaskPlan(
            intent=intent_type,
            tasks=[],
            need_user_input=False,
            summary="当前输入暂不需要规划工具任务。",
        )

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

        return TaskPlan(
            intent=intent_type,
            tasks=tasks,
            need_user_input=False,
            summary="目的地明确，规划航班、酒店、景点、攻略检索和行程生成任务。",
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

    def _llm_plan(self, intent: Dict, context: Dict) -> List[Dict]:
        """
        LLM规划器预留接口

        当前阶段先使用模板规划跑通稳定闭环。后续接入LLM后，可以在这里根据复杂自然语言、
        用户偏好和历史记忆生成更灵活的任务计划。
        """
        return [task.model_dump() for task in self._template_plan(intent, context).tasks]
