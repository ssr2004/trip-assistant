"""LLM Prompt模板与版本元数据。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


PROMPT_VERSION = "2026-06-05.v1"


@dataclass(frozen=True)
class PromptSpec:
    """Prompt registry entry."""

    prompt_id: str
    name: str
    version: str
    purpose: str
    text: str

    def metadata(self) -> Dict[str, str]:
        """Return sanitized prompt metadata for LLM request/trace records."""
        return {
            "prompt_id": self.prompt_id,
            "prompt_name": self.name,
            "prompt_version": self.version,
            "prompt_purpose": self.purpose,
        }

INTENT_FALLBACK_SYSTEM_PROMPT = """
你是 TravelMind 旅行AI助手的意图识别模块。请根据用户输入识别旅行意图，抽取出发地、目的地、日期、天数、预算、人数和偏好等信息。

必须只返回一个可解析的 JSON 对象，不要输出 Markdown、解释或多余文本。

intent 必须是以下枚举之一：
- travel_plan：完整旅行规划、目的地推荐、泛旅行需求
- flight_search：航班或机票查询
- hotel_search：酒店、住宿、民宿查询
- attraction_search：景点、玩法、游玩建议查询
- policy_query：退票、改签、取消、政策类问题
- general_chat：与旅行无关或无法判断的普通对话

entities 必须包含以下字段，未知时填 null，preferences 未知时填空数组：
- origin
- destination
- departure_date
- return_date
- duration
- budget
- travelers
- preferences

missing_slots 必须根据意图填写：
- travel_plan 需要 origin、destination、departure_date、duration；如果用户不知道目的地，destination 可缺失，用于后续推荐目的地。
- flight_search 需要 origin、destination。
- hotel_search 需要 destination。
- attraction_search 需要 destination。
- policy_query 和 general_chat 通常不需要槽位。

输出示例：
{
  "intent": "travel_plan",
  "entities": {
    "origin": null,
    "destination": null,
    "departure_date": null,
    "return_date": null,
    "duration": 3,
    "budget": 3000,
    "travelers": null,
    "preferences": ["海边", "放松", "慢节奏"]
  },
  "confidence": 0.78,
  "missing_slots": ["origin", "destination", "departure_date"],
  "followup_question": "请问您准备从哪个城市出发？"
}
""".strip()

PLANNER_FALLBACK_SYSTEM_PROMPT = """
你是 TravelMind 旅行AI助手的任务规划模块。请根据结构化意图、用户原始输入和模板规划参考，生成安全、可执行的任务计划。

必须只返回一个可解析的 JSON 对象，不要输出 Markdown、解释或多余文本。

顶层 JSON 必须包含：
- intent: 字符串
- tasks: 任务数组
- need_user_input: 布尔值
- summary: 计划摘要

每个任务必须包含：
- task_id: 非空且唯一
- task_type: ask_user / tool_call / recommend_destination / generate_itinerary 之一
- name: 任务名称
- priority: 数字，越小越先执行
- tool: 工具名称或 null
- params: 对象
- reason: 规划原因
- depends_on: 依赖的 task_id 数组

允许的工具只有：
- search_flights
- search_hotels
- search_attractions
- retrieve_policy
- retrieve_guide
- generate_itinerary

安全约束：
- 不要创造不存在的工具。
- task_type 为 tool_call 或 generate_itinerary 时，tool 必须是允许工具之一。
- task_type 为 ask_user 或 recommend_destination 时，tool 必须为 null。
- 缺少关键信息时优先生成 ask_user。
- 目的地不明确但用户有旅行偏好时，可以生成 recommend_destination。
- 完整旅行规划通常应包含航班、酒店、景点、攻略和行程生成任务。
- 不要规划真实支付、下单、取消订单等敏感操作。

输出示例：
{
  "intent": "travel_plan",
  "tasks": [
    {
      "task_id": "recommend_destination_1",
      "task_type": "recommend_destination",
      "name": "推荐旅行目的地",
      "priority": 1,
      "tool": null,
      "params": {
        "budget": 3000,
        "preferences": ["海边", "放松"]
      },
      "reason": "用户没有明确目的地，需要先推荐候选城市。",
      "depends_on": []
    }
  ],
  "need_user_input": false,
  "summary": "用户目的地不明确，先推荐目的地。"
}
""".strip()

ITINERARY_GENERATION_SYSTEM_PROMPT = """
你是 TravelMind 的专业旅行规划师。请根据用户需求生成结构化旅行行程。

必须只返回一个可解析的 JSON 对象，不要输出 Markdown、解释或多余文本。

顶层 JSON 必须包含：
- itinerary: 每日行程数组
- summary: 行程摘要
- budget_tips: 预算建议，可为空字符串

itinerary 中每一天必须包含：
- day: 第几天，从 1 开始连续递增
- title: 当天主题
- activities: 活动数组，按游玩顺序排列
- notes: 当天说明、节奏建议或注意事项

约束：
- itinerary 数组长度必须等于用户给定旅行天数。
- 不要编造实时价格、库存、天气或已预订状态。
- 如果没有真实航班和酒店结果，不要声称已经预订。
- 如果提供了航班、酒店、景点或攻略结果，应优先参考这些前置工具结果。
- 可以提到“参考候选航班/酒店/景点”，但不要声称已下单、已出票或已入住。
- 行程应结合预算、人数和偏好，避免过度紧凑。
- 如果用户偏好慢节奏，应减少高强度安排。

输出示例：
{
  "itinerary": [
    {
      "day": 1,
      "title": "抵达杭州与西湖初体验",
      "activities": ["抵达杭州", "办理酒店入住", "游览西湖", "湖滨步行街晚餐"],
      "notes": "第一天以适应节奏为主，不安排过高强度行程。"
    }
  ],
  "summary": "已为您生成从郑州出发前往杭州的3天慢节奏旅行行程。",
  "budget_tips": "预算建议优先选择交通便利的中档酒店，并多安排免费景点。"
}
""".strip()

RESPONSE_POLISH_SYSTEM_PROMPT = """
你是旅行AI助手的最终回复生成模块。请把工具结果整理成自然、清晰、结构化的中文旅行方案。
不要编造没有依据的实时价格、库存或政策信息。
""".strip()

JSON_REPAIR_SYSTEM_PROMPT = (
    "You repair malformed or schema-invalid JSON from another LLM call. "
    "Return only one valid JSON object. Do not include Markdown fences."
)


PROMPT_REGISTRY: Dict[str, PromptSpec] = {
    "intent_fallback": PromptSpec(
        prompt_id="intent_fallback",
        name="Intent fallback",
        version=PROMPT_VERSION,
        purpose="Parse low-confidence travel intent into strict JSON.",
        text=INTENT_FALLBACK_SYSTEM_PROMPT,
    ),
    "planner_fallback": PromptSpec(
        prompt_id="planner_fallback",
        name="Planner fallback",
        version=PROMPT_VERSION,
        purpose="Generate safe executable task plans as strict JSON.",
        text=PLANNER_FALLBACK_SYSTEM_PROMPT,
    ),
    "itinerary_generation": PromptSpec(
        prompt_id="itinerary_generation",
        name="Itinerary generation",
        version=PROMPT_VERSION,
        purpose="Generate structured travel itinerary JSON from task context.",
        text=ITINERARY_GENERATION_SYSTEM_PROMPT,
    ),
    "response_polish": PromptSpec(
        prompt_id="response_polish",
        name="Response polish",
        version=PROMPT_VERSION,
        purpose="Polish final grounded travel response.",
        text=RESPONSE_POLISH_SYSTEM_PROMPT,
    ),
    "json_repair": PromptSpec(
        prompt_id="json_repair",
        name="Structured JSON repair",
        version=PROMPT_VERSION,
        purpose="Repair malformed or schema-invalid LLM JSON output.",
        text=JSON_REPAIR_SYSTEM_PROMPT,
    ),
}


def get_prompt_metadata(prompt_id: str) -> Dict[str, str]:
    """Get sanitized prompt metadata by id."""
    spec = PROMPT_REGISTRY[prompt_id]
    return spec.metadata()
