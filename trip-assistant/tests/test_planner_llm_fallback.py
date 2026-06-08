"""
任务规划LLM fallback测试
"""
import pytest

from core.llm import LLMResponse
from core.planner import TaskPlanner


class FakeLLMClient:
    """可控LLM客户端"""

    def __init__(self, content: str = "", success: bool = True, available: bool = True):
        self.content = content
        self.success = success
        self.available = available
        self.calls = 0
        self.last_request = None

    async def chat(self, request):
        self.calls += 1
        self.last_request = request
        return LLMResponse(
            success=self.success,
            content=self.content,
            error=None if self.success else "mock planner failed",
            metadata={"mock": True},
        )


def _complex_intent():
    return {
        "intent": "travel_plan",
        "entities": {
            "origin": "郑州",
            "destination": "杭州",
            "departure_date": "2026-06-10",
            "return_date": None,
            "duration": 3,
            "budget": 3000,
            "travelers": 2,
            "preferences": ["慢节奏", "地铁附近", "美食"],
        },
        "missing_slots": [],
        "confidence": 0.8,
    }


def _complex_context():
    return {"query": "我想下个月从郑州去杭州放松三天，预算3000，不要太累，同时帮我兼顾景点、酒店和路线安排"}


@pytest.mark.asyncio
async def test_plan_async_can_force_template_with_mode_off():
    """Planner mode off forces template planning even for complex requests."""
    llm_client = FakeLLMClient(available=True)
    planner = TaskPlanner(llm_client=llm_client, llm_planner_mode="off")

    tasks = await planner.plan_async(_complex_intent(), _complex_context())

    assert len(tasks) == 6
    assert tasks[-1]["tool"] == "generate_itinerary"
    assert llm_client.calls == 0
    assert planner.last_plan_metadata["planner_mode"] == "template"
    assert planner.last_plan_metadata["llm_planner_enabled"] is False
    assert planner.last_plan_metadata["llm_planner_attempted"] is False
    assert planner.last_plan_metadata["skip_reason"] == "mode_off"


@pytest.mark.asyncio
async def test_plan_async_auto_routes_simple_request_to_template():
    """Simple high-confidence requests stay on the stable template planner."""
    llm_client = FakeLLMClient(available=True)
    planner = TaskPlanner(llm_client=llm_client)
    intent = {
        "intent": "travel_plan",
        "entities": {
            "origin": "郑州",
            "destination": "杭州",
            "duration": 3,
            "preferences": [],
        },
        "missing_slots": [],
        "confidence": 0.9,
    }

    tasks = await planner.plan_async(intent, {"query": "郑州去杭州三天"})

    assert len(tasks) == 6
    assert llm_client.calls == 0
    assert planner.last_plan_metadata["planner_mode_config"] == "auto"
    assert planner.last_plan_metadata["llm_planner_enabled"] is True
    assert planner.last_plan_metadata["llm_planner_attempted"] is False
    assert planner.last_plan_metadata["skip_reason"] == "not_complex_enough"


@pytest.mark.asyncio
async def test_plan_async_falls_back_when_llm_unavailable():
    """启用LLM但客户端不可用时回退模板规划"""
    llm_client = FakeLLMClient(available=False)
    planner = TaskPlanner(llm_client=llm_client, llm_planner_enabled=True)

    tasks = await planner.plan_async(_complex_intent(), _complex_context())

    assert len(tasks) == 6
    assert "search_flights" in [task.get("tool") for task in tasks]
    assert "search_trains" in [task.get("tool") for task in tasks]
    assert llm_client.calls == 0


@pytest.mark.asyncio
async def test_plan_async_uses_valid_llm_plan_for_complex_request():
    """复杂请求启用LLM时可以使用合法LLM规划结果"""
    llm_content = """
    {
      "intent": "travel_plan",
      "tasks": [
        {
          "task_id": "search_attractions_1",
          "task_type": "tool_call",
          "name": "先筛选轻松景点",
          "priority": 1,
          "tool": "search_attractions",
          "params": {"location": "杭州", "keywords": ["慢节奏", "美食"]},
          "reason": "用户强调不要太累，先筛选低强度景点。",
          "depends_on": []
        },
        {
          "task_id": "generate_itinerary_1",
          "task_type": "generate_itinerary",
          "name": "生成慢节奏行程",
          "priority": 2,
          "tool": "generate_itinerary",
          "params": {"origin": "郑州", "destination": "杭州", "duration": 3, "budget": 3000, "preferences": ["慢节奏"]},
          "reason": "综合用户偏好生成行程。",
          "depends_on": ["search_attractions_1"]
        }
      ],
      "need_user_input": false,
      "summary": "根据复杂偏好生成轻量化规划。"
    }
    """
    llm_client = FakeLLMClient(content=llm_content)
    planner = TaskPlanner(llm_client=llm_client, llm_planner_enabled=True)

    tasks = await planner.plan_async(_complex_intent(), _complex_context())

    assert len(tasks) == 2
    assert tasks[0]["name"] == "先筛选轻松景点"
    assert tasks[1]["tool"] == "generate_itinerary"
    assert llm_client.calls == 1
    assert llm_client.last_request.response_format == "json_object"
    assert planner.last_plan_metadata["planner_mode"] == "llm"
    assert planner.last_plan_metadata["planner_mode_config"] == "auto"
    assert planner.last_plan_metadata["llm_planner_route_decision"] == "attempt_llm"
    assert planner.last_plan_metadata["llm_planner_complexity_score"] >= 3
    assert "multiple_preferences" in planner.last_plan_metadata["llm_planner_complexity_signals"]
    assert planner.last_plan_metadata["llm_planner_attempted"] is True
    assert planner.last_plan_metadata["llm_planner_adopted"] is True
    assert planner.last_plan_metadata["llm_task_count"] == 2


@pytest.mark.asyncio
async def test_plan_async_supports_markdown_json_from_llm():
    """LLM返回Markdown JSON代码块时也可以解析"""
    llm_content = """```json
{
  "intent": "travel_plan",
  "tasks": [
    {
      "task_id": "recommend_destination_1",
      "task_type": "recommend_destination",
      "name": "推荐旅行目的地",
      "priority": 1,
      "tool": null,
      "params": {"budget": 3000, "preferences": ["海边"]},
      "reason": "用户目的地不明确。",
      "depends_on": []
    }
  ],
  "need_user_input": false,
  "summary": "先推荐目的地。"
}
```"""
    intent = _complex_intent()
    intent["entities"]["destination"] = None
    intent["missing_slots"] = ["destination"]
    llm_client = FakeLLMClient(content=llm_content)
    planner = TaskPlanner(llm_client=llm_client, llm_planner_enabled=True)

    tasks = await planner.plan_async(intent, {"query": "预算3000，想去海边放松三天，同时不要太累"})

    assert len(tasks) == 1
    assert tasks[0]["task_type"] == "recommend_destination"


@pytest.mark.asyncio
async def test_plan_async_falls_back_when_llm_returns_invalid_json():
    """LLM返回非法JSON时回退模板规划"""
    llm_client = FakeLLMClient(content="不是JSON")
    planner = TaskPlanner(llm_client=llm_client, llm_planner_enabled=True)

    tasks = await planner.plan_async(_complex_intent(), _complex_context())

    assert len(tasks) == 6
    assert tasks[-1]["tool"] == "generate_itinerary"
    assert llm_client.calls == 1


@pytest.mark.asyncio
async def test_plan_async_falls_back_when_llm_uses_unknown_tool():
    """LLM生成未知工具时回退模板规划"""
    llm_content = """
    {
      "intent": "travel_plan",
      "tasks": [
        {
          "task_id": "search_weather_1",
          "task_type": "tool_call",
          "name": "查询天气",
          "priority": 1,
          "tool": "search_weather",
          "params": {"location": "杭州"},
          "reason": "用户关心天气。",
          "depends_on": []
        }
      ],
      "need_user_input": false,
      "summary": "包含未知工具。"
    }
    """
    llm_client = FakeLLMClient(content=llm_content)
    planner = TaskPlanner(llm_client=llm_client, llm_planner_enabled=True)

    tasks = await planner.plan_async(_complex_intent(), _complex_context())

    assert len(tasks) == 6
    assert "search_weather" not in [task.get("tool") for task in tasks]
    assert llm_client.calls == 1
    assert planner.last_plan_metadata["planner_mode"] == "template"
    assert planner.last_plan_metadata["llm_planner_attempted"] is True
    assert planner.last_plan_metadata["llm_planner_adopted"] is False
    assert planner.last_plan_metadata["fallback_reason"] == "unsafe_plan"


@pytest.mark.asyncio
async def test_plan_async_falls_back_when_llm_schema_invalid():
    """LLM返回结构不符合TaskPlan时回退模板规划"""
    llm_client = FakeLLMClient(content='{"intent": "travel_plan", "tasks": "not-a-list"}')
    planner = TaskPlanner(llm_client=llm_client, llm_planner_enabled=True)

    tasks = await planner.plan_async(_complex_intent(), _complex_context())

    assert len(tasks) == 6
    assert tasks[0]["tool"] == "search_trains"
