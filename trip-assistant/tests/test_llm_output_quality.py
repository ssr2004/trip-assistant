"""LLM输出质量门禁测试。"""

import pytest

from core.intent import IntentParser
from core.llm import LLMResponse
from core.llm.quality import audit_itinerary_quality, audit_task_plan_quality
from core.planner import TaskPlanner
from models.itinerary import LLMItineraryPlan
from models.task import TaskPlan
from tools.itinerary import ItineraryTool


class FakeLLMClient:
    """可控LLM客户端。"""

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
            error=None if self.success else "mock failed",
            metadata={
                "model": "fake-model",
                "provider": "fake",
                "duration_ms": 1,
                **request.metadata,
            },
        )


def _complex_intent():
    return {
        "intent": "travel_plan",
        "entities": {
            "origin": "郑州",
            "destination": "杭州",
            "departure_date": "2026-06-10",
            "duration": 3,
            "budget": 3000,
            "travelers": 2,
            "preferences": ["慢节奏", "地铁附近", "美食"],
        },
        "missing_slots": [],
        "confidence": 0.8,
    }


def test_task_plan_quality_rejects_forbidden_operations():
    """任务计划质量门禁拒绝支付/下单类敏感操作描述。"""
    plan = TaskPlan.model_validate({
        "intent": "travel_plan",
        "tasks": [
            {
                "task_id": "generate_itinerary_1",
                "task_type": "generate_itinerary",
                "name": "生成行程",
                "priority": 1,
                "tool": "generate_itinerary",
                "params": {"destination": "杭州"},
                "reason": "顺便完成支付并下单。",
                "depends_on": [],
            }
        ],
        "need_user_input": False,
        "summary": "错误计划",
    })

    result = audit_task_plan_quality(plan)

    assert result.passed is False
    assert any(issue.startswith("forbidden_operation_claim") for issue in result.issues)


def test_itinerary_quality_rejects_realtime_booking_claims():
    """行程质量门禁拒绝已预订/已出票类无依据实时状态。"""
    plan = LLMItineraryPlan.model_validate({
        "itinerary": [
            {
                "day": 1,
                "title": "抵达杭州",
                "activities": ["入住已预订酒店"],
                "notes": "已出票，直接出行。",
            }
        ],
        "summary": "杭州1天行程",
        "budget_tips": "无",
    })

    result = audit_itinerary_quality(plan, expected_duration=1)

    assert result.passed is False
    assert "forbidden_realtime_claim:1" in result.issues


@pytest.mark.asyncio
async def test_intent_llm_quality_failure_falls_back_to_rule():
    """LLM意图通过schema但质量门禁失败时回退规则结果。"""
    llm_content = """
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
        "preferences": ["海边"]
      },
      "confidence": 0.8,
      "missing_slots": ["origin", "departure_date"],
      "followup_question": "请问您准备从哪里出发？"
    }
    """
    llm_client = FakeLLMClient(content=llm_content)
    parser = IntentParser(llm_client=llm_client)

    result = await parser.parse_async("我想找个海边城市放松三天，预算3000")

    assert result["metadata"]["source"] == "rule_fallback"
    assert result["metadata"]["llm_error_type"] == "quality_gate_failed"
    assert result["metadata"]["llm_output_quality_passed"] is False
    assert "travel_plan_destination_missing_not_declared" in result["metadata"]["llm_output_quality_issues"]


@pytest.mark.asyncio
async def test_planner_llm_quality_failure_falls_back_to_template():
    """LLM planner输出安全但质量门禁失败时回退模板规划。"""
    llm_content = """
    {
      "intent": "travel_plan",
      "tasks": [
        {
          "task_id": "generate_itinerary_1",
          "task_type": "generate_itinerary",
          "name": "生成并下单",
          "priority": 1,
          "tool": "generate_itinerary",
          "params": {"destination": "杭州", "preferences": ["慢节奏"]},
          "reason": "生成行程并完成支付。",
          "depends_on": []
        }
      ],
      "need_user_input": false,
      "summary": "包含敏感操作描述。"
    }
    """
    planner = TaskPlanner(llm_client=FakeLLMClient(content=llm_content), llm_planner_enabled=True)

    tasks = await planner.plan_async(_complex_intent(), {"query": "复杂杭州三天规划，同时兼顾预算、路线、酒店和美食"})

    assert len(tasks) == 6
    assert planner.last_plan_metadata["planner_mode"] == "template"
    assert planner.last_plan_metadata["fallback_reason"] == "quality_gate_failed"
    assert planner.last_plan_metadata["llm_output_quality_passed"] is False


@pytest.mark.asyncio
async def test_itinerary_llm_quality_failure_falls_back_to_template():
    """LLM行程输出包含无依据实时状态时回退模板。"""
    llm_content = """
    {
      "itinerary": [
        {
          "day": 1,
          "title": "抵达杭州",
          "activities": ["入住已预订酒店", "西湖散步"],
          "notes": "已出票，直接出发。"
        }
      ],
      "summary": "杭州1天行程",
      "budget_tips": "无"
    }
    """
    tool = ItineraryTool(llm_client=FakeLLMClient(content=llm_content), llm_enabled=True)

    result = await tool.execute(destination="杭州", duration=1)

    assert result["data"]["generation_mode"] == "template"
    assert result["metadata"]["fallback_reason"] == "quality_gate_failed"
    assert result["metadata"]["llm_output_quality_passed"] is False
    assert "forbidden_realtime_claim:1" in result["metadata"]["llm_output_quality_issues"]
