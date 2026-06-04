"""Tests for agent execution trace observability."""

import time

import pytest
from pydantic import ValidationError

from core.agent import TravelAgent
from core.trace import ExecutionTrace, normalize_execution_trace


def test_empty_execution_trace_has_stable_shape():
    assert normalize_execution_trace({}) == {"steps": [], "summary": {}}


def test_execution_trace_rejects_invalid_step_shape():
    with pytest.raises(ValidationError):
        ExecutionTrace.model_validate({"steps": [{"stage": "tool"}]})


@pytest.mark.asyncio
async def test_agent_returns_sanitized_execution_trace_for_tool_call():
    agent = TravelAgent()

    result = await agent.arun_with_artifacts("杭州明天天气怎么样？", "test-session-trace-weather")

    trace = result["execution_trace"]
    assert trace["summary"]["intent"] == "weather_query"
    assert trace["summary"]["intent_source"] == "rule"
    assert trace["summary"]["llm_mode"] in {"real_llm", "rule_fallback"}
    assert trace["summary"]["llm_model"]
    assert trace["summary"]["task_count"] >= 1
    assert any(step["stage"] == "intent" for step in trace["steps"])
    intent_step = next(step for step in trace["steps"] if step["stage"] == "intent")
    assert intent_step["execution_mode"] == "internal_rule"
    weather_step = next(step for step in trace["steps"] if step.get("tool") == "get_weather_forecast")
    assert isinstance(weather_step["duration_ms"], int)
    assert weather_step["duration_ms"] >= 0
    assert weather_step["execution_mode"] in {"real_api", "mock_fallback"}
    assert weather_step["result_summary"].startswith("forecasts=")
    assert trace["summary"]["total_duration_ms"] >= weather_step["duration_ms"]
    assert "AMAP_API_KEY" not in str(trace)
    assert "WEATHER_API_KEY" not in str(trace)


def test_execution_trace_includes_failure_error_type():
    agent = TravelAgent()
    task = {"task_type": "tool_call", "name": "Broken task"}
    task_result = agent._finalize_task_result(
        agent._build_error_result(task, "boom"),
        time.perf_counter(),
        error_type="missing_tool",
    )

    trace = agent._build_execution_trace({
        "intent": {"intent": "general_chat"},
        "tasks": [task],
        "task_results": [task_result],
        "rag_context": [],
    })

    failed_step = next(step for step in trace["steps"] if step["label"] == "Broken task")
    assert failed_step["status"] == "failed"
    assert failed_step["error_type"] == "missing_tool"
    assert failed_step["execution_mode"] == "tool"


def test_execution_trace_includes_intent_json_repair_metadata():
    agent = TravelAgent()

    trace = agent._build_execution_trace({
        "intent": {
            "intent": "travel_plan",
            "confidence": 0.8,
            "metadata": {
                "source": "llm",
                "json_repair_attempted": True,
                "json_repair_success": True,
            },
        },
        "tasks": [],
        "task_results": [],
        "rag_context": [],
    })

    intent_step = next(step for step in trace["steps"] if step["stage"] == "intent")
    assert intent_step["execution_mode"] == "llm"
    assert "json_repair=success" in intent_step["detail"]
    assert trace["summary"]["json_repair_attempted"] is True
    assert trace["summary"]["json_repair_success"] is True
