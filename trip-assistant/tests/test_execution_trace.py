"""Tests for agent execution trace observability."""

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
    assert trace["summary"]["task_count"] >= 1
    assert any(step["stage"] == "intent" for step in trace["steps"])
    assert any(step.get("tool") == "get_weather_forecast" for step in trace["steps"])
    assert "AMAP_API_KEY" not in str(trace)
    assert "WEATHER_API_KEY" not in str(trace)
