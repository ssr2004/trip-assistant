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
                "llm_call_count": 2,
                "llm_success_count": 2,
                "llm_failure_count": 0,
                "llm_duration_ms": 180,
                "llm_prompt_tokens": 40,
                "llm_completion_tokens": 20,
                "llm_total_tokens": 60,
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
    assert trace["summary"]["llm_call_count"] == 2
    assert trace["summary"]["llm_repair_count"] == 1
    assert trace["summary"]["llm_duration_ms"] == 180
    assert trace["summary"]["llm_total_tokens"] == 60
    assert trace["summary"]["llm_token_usage_available"] is True
    assert trace["summary"]["llm_cost_basis"] == "provider_token_usage"


def test_execution_trace_aggregates_runtime_tool_modes():
    agent = TravelAgent()
    real_api_result = {
        "success": True,
        "task": {"task_type": "tool_call", "tool": "search_attractions", "name": "Attractions"},
        "result": {"metadata": {"source": "external_api"}, "data": {"attractions": [1, 2]}},
        "meta": {
            "duration_ms": 30,
            "execution_mode": "real_api",
            "result_summary": "attractions=2",
        },
    }
    mock_result = {
        "success": True,
        "task": {"task_type": "tool_call", "tool": "search_hotels", "name": "Hotels"},
        "result": {"metadata": {"mock": True}, "data": {"hotels": [1]}},
        "meta": {
            "duration_ms": 20,
            "execution_mode": "mock_fallback",
            "result_summary": "hotels=1",
        },
    }
    template_result = {
        "success": True,
        "task": {"task_type": "generate_itinerary", "tool": "generate_itinerary", "name": "Itinerary"},
        "result": {"metadata": {"source": "template_itinerary_generator"}, "data": {"itinerary": [1]}},
        "meta": {
            "duration_ms": 10,
            "execution_mode": "template",
            "result_summary": "itinerary_days=1",
        },
    }

    trace = agent._build_execution_trace({
        "intent": {"intent": "travel_plan", "metadata": {"source": "rule"}},
        "tasks": [real_api_result["task"], mock_result["task"], template_result["task"]],
        "task_results": [real_api_result, mock_result, template_result],
        "rag_context": [],
    })

    assert trace["summary"]["tool_total_duration_ms"] == 60
    assert trace["summary"]["real_api_count"] == 1
    assert trace["summary"]["mock_fallback_count"] == 1
    assert trace["summary"]["template_task_count"] == 1


def test_execution_trace_includes_dependency_resolution_metadata():
    agent = TravelAgent()
    task_result = {
        "success": True,
        "task": {"task_type": "generate_itinerary", "tool": "generate_itinerary", "name": "Itinerary"},
        "result": {"metadata": {"source": "template_itinerary_generator"}, "data": {"itinerary": [1]}},
        "meta": {
            "duration_ms": 10,
            "execution_mode": "template",
            "result_summary": "itinerary_days=1",
            "dependency_ids": ["search_attractions_1", "get_weather_forecast_1", "missing_task"],
            "resolved_dependencies": ["search_attractions_1", "get_weather_forecast_1"],
            "missing_dependencies": ["missing_task"],
            "failed_dependencies": [],
            "dependency_context_keys": ["attractions", "weather", "errors"],
            "dependency_error_count": 1,
        },
    }

    trace = agent._build_execution_trace({
        "intent": {"intent": "travel_plan", "metadata": {"source": "rule"}},
        "tasks": [task_result["task"]],
        "task_results": [task_result],
        "rag_context": [],
    })

    itinerary_step = next(step for step in trace["steps"] if step.get("tool") == "generate_itinerary")
    assert itinerary_step["dependency_ids"] == ["search_attractions_1", "get_weather_forecast_1", "missing_task"]
    assert itinerary_step["resolved_dependencies"] == ["search_attractions_1", "get_weather_forecast_1"]
    assert itinerary_step["missing_dependencies"] == ["missing_task"]
    assert itinerary_step["failed_dependencies"] == []
    assert itinerary_step["dependency_context_keys"] == ["attractions", "weather", "errors"]
    assert itinerary_step["dependency_error_count"] == 1


def test_execution_trace_includes_llm_planner_metadata():
    agent = TravelAgent()

    trace = agent._build_execution_trace({
        "intent": {"intent": "travel_plan", "metadata": {"source": "rule"}},
        "tasks": [{"task_type": "tool_call", "tool": "search_attractions"}],
        "task_results": [],
        "rag_context": [],
        "planner_metadata": {
            "planner_mode": "llm",
            "planner_mode_config": "auto",
            "llm_planner_auto_route": True,
            "llm_planner_complexity_score": 6,
            "llm_planner_complexity_signals": ["long_query", "multiple_preferences"],
            "llm_planner_route_decision": "attempt_llm",
            "llm_planner_enabled": True,
            "llm_planner_available": True,
            "llm_planner_attempted": True,
            "llm_planner_adopted": True,
            "llm_planner_duration_ms": 123,
            "llm_planner_total_tokens": 456,
        },
    })

    planning_step = next(step for step in trace["steps"] if step["stage"] == "planning")
    assert planning_step["execution_mode"] == "llm"
    assert "planner=llm" in planning_step["detail"]
    assert "route=attempt_llm" in planning_step["detail"]
    assert trace["summary"]["planner_mode"] == "llm"
    assert trace["summary"]["planner_mode_config"] == "auto"
    assert trace["summary"]["llm_planner_auto_route"] is True
    assert trace["summary"]["llm_planner_complexity_score"] == 6
    assert trace["summary"]["llm_planner_complexity_signals"] == ["long_query", "multiple_preferences"]
    assert trace["summary"]["llm_planner_route_decision"] == "attempt_llm"
    assert trace["summary"]["llm_planner_enabled"] is True
    assert trace["summary"]["llm_planner_attempted"] is True
    assert trace["summary"]["llm_planner_adopted"] is True
    assert trace["summary"]["llm_planner_duration_ms"] == 123
    assert trace["summary"]["llm_planner_total_tokens"] == 456
