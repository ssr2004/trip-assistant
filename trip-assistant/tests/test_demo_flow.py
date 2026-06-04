"""
演示链路测试
"""
import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.config import settings

from core.agent import TravelAgent


@pytest.mark.asyncio
async def test_demo_flow_plan_weather_route_dynamic_rag_with_trace():
    """完整演示链路：规划 -> 雨天调整 -> 路线优化 -> 景点追问"""
    agent = TravelAgent()
    session_id = "test-session-demo-flow-module37"

    plan = await agent.arun_with_artifacts("我要从郑州去杭州玩三天，预算3000，6月10日出发", session_id)
    assert_demo_contract(plan, expected_intent="travel_plan", required_artifacts=["itinerary", "attractions"])
    assert plan["artifacts"]["itinerary"]["destination"] == "杭州"
    assert len(plan["artifacts"]["itinerary"]["days"]) == 3
    assert_trace_has_tool(plan, "search_attractions")
    assert_trace_has_tool(plan, "generate_itinerary")

    weather_adjust = await agent.arun_with_artifacts("如果下雨怎么办？", session_id)
    assert_demo_contract(weather_adjust, expected_intent="itinerary_revision", required_artifacts=["itinerary", "weather_adjustment"])
    assert "已根据您的要求调整行程" in weather_adjust["response"]
    assert_trace_has_execution_mode(weather_adjust, "internal_revision")

    route = await agent.arun_with_artifacts("帮我按距离优化一下第二天行程", session_id)
    assert_demo_contract(route, expected_intent="itinerary_revision", required_artifacts=["itinerary", "route"])
    assert "已根据您的要求调整行程" in route["response"]
    assert route["artifacts"]["route"]["segments"]
    assert_trace_has_execution_mode(route, "internal_revision")

    dynamic_followup = await agent.arun_with_artifacts("西湖在哪里？", session_id)
    assert_demo_contract(dynamic_followup, expected_intent="dynamic_knowledge_query", required_artifacts=["attractions"])
    assert "根据刚才推荐的外部景点数据" in dynamic_followup["response"]
    assert dynamic_followup["artifacts"]["attractions"]["sources"]
    assert_trace_has_execution_mode(dynamic_followup, "dynamic_rag")


def test_demo_external_status_contract_supports_resume_script(monkeypatch):
    """演示控制台先展示外部API状态，且不泄露真实Key值。"""
    monkeypatch.setattr(settings, "AMAP_API_KEY", "demo-amap-key")
    monkeypatch.setattr(settings, "WEATHER_API_KEY", "")
    monkeypatch.setattr(settings, "EXTERNAL_API_MOCK_ENABLED", True)
    client = TestClient(app_main.app)

    response = client.get("/api/external/status")

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["real_api_count"] == 3
    assert data["summary"]["all_operational"] is True
    assert {service["capability"] for service in data["services"]} == {
        "poi_search",
        "route_distance",
        "weather_forecast",
    }
    assert "demo-amap-key" not in response.text


def assert_demo_contract(result, expected_intent, required_artifacts):
    """校验每个演示步骤都返回可展示结果和可观测Trace。"""
    assert result["response"]
    for artifact in required_artifacts:
        assert artifact in result["artifacts"]

    trace = result["execution_trace"]
    assert trace["summary"]["intent"] == expected_intent
    assert trace["summary"]["task_count"] >= 1
    assert trace["steps"][0]["stage"] == "intent"
    assert any(step["stage"] == "context" for step in trace["steps"])
    assert any(step["stage"] == "planning" for step in trace["steps"])
    assert "AMAP_API_KEY" not in str(trace)
    assert "demo-amap-key" not in str(trace)


def assert_trace_has_tool(result, tool_name):
    assert any(step.get("tool") == tool_name for step in result["execution_trace"]["steps"])


def assert_trace_has_execution_mode(result, execution_mode):
    assert any(step.get("execution_mode") == execution_mode for step in result["execution_trace"]["steps"])
