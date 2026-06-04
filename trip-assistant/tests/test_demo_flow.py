"""
演示链路测试
"""
import pytest

from core.agent import TravelAgent


@pytest.mark.asyncio
async def test_demo_flow_plan_weather_adjust_and_route_optimize():
    """完整演示链路：规划 -> 雨天调整 -> 路线优化"""
    agent = TravelAgent()
    session_id = "test-session-demo-flow"

    plan = await agent.arun_with_artifacts("我要从郑州去杭州玩三天，预算3000，6月10日出发", session_id)
    assert plan["artifacts"]["itinerary"]["destination"] == "杭州"
    assert len(plan["artifacts"]["itinerary"]["days"]) == 3
    assert "attractions" in plan["artifacts"]

    weather_adjust = await agent.arun_with_artifacts("如果下雨怎么办？", session_id)
    assert "已根据您的要求调整行程" in weather_adjust["response"]
    assert "itinerary" in weather_adjust["artifacts"]
    assert "weather_adjustment" in weather_adjust["artifacts"]

    route = await agent.arun_with_artifacts("帮我按距离优化一下第二天行程", session_id)
    assert "已根据您的要求调整行程" in route["response"]
    assert "itinerary" in route["artifacts"]
    assert "route" in route["artifacts"]
    assert route["artifacts"]["route"]["segments"]
