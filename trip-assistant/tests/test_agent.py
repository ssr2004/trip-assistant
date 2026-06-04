"""
Agent测试
"""
import pytest
import asyncio
from langchain_core.messages import HumanMessage

from core.agent import TravelAgent


@pytest.fixture
def agent():
    """创建Agent实例"""
    return TravelAgent()


@pytest.mark.asyncio
async def test_travel_plan(agent):
    """测试旅行规划追问"""
    response = await agent.arun("我要从郑州去杭州", "test-session-1")
    assert response is not None
    assert "请问" in response
    assert "task_type" not in response


@pytest.mark.asyncio
async def test_complete_travel_plan_response(agent):
    """测试完整旅行规划回复"""
    response = await agent.arun("我要从郑州去杭州玩三天，预算3000，6月10日出发", "test-session-4")
    assert response is not None
    assert "已为您规划郑州到杭州的3天旅行方案" in response
    assert "航班推荐" in response
    assert "酒店推荐" in response
    assert "景点推荐" in response
    assert "每日行程" in response
    assert "{'" not in response


@pytest.mark.asyncio
async def test_agent_returns_itinerary_artifact(agent):
    """Agent可返回前端展示用的结构化行程数据"""
    result = await agent.arun_with_artifacts("我要从郑州去杭州玩三天，预算3000，6月10日出发", "test-session-artifact-plan")

    assert "已为您规划郑州到杭州的3天旅行方案" in result["response"]
    itinerary = result["artifacts"]["itinerary"]
    assert itinerary["destination"] == "杭州"
    assert len(itinerary["days"]) == 3
    assert itinerary["days"][0]["day"] == 1
    assert "budget_summary" in itinerary


@pytest.mark.asyncio
async def test_agent_returns_weather_artifact(agent):
    """Agent可返回前端展示用的结构化天气数据"""
    result = await agent.arun_with_artifacts("杭州明天天气怎么样？", "test-session-artifact-weather")

    weather = result["artifacts"]["weather"]
    assert weather["city"] == "杭州"
    assert len(weather["forecasts"]) == 3
    assert weather["travel_advice"]


@pytest.mark.asyncio
async def test_flight_search(agent):
    """测试航班搜索"""
    response = await agent.arun("搜索从北京到上海的航班", "test-session-2")
    assert response is not None
    assert "航班推荐" in response
    assert "东方航空" in response
    assert "None" not in response


@pytest.mark.asyncio
async def test_hotel_search(agent):
    """测试酒店搜索"""
    response = await agent.arun("搜索杭州的酒店", "test-session-3")
    assert response is not None
    assert "酒店推荐" in response
    assert "杭州西湖国宾馆" in response


@pytest.mark.asyncio
async def test_attraction_search_shows_external_rag_source(agent):
    """测试景点查询展示外部POI来源"""
    response = await agent.arun("杭州有什么好玩的", "test-session-attraction-source")
    assert response is not None
    assert "景点推荐" in response
    assert "西湖" in response
    assert "资料来源" in response
    assert "api/amap/attraction" in response


@pytest.mark.asyncio
async def test_agent_revises_itinerary_after_travel_plan(agent):
    """Agent支持基于上一轮完整行程调整景点到指定天数"""
    session_id = "test-session-itinerary-revision-move"

    await agent.arun("我要从郑州去杭州玩三天，预算3000，6月10日出发", session_id)
    response = await agent.arun("把西湖安排到第一天", session_id)

    assert "已根据您的要求调整行程" in response
    assert "已将西湖安排到第1天" in response
    assert "调整后的每日行程" in response
    assert "Day 1" in response
    assert "西湖" in response


@pytest.mark.asyncio
async def test_agent_replaces_attraction_after_travel_plan(agent):
    """Agent支持移除并替换历史行程中的景点"""
    session_id = "test-session-itinerary-revision-replace"

    await agent.arun("我要从郑州去杭州玩三天，预算3000，6月10日出发", session_id)
    response = await agent.arun("不要去宋城，换一个自然风光景点", session_id)

    assert "已根据您的要求调整行程" in response
    assert "已移除宋城" in response
    assert "调整后的每日行程" in response
    assert "Day" in response


@pytest.mark.asyncio
async def test_agent_optimizes_itinerary_route_order(agent):
    """Agent支持按距离优化某天景点顺序"""
    session_id = "test-session-itinerary-route-optimize"

    await agent.arun("我要从郑州去杭州玩三天，预算3000，6月10日出发", session_id)
    response = await agent.arun("帮我按距离优化一下第二天行程", session_id)

    assert "已根据您的要求调整行程" in response
    assert "已按距离优化第2天景点顺序" in response
    assert "路线优化摘要" in response
    assert "灵隐寺" in response
    assert "西溪国家湿地公园" in response
    assert "总距离" in response


@pytest.mark.asyncio
async def test_agent_queries_weather(agent):
    """Agent可以查询城市天气预报"""
    response = await agent.arun("杭州明天天气怎么样？", "test-session-weather-query")

    assert "杭州未来" in response
    assert "小雨" in response


@pytest.mark.asyncio
async def test_agent_adjusts_itinerary_for_rain(agent):
    """Agent可以根据雨天天气调整历史行程"""
    session_id = "test-session-weather-adjust"

    await agent.arun("我要从郑州去杭州玩三天，预算3000，6月10日出发", session_id)
    response = await agent.arun("如果下雨怎么办？", session_id)

    assert "已根据您的要求调整行程" in response
    assert "天气调整依据" in response


@pytest.mark.asyncio
async def test_agent_revise_itinerary_without_history(agent):
    """没有历史行程时行程调整给出合理提示"""
    response = await agent.arun("把西湖安排到第一天", "test-session-itinerary-revision-empty")

    assert "暂时还没有可调整的历史行程" in response


@pytest.mark.asyncio
async def test_itinerary_revision_isolated_by_session(agent):
    """不同会话的历史行程上下文互相隔离"""
    await agent.arun("我要从郑州去杭州玩三天，预算3000，6月10日出发", "test-session-with-itinerary")

    response = await agent.arun("把西湖安排到第一天", "test-session-without-itinerary")

    assert "暂时还没有可调整的历史行程" in response


@pytest.mark.asyncio
async def test_agent_answers_dynamic_rag_followup(agent):
    """Agent支持基于上一轮外部景点数据的追问"""
    session_id = "test-session-dynamic-followup"

    first_response = await agent.arun("杭州有什么好玩的", session_id)
    second_response = await agent.arun("西湖在哪里？", session_id)

    assert "西湖" in first_response
    assert "杭州市西湖区" in second_response
    assert "120.1551,30.2741" in second_response
    assert "资料来源" in second_response
    assert "api/amap/attraction" in second_response


@pytest.mark.asyncio
async def test_agent_lists_previous_dynamic_recommendations(agent):
    """Agent可以回答刚才推荐过哪些景点"""
    session_id = "test-session-dynamic-list"

    await agent.arun("杭州有什么好玩的", session_id)
    response = await agent.arun("刚才推荐的景点有哪些？", session_id)

    assert "根据刚才推荐的外部景点数据" in response
    assert "西湖" in response
    assert "灵隐寺" in response


@pytest.mark.asyncio
async def test_dynamic_rag_isolated_by_session(agent):
    """不同会话的动态RAG文档互相隔离"""
    await agent.arun("杭州有什么好玩的", "test-session-with-dynamic-rag")

    response = await agent.arun("西湖在哪里？", "test-session-without-dynamic-rag")

    assert "暂时没有在本轮对话的外部景点数据中找到相关信息" in response


@pytest.mark.asyncio
async def test_agent_collects_external_rag_documents(agent):
    """Agent执行景点工具后收集外部RAG动态文档"""
    result = await agent._execute_tasks({
        "messages": [HumanMessage(content="杭州有什么好玩的")],
        "tasks": [
            {
                "task_type": "tool_call",
                "tool": "search_attractions",
                "name": "搜索景点",
                "priority": 1,
                "params": {"location": "杭州"},
            }
        ],
    })

    dynamic_context = result["dynamic_rag_context"]
    assert dynamic_context["documents"]
    assert dynamic_context["document_count"] >= 1
    assert dynamic_context["sources"]
    assert dynamic_context["sources"][0]["source"].startswith("api/amap/attraction/")


@pytest.mark.asyncio
async def test_agent_propagates_tool_failure(agent):
    """Agent执行器可以识别工具内部失败状态"""
    result = await agent._execute_tasks({
        "tasks": [
            {
                "task_type": "tool_call",
                "tool": "search_flights",
                "name": "搜索航班",
                "priority": 1,
                "params": {"origin": "郑州"},
            }
        ]
    })

    task_result = result["task_results"][0]
    assert task_result["success"] is False
    assert "出发地和目的地" in task_result["error"]


if __name__ == "__main__":
    pytest.main([__file__])
