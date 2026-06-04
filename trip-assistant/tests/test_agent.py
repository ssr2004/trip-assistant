"""
Agent测试
"""
import pytest
import asyncio
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
