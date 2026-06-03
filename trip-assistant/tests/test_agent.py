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
    """测试旅行规划"""
    response = await agent.arun("我要从郑州去杭州", "test-session-1")
    assert response is not None
    assert len(response) > 0


@pytest.mark.asyncio
async def test_flight_search(agent):
    """测试航班搜索"""
    response = await agent.arun("搜索从北京到上海的航班", "test-session-2")
    assert response is not None


@pytest.mark.asyncio
async def test_hotel_search(agent):
    """测试酒店搜索"""
    response = await agent.arun("搜索杭州的酒店", "test-session-3")
    assert response is not None


if __name__ == "__main__":
    pytest.main([__file__])
