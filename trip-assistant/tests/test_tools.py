"""
旅行工具模块测试
"""
import pytest

from tools.registry import ToolRegistry
from tools.policy import PolicyTool
from tools.guide import GuideTool
from tools.itinerary import ItineraryTool


@pytest.mark.asyncio
async def test_policy_tool_retrieves_policy_document():
    """政策工具可以检索政策文档"""
    tool = PolicyTool()

    result = await tool.execute(query="机票能退吗")

    assert result["success"] is True
    assert result["data"]["query"] == "机票能退吗"
    assert "answer" in result["data"]
    assert result["metadata"]["tool"] == "retrieve_policy"


@pytest.mark.asyncio
async def test_guide_tool_retrieves_hangzhou_guide():
    """攻略工具可以检索杭州攻略"""
    tool = GuideTool()

    result = await tool.execute(query="杭州三天旅行攻略", destination="杭州")

    assert result["success"] is True
    assert result["data"]["destination"] == "杭州"
    assert "杭州" in result["data"]["answer"]
    assert result["metadata"]["tool"] == "retrieve_guide"


@pytest.mark.asyncio
async def test_itinerary_tool_generates_itinerary():
    """行程工具可以生成基础行程"""
    tool = ItineraryTool()

    result = await tool.execute(
        origin="郑州",
        destination="杭州",
        duration=3,
        budget=3000,
        travelers=2,
        preferences=["慢节奏", "自然风光"],
    )

    assert result["success"] is True
    assert result["data"]["origin"] == "郑州"
    assert result["data"]["destination"] == "杭州"
    assert len(result["data"]["itinerary"]) == 3
    assert result["data"]["budget_summary"]["input_budget"] == 3000


@pytest.mark.asyncio
async def test_registry_executes_new_tools():
    """工具注册表可以执行新增工具"""
    registry = ToolRegistry()
    tools = {tool["name"] for tool in registry.list_tools()}

    assert "retrieve_policy" in tools
    assert "retrieve_guide" in tools
    assert "generate_itinerary" in tools

    policy_result = await registry.execute("retrieve_policy", {"query": "机票能退吗"})
    guide_result = await registry.execute("retrieve_guide", {"query": "杭州三天旅行攻略", "destination": "杭州"})
    itinerary_result = await registry.execute("generate_itinerary", {"destination": "杭州", "duration": 3})

    assert policy_result["success"] is True
    assert guide_result["success"] is True
    assert itinerary_result["success"] is True
