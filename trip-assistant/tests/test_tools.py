"""
旅行工具模块测试
"""
import pytest

from tools.registry import ToolRegistry
from tools.flights import FlightTool
from tools.hotels import HotelTool
from tools.attractions import AttractionTool
from tools.policy import PolicyTool
from tools.guide import GuideTool
from tools.itinerary import ItineraryTool


@pytest.mark.asyncio
async def test_flight_tool_returns_standard_result():
    """航班工具返回标准化结构"""
    tool = FlightTool()

    result = await tool.execute(origin="郑州", destination="杭州", date="2026-06-10")

    assert result["success"] is True
    assert result["error"] is None
    assert result["metadata"]["tool"] == "search_flights"
    assert result["metadata"]["source"] == "mock_flight_data"
    assert len(result["data"]["flights"]) == 3


@pytest.mark.asyncio
async def test_flight_tool_handles_missing_route():
    """航班工具缺少路线参数时返回失败结构"""
    tool = FlightTool()

    result = await tool.execute(origin="郑州")

    assert result["success"] is False
    assert result["data"]["flights"] == []
    assert "出发地和目的地" in result["error"]


@pytest.mark.asyncio
async def test_hotel_tool_returns_standard_result():
    """酒店工具返回标准化结构"""
    tool = HotelTool()

    result = await tool.execute(location="杭州")

    assert result["success"] is True
    assert result["metadata"]["tool"] == "search_hotels"
    assert result["metadata"]["source"] == "mock_hotel_data"
    assert len(result["data"]["hotels"]) == 3


@pytest.mark.asyncio
async def test_hotel_tool_handles_missing_location():
    """酒店工具缺少城市时返回失败结构"""
    tool = HotelTool()

    result = await tool.execute()

    assert result["success"] is False
    assert result["data"]["hotels"] == []
    assert "目的地" in result["error"]


@pytest.mark.asyncio
async def test_attraction_tool_returns_standard_result():
    """景点工具返回标准化结构"""
    tool = AttractionTool()

    result = await tool.execute(location="杭州")

    assert result["success"] is True
    assert result["metadata"]["tool"] == "search_attractions"
    assert result["metadata"]["source"] == "mock_attraction_data"
    assert len(result["data"]["attractions"]) == 4


@pytest.mark.asyncio
async def test_attraction_tool_handles_missing_location():
    """景点工具缺少城市时返回失败结构"""
    tool = AttractionTool()

    result = await tool.execute()

    assert result["success"] is False
    assert result["data"]["attractions"] == []
    assert "目的地" in result["error"]


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

    flight_result = await registry.execute("search_flights", {"origin": "郑州", "destination": "杭州"})
    hotel_result = await registry.execute("search_hotels", {"location": "杭州"})
    attraction_result = await registry.execute("search_attractions", {"location": "杭州"})
    policy_result = await registry.execute("retrieve_policy", {"query": "机票能退吗"})
    guide_result = await registry.execute("retrieve_guide", {"query": "杭州三天旅行攻略", "destination": "杭州"})
    itinerary_result = await registry.execute("generate_itinerary", {"destination": "杭州", "duration": 3})

    assert flight_result["success"] is True
    assert hotel_result["success"] is True
    assert attraction_result["success"] is True
    assert policy_result["success"] is True
    assert guide_result["success"] is True
    assert itinerary_result["success"] is True
