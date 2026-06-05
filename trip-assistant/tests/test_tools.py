"""
旅行工具模块测试
"""
import pytest

from core.amap_client import AMapPOIClient
from core.cache import ExternalAPICache
from tools.registry import ToolRegistry
from tools.flights import FlightTool
from tools.hotels import HotelTool
from tools.attractions import AttractionTool
from tools.policy import PolicyTool
from tools.guide import GuideTool
from tools.itinerary import ItineraryTool


class FakeAMapResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def fake_amap_request(pois_by_city):
    def request(method, url, params=None, **kwargs):
        city = (params or {}).get("city")
        return FakeAMapResponse({
            "status": "1",
            "info": "OK",
            "count": str(len(pois_by_city.get(city, []))),
            "pois": pois_by_city.get(city, []),
        })
    return request


def amap_test_client(pois_by_city):
    return AMapPOIClient(
        api_key="test-amap-key",
        request_func=fake_amap_request(pois_by_city),
        mock_enabled=False,
        cache=ExternalAPICache(enabled=False),
    )


@pytest.mark.asyncio
async def test_flight_tool_returns_standard_result():
    """航班工具返回真实机场POI衔接建议，不返回模拟航班库存"""
    tool = FlightTool(amap_client=amap_test_client({
        "郑州": [
            {
                "id": "zz-airport",
                "name": "郑州新郑国际机场",
                "type": "交通设施服务;机场相关;机场",
                "address": "郑州市新郑市迎宾大道",
                "cityname": "郑州市",
                "adname": "新郑市",
                "tel": "0371-96666",
            }
        ],
        "杭州": [
            {
                "id": "hz-airport",
                "name": "杭州萧山国际机场",
                "type": "交通设施服务;机场相关;机场",
                "address": "杭州市萧山区空港大道",
                "cityname": "杭州市",
                "adname": "萧山区",
            }
        ],
    }))

    result = await tool.execute(origin="郑州", destination="杭州", date="2026-06-10")

    assert result["success"] is True
    assert result["error"] is None
    assert result["metadata"]["tool"] == "search_flights"
    assert result["metadata"]["source"] == "amap_airport_poi"
    assert result["metadata"]["mock"] is False
    assert result["metadata"]["real_flight_inventory"] is False
    assert result["data"]["flights"] == []
    assert result["data"]["airport_guidance"]["airport_pairs"][0]["origin_airport"]["name"] == "郑州新郑国际机场"
    pair = result["data"]["airport_guidance"]["airport_pairs"][0]
    assert "flight_no" not in pair
    assert "price" not in pair
    assert "cabin_class" not in pair


@pytest.mark.asyncio
async def test_flight_tool_without_real_provider_does_not_mock():
    """没有真实机场POI数据源时不返回模拟航班推荐"""
    tool = FlightTool()

    result = await tool.execute(origin="郑州", destination="杭州", date="2026-06-10")

    assert result["success"] is False
    assert result["metadata"]["tool"] == "search_flights"
    assert result["metadata"]["mock"] is False
    assert result["data"]["flights"] == []
    assert result["data"]["airport_guidance"]["airport_pairs"] == []


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
    """酒店工具返回真实高德酒店POI，不返回模拟价格库存房型"""
    tool = HotelTool(amap_client=amap_test_client({
        "杭州": [
            {
                "id": "hotel-1",
                "name": "杭州西湖国宾馆",
                "type": "住宿服务;宾馆酒店;五星级宾馆",
                "address": "杭州市西湖区杨公堤18号",
                "cityname": "杭州市",
                "adname": "西湖区",
                "tel": "0571-87979889",
                "biz_ext": {"rating": "4.8"},
            }
        ]
    }))

    result = await tool.execute(location="杭州")

    assert result["success"] is True
    assert result["metadata"]["tool"] == "search_hotels"
    assert result["metadata"]["source"] == "amap_hotel_poi"
    assert result["metadata"]["mock"] is False
    assert result["metadata"]["real_inventory"] is False
    assert len(result["data"]["hotels"]) == 1
    hotel = result["data"]["hotels"][0]
    assert hotel["name"] == "杭州西湖国宾馆"
    assert hotel["source"] == "amap"
    assert "location" not in hotel
    assert "price_per_night" not in hotel
    assert "room_type" not in hotel
    assert "availability" not in hotel


@pytest.mark.asyncio
async def test_hotel_tool_without_real_provider_does_not_mock():
    """没有真实酒店POI数据源时不返回模拟酒店"""
    tool = HotelTool()

    result = await tool.execute(location="杭州")

    assert result["success"] is False
    assert result["metadata"]["tool"] == "search_hotels"
    assert result["metadata"]["mock"] is False
    assert result["data"]["hotels"] == []


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
    assert result["metadata"]["source"] == "amap_poi_mock"
    assert result["metadata"]["provider"] == "amap"
    assert result["metadata"]["mock"] is True
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
    assert result["data"]["context_summary"]["flight_count"] == 0


@pytest.mark.asyncio
async def test_itinerary_tool_applies_memory_profile_constraints():
    """行程工具会消费长期记忆画像并输出个性化摘要。"""
    tool = ItineraryTool()

    result = await tool.execute(
        destination="杭州",
        duration=2,
        preferences=["慢节奏", "少走路", "地铁附近"],
        memory_profile={
            "used_preferences": ["慢节奏", "少走路", "地铁附近", "不吃辣"],
            "budget_preference": "经济型",
            "excluded_preferences": ["不吃辣"],
            "conflicts": [],
        },
    )

    assert result["success"] is True
    data = result["data"]
    assert data["personalization_summary"]["memory_applied"] is True
    assert "少走路" in data["personalization_summary"]["used_preferences"]
    assert "控制步行距离" in data["itinerary"][0]["notes"]
    assert "不吃辣" in data["itinerary"][0]["notes"]


@pytest.mark.asyncio
async def test_registry_executes_new_tools():
    """工具注册表可以执行新增工具"""
    registry = ToolRegistry()
    tools = {tool["name"] for tool in registry.list_tools()}

    assert "retrieve_policy" in tools
    assert "retrieve_guide" in tools
    assert "generate_itinerary" in tools
    assert "optimize_route_order" in tools
    assert "get_weather_forecast" in tools

    flight_result = await registry.execute("search_flights", {"origin": "郑州", "destination": "杭州"})
    hotel_result = await registry.execute("search_hotels", {"location": "杭州"})
    attraction_result = await registry.execute("search_attractions", {"location": "杭州"})
    policy_result = await registry.execute("retrieve_policy", {"query": "机票能退吗"})
    guide_result = await registry.execute("retrieve_guide", {"query": "杭州三天旅行攻略", "destination": "杭州"})
    itinerary_result = await registry.execute("generate_itinerary", {"destination": "杭州", "duration": 3})
    weather_result = await registry.execute("get_weather_forecast", {"city": "杭州", "days": 3})

    assert flight_result["success"] is False
    assert flight_result["metadata"]["mock"] is False
    assert hotel_result["success"] is False
    assert hotel_result["metadata"]["mock"] is False
    assert attraction_result["success"] is True
    assert policy_result["success"] is True
    assert guide_result["success"] is True
    assert itinerary_result["success"] is True
    assert weather_result["success"] is True
