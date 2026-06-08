"""
旅行工具模块测试
"""
import pytest

from core.amap_client import AMapPOIClient
from core.cache import ExternalAPICache
from tools.registry import ToolRegistry
from tools.flights import FlightTool
from tools.hotels import HotelTool
from tools.trains import TrainTool
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


class FakeMCPClient:
    def __init__(self, result):
        self.result = result
        self.last_tool_name = None
        self.last_arguments = None
        self.calls = []

    async def call_tool(self, tool_name, arguments):
        self.last_tool_name = tool_name
        self.last_arguments = arguments
        self.calls.append((tool_name, arguments))
        return self.result


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
async def test_hotel_tool_prefers_amap_mcp_result_when_available():
    """酒店工具优先使用高德MCP真实搜索结果，且不补模拟库存字段"""
    mcp_client = FakeMCPClient({
        "success": True,
        "data": {
            "pois": [
                {
                    "id": "mcp-hotel-1",
                    "name": "杭州城中香格里拉",
                    "type": "住宿服务;宾馆酒店;五星级宾馆",
                    "address": "杭州市拱墅区长寿路6号",
                    "cityname": "杭州市",
                    "adname": "拱墅区",
                    "tel": "0571-87977999",
                    "biz_ext": {"rating": "4.7", "cost": "1280"},
                }
            ]
        },
        "error": None,
        "metadata": {"provider": "mcp_amap", "api_status": "success", "execution_mode": "real_mcp"},
    })
    tool = HotelTool(
        amap_client=amap_test_client({"杭州": []}),
        mcp_client=mcp_client,
    )

    result = await tool.execute(location="杭州", preferences=["西湖附近"])

    assert result["success"] is True
    assert result["metadata"]["source"] == "mcp_amap_hotel_search"
    assert result["metadata"]["provider"] == "mcp_amap"
    assert result["metadata"]["mock"] is False
    assert mcp_client.last_tool_name == "maps_text_search"
    assert mcp_client.last_arguments["city"] == "杭州"
    assert mcp_client.last_arguments["types"] == "100000"
    assert "酒店" in mcp_client.last_arguments["keywords"]
    hotel = result["data"]["hotels"][0]
    assert hotel["name"] == "杭州城中香格里拉"
    assert hotel["source"] == "mcp_amap"
    assert hotel["price"] == "1280"
    assert "price_per_night" not in hotel
    assert "room_type" not in hotel
    assert "availability" not in hotel


@pytest.mark.asyncio
async def test_hotel_tool_ranks_core_city_hotels_above_remote_counties():
    """酒店推荐应优先展示目的地核心城区和高相关酒店"""
    tool = HotelTool(amap_client=amap_test_client({
        "烟台": [
            {
                "id": "remote-hotel",
                "name": "莱州大成宾馆",
                "type": "住宿服务;宾馆酒店;宾馆",
                "address": "土山镇",
                "cityname": "烟台市",
                "adname": "莱州市",
                "biz_ext": {"rating": "5.0"},
            },
            {
                "id": "core-hotel",
                "name": "烟台中心假日酒店",
                "type": "住宿服务;宾馆酒店;四星级宾馆",
                "address": "烟台市芝罘区南大街",
                "cityname": "烟台市",
                "adname": "芝罘区",
                "tel": "0535-1234567",
                "biz_ext": {"rating": "4.7"},
            },
        ]
    }))

    result = await tool.execute(location="烟台")

    assert result["success"] is True
    hotels = result["data"]["hotels"]
    assert hotels[0]["name"] == "烟台中心假日酒店"
    assert hotels[0]["rank"] == 1
    assert "核心城区" in hotels[0]["selection_reason"]


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
async def test_train_tool_returns_real_mcp_ticket_result():
    """火车工具通过12306 MCP结果返回真实车次结构"""
    mcp_client = FakeMCPClient({
        "success": True,
        "data": {
            "trains": [
                {
                    "train_code": "G1964",
                    "from_station_name": "郑州东",
                    "to_station_name": "杭州东",
                    "start_time": "07:12",
                    "arrive_time": "12:35",
                    "lishi": "05:23",
                    "ze_num": "有",
                    "zy_num": "12",
                }
            ]
        },
        "error": None,
        "metadata": {"provider": "mcp_12306", "api_status": "success", "execution_mode": "real_mcp"},
    })
    tool = TrainTool(mcp_client=mcp_client)

    result = await tool.execute(origin="郑州", destination="杭州", date="2026-06-10")

    assert result["success"] is True
    assert result["metadata"]["source"] == "mcp_12306"
    assert result["metadata"]["mock"] is False
    assert result["metadata"]["real_inventory"] is True
    called_tools = [tool_name for tool_name, _ in mcp_client.calls]
    assert "search-stations" in called_tools
    assert "query-tickets" in called_tools
    assert "query-ticket-price" in called_tools
    query_ticket_args = next(arguments for tool_name, arguments in mcp_client.calls if tool_name == "query-tickets")
    assert query_ticket_args == {
        "from_station": "郑州",
        "to_station": "杭州",
        "train_date": "2026-06-10",
    }
    train = result["data"]["trains"][0]
    assert train["train_code"] == "G1964"
    assert train["from_station"] == "郑州东"
    assert train["to_station"] == "杭州东"
    assert train["seats"]["second_class"] == "有"


@pytest.mark.asyncio
async def test_train_tool_requires_date_and_does_not_mock():
    """没有日期时不调用真实12306，也不返回模拟车次"""
    tool = TrainTool(mcp_client=FakeMCPClient({"success": True, "data": {"trains": []}}))

    result = await tool.execute(origin="郑州", destination="杭州")

    assert result["success"] is False
    assert result["data"]["trains"] == []
    assert result["metadata"]["mock"] is False
    assert result["metadata"]["error_type"] == "missing_date"


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
async def test_itinerary_tool_uses_guide_planning_insights():
    """行程工具应消费攻略信号生成更具体的每日安排，而不是输出泛化模板。"""
    tool = ItineraryTool()

    result = await tool.execute(
        origin="太原",
        destination="郑州",
        duration=3,
        context={
            "guide": {
                "planning_insights": {
                    "highlights": ["河南博物院", "二七纪念塔", "少林寺"],
                    "route_hints": ["河南博物院和二七纪念塔适合安排在市区一天。"],
                    "food_hints": ["二七广场附近餐饮集中。"],
                    "caution_hints": ["热门景点建议提前预约门票。"],
                }
            }
        },
    )

    assert result["success"] is True
    itinerary = result["data"]["itinerary"]
    flattened = " ".join(" ".join(day["activities"]) + day["notes"] for day in itinerary)
    assert "河南博物院" in flattened
    assert "二七纪念塔" in flattened
    assert "少林寺" in flattened
    assert "热门景点建议提前预约门票" in flattened
    assert result["data"]["context_summary"]["has_guide"] is True


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
    assert "search_trains" in tools

    flight_result = await registry.execute("search_flights", {"origin": "郑州", "destination": "杭州"})
    train_result = await registry.execute("search_trains", {"origin": "郑州", "destination": "杭州", "date": "2026-06-10"})
    hotel_result = await registry.execute("search_hotels", {"location": "杭州"})
    attraction_result = await registry.execute("search_attractions", {"location": "杭州"})
    policy_result = await registry.execute("retrieve_policy", {"query": "机票能退吗"})
    guide_result = await registry.execute("retrieve_guide", {"query": "杭州三天旅行攻略", "destination": "杭州"})
    itinerary_result = await registry.execute("generate_itinerary", {"destination": "杭州", "duration": 3})
    weather_result = await registry.execute("get_weather_forecast", {"city": "杭州", "days": 3})

    assert flight_result["success"] is False
    assert flight_result["metadata"]["mock"] is False
    assert train_result["success"] is False
    assert train_result["metadata"]["mock"] is False
    assert train_result["data"]["trains"] == []
    assert hotel_result["success"] is False
    assert hotel_result["metadata"]["mock"] is False
    assert attraction_result["success"] is True
    assert policy_result["success"] is True
    assert guide_result["success"] is True
    assert itinerary_result["success"] is True
    assert weather_result["success"] is True
