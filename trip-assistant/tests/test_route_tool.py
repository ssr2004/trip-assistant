"""
路线距离工具测试
"""
import pytest

from core.amap_route_client import AMapRouteClient
from tools.routes import RouteTool


@pytest.mark.asyncio
async def test_route_tool_optimizes_place_order_with_mock_distance():
    """路线工具可以按mock距离优化景点顺序"""
    tool = RouteTool(route_client=AMapRouteClient(api_key="", mock_enabled=True))
    places = [
        {"name": "西湖", "location": "120.1551,30.2741"},
        {"name": "宋城", "location": "120.0962,30.1607"},
        {"name": "灵隐寺", "location": "120.1022,30.2400"},
        {"name": "西溪国家湿地公园", "location": "120.0648,30.2745"},
    ]

    result = await tool.execute(places=places, start_place="西湖")

    assert result["success"] is True
    assert result["metadata"]["source"] == "amap_route_mock"
    assert result["metadata"]["provider"] == "amap"
    ordered_names = [place["name"] for place in result["data"]["ordered_places"]]
    assert ordered_names == ["西湖", "灵隐寺", "西溪国家湿地公园", "宋城"]
    assert result["data"]["segments"][0]["from"] == "西湖"
    assert result["data"]["segments"][0]["to"] == "灵隐寺"
    assert result["data"]["total_distance"] > 0
    assert result["data"]["total_duration"] > 0


@pytest.mark.asyncio
async def test_route_tool_requires_at_least_two_places():
    """路线工具至少需要两个景点"""
    tool = RouteTool(route_client=AMapRouteClient(api_key="", mock_enabled=True))

    result = await tool.execute(places=[{"name": "西湖"}])

    assert result["success"] is False
    assert "至少需要两个景点" in result["error"]
