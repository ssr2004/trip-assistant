"""
高德路线距离客户端测试
"""
import pytest

from core.amap_route_client import AMapRouteClient


@pytest.mark.asyncio
async def test_amap_route_client_returns_mock_without_api_key():
    """无高德Key时返回路线距离mock"""
    client = AMapRouteClient(api_key="", mock_enabled=True)
    places = [
        {"name": "西湖", "location": "120.1551,30.2741"},
        {"name": "灵隐寺", "location": "120.1022,30.2400"},
    ]

    result = await client.distance_matrix(places, places, mode="walking")

    assert result["success"] is True
    assert result["metadata"]["provider"] == "amap"
    assert result["metadata"]["mock"] is True
    assert result["data"]["mode"] == "walking"
    assert result["data"]["results"]
    distances = [item for item in result["data"]["results"] if item["origin"] == "西湖" and item["destination"] == "灵隐寺"]
    assert distances[0]["distance"] == 6200
    assert distances[0]["duration"] > 0
