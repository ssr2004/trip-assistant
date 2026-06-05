"""
高德路线距离客户端测试
"""
import pytest

from core.amap_route_client import AMapRouteClient
from core.cache import ExternalAPICache, MemoryCacheBackend


class FakeHTTPResponse:
    """测试用HTTP响应"""

    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


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


@pytest.mark.asyncio
async def test_amap_route_client_normalizes_real_distance_response_and_uses_cache():
    """有Key时标准化高德距离响应，第二次相同查询命中缓存"""
    calls = []

    def fake_request(method, url, params=None, headers=None, json=None, timeout=None):
        calls.append({"method": method, "url": url, "params": params})
        return FakeHTTPResponse({
            "status": "1",
            "info": "OK",
            "results": [
                {"origin_id": "1", "dest_id": "1", "distance": "0", "duration": "0"},
                {"origin_id": "1", "dest_id": "2", "distance": "6200", "duration": "1800"},
            ],
        })

    cache = ExternalAPICache(backend=MemoryCacheBackend(), enabled=True, ttl=3600)
    client = AMapRouteClient(
        api_key="test-key",
        request_func=fake_request,
        mock_enabled=False,
        cache=cache,
    )
    places = [
        {"id": "p1", "name": "西湖", "location": "120.1551,30.2741"},
        {"id": "p2", "name": "灵隐寺", "location": "120.1022,30.2400"},
    ]

    first_result = await client.distance_matrix(places, places, mode="driving")
    second_result = await client.distance_matrix(places, places, mode="driving")

    assert len(calls) == 1
    assert first_result["success"] is True
    assert first_result["metadata"]["mock"] is False
    assert first_result["metadata"]["cache_hit"] is False
    assert first_result["metadata"]["cache_write"] is True
    assert second_result["metadata"]["cache_hit"] is True
    segment = first_result["data"]["results"][1]
    assert segment["origin"] == "西湖"
    assert segment["destination"] == "灵隐寺"
    assert segment["distance"] == 6200
    assert segment["duration"] == 1800
