"""
外部API缓存测试
"""
import pytest

from core.amap_client import AMapPOIClient
from core.cache import CacheBackend, ExternalAPICache, MemoryCacheBackend


class FakeHTTPResponse:
    """测试用HTTP响应"""

    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class FailingRedisBackend(CacheBackend):
    """模拟不可用的Redis后端"""

    name = "redis"

    async def get(self, key: str):
        raise RuntimeError("redis unavailable")

    async def set(self, key: str, value, ttl: int) -> None:
        raise RuntimeError("redis unavailable")


@pytest.mark.asyncio
async def test_external_api_cache_builds_stable_key_without_secret():
    """缓存Key稳定且不包含API Key等敏感参数"""
    cache = ExternalAPICache(backend=MemoryCacheBackend(), enabled=True, ttl=3600)

    key1 = cache.build_key("amap", "poi", {"city": "杭州", "keywords": "景点", "key": "secret-1"})
    key2 = cache.build_key("amap", "poi", {"keywords": "景点", "city": "杭州", "key": "secret-2"})

    assert key1 == key2
    assert key1.startswith("travelmind:external_api:amap:poi:")
    assert "secret" not in key1
    assert "杭州" not in key1


@pytest.mark.asyncio
async def test_memory_cache_backend_reads_written_value():
    """内存缓存可以写入并读取外部API响应"""
    cache = ExternalAPICache(backend=MemoryCacheBackend(), enabled=True, ttl=3600)

    await cache.set("amap", "poi", {"city": "杭州"}, {"success": True, "data": {"pois": []}})
    value, metadata = await cache.get("amap", "poi", {"city": "杭州"})

    assert value["success"] is True
    assert value["data"]["pois"] == []
    assert metadata["cache_hit"] is True
    assert metadata["cache_backend"] == "memory"


@pytest.mark.asyncio
async def test_external_api_cache_falls_back_when_redis_unavailable():
    """Redis不可用时降级为内存缓存"""
    cache = ExternalAPICache(
        backend=FailingRedisBackend(),
        fallback_backend=MemoryCacheBackend(),
        enabled=True,
        ttl=3600,
    )

    first_value, first_metadata = await cache.get("amap", "poi", {"city": "杭州"})
    assert first_value is None
    assert first_metadata["cache_hit"] is False
    assert first_metadata["cache_backend"] == "memory_fallback"
    assert "redis unavailable" in first_metadata["cache_error"]

    set_metadata = await cache.set(
        "amap",
        "poi",
        {"city": "杭州"},
        {"success": True, "data": {"pois": [{"name": "西湖"}]}},
    )
    assert set_metadata["cache_backend"] == "memory_fallback"
    assert set_metadata["cache_write"] is True

    second_value, second_metadata = await cache.get("amap", "poi", {"city": "杭州"})
    assert second_value["data"]["pois"][0]["name"] == "西湖"
    assert second_metadata["cache_hit"] is True
    assert second_metadata["cache_backend"] == "memory_fallback"


@pytest.mark.asyncio
async def test_amap_poi_client_uses_external_api_cache():
    """高德POI客户端第二次相同查询命中缓存，不重复请求外部API"""
    calls = []

    def fake_request(method, url, params=None, headers=None, json=None, timeout=None):
        calls.append({"method": method, "url": url, "params": params})
        return FakeHTTPResponse({
            "status": "1",
            "pois": [
                {
                    "id": "real-poi-1",
                    "name": "测试景点",
                    "type": "风景名胜",
                    "address": "测试地址",
                    "location": "120.1,30.1",
                    "cityname": "杭州市",
                }
            ],
        })

    cache = ExternalAPICache(backend=MemoryCacheBackend(), enabled=True, ttl=3600)
    client = AMapPOIClient(
        api_key="test-key",
        request_func=fake_request,
        mock_enabled=False,
        cache=cache,
    )

    first_result = await client.search_pois(city="杭州", keywords="西湖")
    second_result = await client.search_pois(city="杭州", keywords="西湖")

    assert len(calls) == 1
    assert first_result["metadata"]["cache_hit"] is False
    assert first_result["metadata"]["cache_backend"] == "memory"
    assert first_result["metadata"]["cache_write"] is True
    assert second_result["metadata"]["cache_hit"] is True
    assert second_result["metadata"]["cache_backend"] == "memory"
    assert second_result["data"]["pois"][0]["name"] == "测试景点"
