"""
高德POI客户端测试
"""
import pytest

from core.amap_client import AMapPOIClient


class FakeHTTPResponse:
    """测试用HTTP响应"""

    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


@pytest.mark.asyncio
async def test_amap_poi_client_returns_mock_without_api_key():
    """无高德Key时返回mock POI"""
    client = AMapPOIClient(api_key="", mock_enabled=True)

    result = await client.search_pois(city="杭州", keywords="景点")

    assert result["success"] is True
    assert result["metadata"]["provider"] == "amap"
    assert result["metadata"]["mock"] is True
    assert result["metadata"]["source"] == "external_api"
    assert result["data"]["pois"][0]["name"] == "西湖"
    assert len(result["data"]["pois"]) == 4


@pytest.mark.asyncio
async def test_amap_poi_client_returns_city_specific_mock():
    """不同城市返回对应mock POI"""
    client = AMapPOIClient(api_key="", mock_enabled=True)

    result = await client.search_pois(city="成都", keywords="景点")

    names = [poi["name"] for poi in result["data"]["pois"]]
    assert "成都大熊猫繁育研究基地" in names
    assert "宽窄巷子" in names


@pytest.mark.asyncio
async def test_amap_poi_client_uses_injected_request_function_with_key():
    """有Key时可以通过注入请求函数模拟真实高德返回"""
    calls = []

    def fake_request(method, url, params=None, headers=None, json=None, timeout=None):
        calls.append({"method": method, "url": url, "params": params, "timeout": timeout})
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

    client = AMapPOIClient(api_key="test-key", request_func=fake_request, mock_enabled=False)

    result = await client.search_pois(city="杭州", keywords="西湖")

    assert result["success"] is True
    assert result["metadata"]["provider"] == "amap"
    assert result["metadata"]["mock"] is False
    assert result["data"]["pois"][0]["name"] == "测试景点"
    assert calls[0]["params"]["key"] == "test-key"
    assert calls[0]["params"]["city"] == "杭州"
    assert calls[0]["params"]["types"] == AMapPOIClient.SCENIC_TYPES
