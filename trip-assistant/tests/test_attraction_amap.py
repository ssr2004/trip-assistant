"""
景点工具高德POI增强测试
"""
import pytest

from core.amap_client import AMapPOIClient
from tools.attractions import AttractionTool


class FakeHTTPResponse:
    """测试用HTTP响应"""

    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


@pytest.mark.asyncio
async def test_attraction_tool_uses_amap_mock_without_key():
    """景点工具无Key时使用高德POI mock fallback"""
    tool = AttractionTool(amap_client=AMapPOIClient(api_key="", mock_enabled=True))

    result = await tool.execute(location="杭州")

    assert result["success"] is True
    assert result["metadata"]["source"] == "amap_poi_mock"
    assert result["metadata"]["provider"] == "amap"
    assert result["metadata"]["mock"] is True
    assert len(result["data"]["attractions"]) == 4
    first_attraction = result["data"]["attractions"][0]
    assert first_attraction["name"] == "西湖"
    assert first_attraction["category"] == "风景名胜"
    assert first_attraction["ticket_price"] == "待定"
    assert first_attraction["source"] == "amap"
    assert "来自高德POI" in first_attraction["description"]


@pytest.mark.asyncio
async def test_attraction_tool_keeps_missing_location_failure():
    """缺少目的地时仍返回失败结构"""
    tool = AttractionTool()

    result = await tool.execute()

    assert result["success"] is False
    assert result["data"]["attractions"] == []
    assert "目的地" in result["error"]


@pytest.mark.asyncio
async def test_attraction_tool_normalizes_real_amap_response():
    """景点工具可以标准化真实高德POI响应结构"""
    def fake_request(method, url, params=None, headers=None, json=None, timeout=None):
        return FakeHTTPResponse({
            "status": "1",
            "pois": [
                {
                    "id": "real-poi-1",
                    "name": "测试景点",
                    "type": "风景名胜;旅游景点",
                    "address": "测试地址",
                    "location": "120.1,30.1",
                    "cityname": "杭州市",
                    "biz_ext": {"rating": "4.7"},
                }
            ],
        })

    tool = AttractionTool(
        amap_client=AMapPOIClient(api_key="test-key", request_func=fake_request, mock_enabled=False)
    )

    result = await tool.execute(location="杭州", keywords=["西湖"])

    assert result["success"] is True
    assert result["metadata"]["source"] == "amap_poi"
    assert result["metadata"]["provider"] == "amap"
    assert result["metadata"]["mock"] is False
    attraction = result["data"]["attractions"][0]
    assert attraction["name"] == "测试景点"
    assert attraction["category"] == "风景名胜"
    assert attraction["rating"] == 4.7
    assert attraction["address"] == "测试地址"
    assert attraction["location"] == "120.1,30.1"


@pytest.mark.asyncio
async def test_attraction_tool_falls_back_to_local_mock_when_amap_returns_empty():
    """高德返回空POI时降级为本地mock景点"""
    def fake_request(method, url, params=None, headers=None, json=None, timeout=None):
        return FakeHTTPResponse({"status": "1", "pois": []})

    tool = AttractionTool(
        amap_client=AMapPOIClient(api_key="test-key", request_func=fake_request, mock_enabled=False)
    )

    result = await tool.execute(location="杭州")

    assert result["success"] is True
    assert result["metadata"]["source"] == "mock_attraction_data"
    assert result["metadata"]["provider"] == "local"
    assert result["metadata"]["mock"] is True
    assert result["metadata"]["fallback_reason"] == "amap_poi_empty"
    assert result["data"]["attractions"][0]["name"] == "西湖"
