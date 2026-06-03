"""
外部API客户端基础设施测试
"""
import pytest

from core.external_api import ExternalAPIClient, ExternalAPIResponse, ExternalDataSource


class FakeHTTPResponse:
    """测试用HTTP响应"""

    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


def test_external_api_response_to_dict():
    """外部API响应模型可转换为标准字典"""
    response = ExternalAPIResponse(
        success=True,
        data={"items": []},
        metadata={"provider": "amap", "mock": True, "source": "external_api"},
    )

    result = response.to_dict()

    assert result["success"] is True
    assert result["data"] == {"items": []}
    assert result["error"] is None
    assert result["metadata"]["provider"] == "amap"


def test_external_data_source_model():
    """外部数据源描述模型可表达数据来源"""
    source = ExternalDataSource(
        provider="amap",
        source_type="poi",
        description="高德POI景点数据",
    )

    assert source.provider == "amap"
    assert source.source_type == "poi"
    assert source.requires_api_key is True


def test_client_available_depends_on_api_key():
    """客户端是否可用取决于API Key"""
    assert ExternalAPIClient(name="amap", api_key="").available() is False
    assert ExternalAPIClient(name="amap", api_key="test-key").available() is True


@pytest.mark.asyncio
async def test_request_json_returns_mock_when_key_missing_and_mock_enabled():
    """无Key且mock开启时返回mock成功响应"""
    client = ExternalAPIClient(name="amap", api_key="", mock_enabled=True)

    result = await client.request_json(
        method="GET",
        url="https://example.com/poi",
        mock_data={"pois": [{"name": "西湖"}]},
    )

    assert result["success"] is True
    assert result["data"]["pois"][0]["name"] == "西湖"
    assert result["error"] is None
    assert result["metadata"]["provider"] == "amap"
    assert result["metadata"]["mock"] is True
    assert result["metadata"]["source"] == "external_api"
    assert result["metadata"]["mock_reason"] == "api_key_missing"


@pytest.mark.asyncio
async def test_request_json_fails_when_key_missing_and_mock_disabled():
    """无Key且mock关闭时返回失败响应"""
    client = ExternalAPIClient(name="amap", api_key="", mock_enabled=False)

    result = await client.request_json(method="GET", url="https://example.com/poi")

    assert result["success"] is False
    assert "API Key" in result["error"]
    assert result["metadata"]["provider"] == "amap"
    assert result["metadata"]["mock"] is False


@pytest.mark.asyncio
async def test_request_json_uses_injected_request_function_when_key_exists():
    """有Key时可使用注入请求函数返回真实调用结构"""
    calls = []

    def fake_request(method, url, params=None, headers=None, json=None, timeout=None):
        calls.append({
            "method": method,
            "url": url,
            "params": params,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        })
        return FakeHTTPResponse({"status": "ok"})

    client = ExternalAPIClient(
        name="amap",
        api_key="test-key",
        timeout=3,
        retry_times=0,
        mock_enabled=False,
        request_func=fake_request,
    )

    result = await client.request_json(
        method="GET",
        url="https://example.com/poi",
        params={"city": "杭州"},
    )

    assert result["success"] is True
    assert result["data"] == {"status": "ok"}
    assert result["metadata"]["provider"] == "amap"
    assert result["metadata"]["mock"] is False
    assert result["metadata"]["attempt"] == 1
    assert calls[0]["timeout"] == 3
    assert calls[0]["params"] == {"city": "杭州"}
