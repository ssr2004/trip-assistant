"""
天气客户端测试
"""
import pytest

from core.weather_client import WeatherClient


class FakeHTTPResponse:
    """测试用HTTP响应"""

    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


@pytest.mark.asyncio
async def test_weather_client_returns_mock_without_api_key():
    """无天气Key时返回mock天气预报"""
    client = WeatherClient(api_key="", mock_enabled=True)

    result = await client.forecast(city="杭州", days=3)

    assert result["success"] is True
    assert result["metadata"]["provider"] == "weather"
    assert result["metadata"]["mock"] is True
    assert result["data"]["city"] == "杭州"
    assert len(result["data"]["forecasts"]) == 3
    assert result["data"]["forecasts"][0]["weather"] == "小雨"
    assert result["data"]["forecasts"][0]["suitable_for_outdoor"] is False


@pytest.mark.asyncio
async def test_weather_client_uses_amap_forecast_api_with_key():
    """有Key时调用高德天气预报API并标准化返回"""
    calls = []

    def fake_request(method, url, params=None, headers=None, json=None, timeout=None):
        calls.append({"method": method, "url": url, "params": params, "timeout": timeout})
        return FakeHTTPResponse({
            "status": "1",
            "info": "OK",
            "forecasts": [
                {
                    "city": "杭州市",
                    "adcode": "330100",
                    "province": "浙江",
                    "reporttime": "2026-06-04 11:00:00",
                    "casts": [
                        {
                            "date": "2026-06-10",
                            "week": "3",
                            "dayweather": "小雨",
                            "nightweather": "多云",
                            "daytemp": "27",
                            "nighttemp": "22",
                            "daywind": "东北",
                            "daypower": "3",
                        },
                        {
                            "date": "2026-06-11",
                            "week": "4",
                            "dayweather": "晴",
                            "nightweather": "晴",
                            "daytemp": "31",
                            "nighttemp": "24",
                            "daywind": "北",
                            "daypower": "≤3",
                        },
                    ],
                }
            ],
        })

    client = WeatherClient(api_key="test-amap-key", request_func=fake_request, mock_enabled=False)

    result = await client.forecast(city="杭州", days=2)

    assert result["success"] is True
    assert result["metadata"]["provider"] == "weather"
    assert result["metadata"]["mock"] is False
    assert result["data"]["city"] == "杭州市"
    assert result["data"]["adcode"] == "330100"
    assert result["data"]["forecasts"][0]["weather"] == "小雨转多云"
    assert result["data"]["forecasts"][0]["temperature"] == "22-27℃"
    assert result["data"]["forecasts"][0]["wind"] == "东北风3级"
    assert result["data"]["forecasts"][0]["suitable_for_outdoor"] is False
    assert result["data"]["forecasts"][1]["suitable_for_outdoor"] is True
    assert calls[0]["url"] == WeatherClient.FORECAST_URL
    assert calls[0]["params"]["key"] == "test-amap-key"
    assert calls[0]["params"]["city"] == "330100"
    assert calls[0]["params"]["extensions"] == "all"


@pytest.mark.asyncio
async def test_weather_client_falls_back_when_amap_business_error():
    """高德业务错误时可降级到mock天气"""

    def fake_request(method, url, params=None, headers=None, json=None, timeout=None):
        return FakeHTTPResponse({"status": "0", "info": "INVALID_USER_KEY"})

    client = WeatherClient(api_key="bad-key", request_func=fake_request, mock_enabled=True)

    result = await client.forecast(city="杭州", days=1)

    assert result["success"] is True
    assert result["metadata"]["mock"] is True
    assert result["metadata"]["mock_reason"] == "INVALID_USER_KEY"
    assert result["data"]["forecasts"][0]["weather"] == "小雨"


def test_weather_client_uses_amap_key_as_default(monkeypatch):
    """WEATHER_API_KEY为空时默认复用AMAP_API_KEY"""
    from app.config import settings

    monkeypatch.setattr(settings, "WEATHER_API_KEY", "")
    monkeypatch.setattr(settings, "AMAP_API_KEY", "amap-key-from-env")

    client = WeatherClient(api_key=None)

    assert client.client.api_key == "amap-key-from-env"
