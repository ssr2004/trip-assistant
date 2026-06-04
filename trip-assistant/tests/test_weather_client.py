"""
天气客户端测试
"""
import pytest

from core.weather_client import WeatherClient


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
