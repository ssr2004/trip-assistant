"""
天气工具测试
"""
import pytest

from core.weather_client import WeatherClient
from tools.weather import WeatherTool


@pytest.mark.asyncio
async def test_weather_tool_returns_standard_result():
    """天气工具返回标准化结构和旅行建议"""
    tool = WeatherTool(weather_client=WeatherClient(api_key="", mock_enabled=True))

    result = await tool.execute(city="杭州", days=3)

    assert result["success"] is True
    assert result["metadata"]["tool"] == "get_weather_forecast"
    assert result["metadata"]["source"] == "weather_mock"
    assert result["metadata"]["mock"] is True
    assert result["data"]["city"] == "杭州"
    assert len(result["data"]["forecasts"]) == 3
    assert result["data"]["travel_advice"]
    assert "减少长时间户外景点" in result["data"]["travel_advice"][0]


@pytest.mark.asyncio
async def test_weather_tool_requires_city():
    """天气工具缺少城市时返回失败结构"""
    tool = WeatherTool(weather_client=WeatherClient(api_key="", mock_enabled=True))

    result = await tool.execute()

    assert result["success"] is False
    assert "城市" in result["error"]
    assert result["data"]["forecasts"] == []
