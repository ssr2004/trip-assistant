"""
天气工具
提供城市天气预报和旅行天气建议
"""
from typing import Any, Dict, List, Optional

from core.weather_client import WeatherClient
from tools.registry import BaseTool


class WeatherTool(BaseTool):
    """天气预报工具"""

    def __init__(self, weather_client: Optional[WeatherClient] = None):
        """初始化天气工具"""
        self.weather_client = weather_client or WeatherClient()

    @property
    def name(self) -> str:
        return "get_weather_forecast"

    @property
    def description(self) -> str:
        return "查询目的地天气预报"

    async def execute(self, city: str = None, days: int = 3, **kwargs) -> Dict[str, Any]:
        """查询天气预报"""
        if not city:
            return self.error_result(
                error="查询天气需要提供城市",
                data={"city": None, "forecasts": [], "travel_advice": []},
                metadata={"source": "weather_mock", "provider": "weather", "mock": True},
            )

        api_result = await self.weather_client.forecast(city=city, days=days)
        if not api_result.get("success"):
            return self.error_result(
                error=api_result.get("error") or "天气查询失败",
                data={"city": city, "forecasts": [], "travel_advice": []},
                metadata={"source": "weather_api", "provider": "weather", "mock": False},
            )

        data = api_result.get("data", {}) or {}
        forecasts = data.get("forecasts", []) if isinstance(data, dict) else []
        api_metadata = api_result.get("metadata", {}) or {}
        is_mock = api_metadata.get("mock", False)
        return self.success_result(
            data={
                "city": data.get("city") or city,
                "forecasts": forecasts,
                "travel_advice": self._build_travel_advice(forecasts),
            },
            metadata={
                "source": "weather_mock" if is_mock else "weather_api",
                "provider": "weather",
                "mock": is_mock,
                "count": len(forecasts),
            },
        )

    def _build_travel_advice(self, forecasts: List[Dict[str, Any]]) -> List[str]:
        """根据天气构建旅行建议"""
        advice = []
        for forecast in forecasts or []:
            date = forecast.get("date", "日期待定")
            weather = forecast.get("weather", "天气待定")
            if forecast.get("suitable_for_outdoor") is False:
                advice.append(f"{date} {weather}，建议携带雨具，减少长时间户外景点，优先安排室内或低强度活动。")
            else:
                advice.append(f"{date} {weather}，整体适合户外游览，注意防晒和补水。")
        return advice
