"""
天气客户端
封装天气预报能力，支持无Key时mock fallback
"""
from typing import Any, Callable, Dict, List, Optional

from app.config import settings
from core.external_api import ExternalAPIClient


class WeatherClient:
    """天气预报客户端"""

    FORECAST_URL = "https://api.weather.example.com/v1/forecast"

    def __init__(
        self,
        api_key: Optional[str] = None,
        request_func: Optional[Callable[..., Any]] = None,
        mock_enabled: Optional[bool] = None,
    ):
        """初始化天气客户端"""
        resolved_key = settings.WEATHER_API_KEY if api_key is None else api_key
        self.client = ExternalAPIClient(
            name="weather",
            api_key=resolved_key,
            timeout=settings.EXTERNAL_API_TIMEOUT,
            retry_times=settings.EXTERNAL_API_RETRY_TIMES,
            mock_enabled=mock_enabled,
            request_func=request_func,
        )

    async def forecast(self, city: str, days: int = 3) -> Dict[str, Any]:
        """查询城市天气预报"""
        days = max(1, min(int(days or 3), 7))
        params = {
            "key": self.client.api_key,
            "city": city,
            "days": days,
        }
        return await self.client.request_json(
            method="GET",
            url=self.FORECAST_URL,
            params=params,
            mock_data=self._mock_forecast(city, days),
        )

    def _mock_forecast(self, city: str, days: int) -> Dict[str, Any]:
        """构建天气mock预报"""
        forecasts = self._city_forecasts(city)[:days]
        return {
            "city": city,
            "forecasts": forecasts,
        }

    def _city_forecasts(self, city: str) -> List[Dict[str, Any]]:
        """按城市返回天气mock数据"""
        city_forecasts = {
            "杭州": [
                self._forecast("2026-06-10", "小雨", "22-27℃", "东北风3级", False),
                self._forecast("2026-06-11", "多云", "23-29℃", "微风", True),
                self._forecast("2026-06-12", "晴", "24-31℃", "微风", True),
            ],
            "成都": [
                self._forecast("2026-06-10", "阴", "21-28℃", "微风", True),
                self._forecast("2026-06-11", "小雨", "20-26℃", "微风", False),
                self._forecast("2026-06-12", "多云", "22-29℃", "微风", True),
            ],
            "厦门": [
                self._forecast("2026-06-10", "多云", "25-31℃", "东南风3级", True),
                self._forecast("2026-06-11", "阵雨", "24-30℃", "东南风4级", False),
                self._forecast("2026-06-12", "晴", "26-32℃", "微风", True),
            ],
            "三亚": [
                self._forecast("2026-06-10", "晴", "27-33℃", "东南风3级", True),
                self._forecast("2026-06-11", "雷阵雨", "26-31℃", "东南风4级", False),
                self._forecast("2026-06-12", "多云", "27-32℃", "微风", True),
            ],
        }
        return city_forecasts.get(city, city_forecasts["杭州"])

    def _forecast(
        self,
        date: str,
        weather: str,
        temperature: str,
        wind: str,
        suitable_for_outdoor: bool,
    ) -> Dict[str, Any]:
        """构建单天天气预报"""
        return {
            "date": date,
            "weather": weather,
            "temperature": temperature,
            "wind": wind,
            "suitable_for_outdoor": suitable_for_outdoor,
        }
