"""
天气客户端
封装天气预报能力，支持高德天气API和无Key mock fallback
"""
from typing import Any, Callable, Dict, List, Optional

from app.config import settings
from core.external_api import ExternalAPIClient


class WeatherClient:
    """天气预报客户端"""

    FORECAST_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
    CITY_ADCODES = {
        "北京": "110000",
        "上海": "310000",
        "广州": "440100",
        "深圳": "440300",
        "杭州": "330100",
        "成都": "510100",
        "厦门": "350200",
        "三亚": "460200",
        "郑州": "410100",
        "南京": "320100",
        "苏州": "320500",
        "西安": "610100",
        "重庆": "500000",
        "武汉": "420100",
        "长沙": "430100",
        "青岛": "370200",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        request_func: Optional[Callable[..., Any]] = None,
        mock_enabled: Optional[bool] = None,
    ):
        """初始化天气客户端"""
        resolved_key = (settings.WEATHER_API_KEY or settings.AMAP_API_KEY) if api_key is None else api_key
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
            "city": self._resolve_city_code(city),
            "extensions": "all",
            "output": "JSON",
        }
        result = await self.client.request_json(
            method="GET",
            url=self.FORECAST_URL,
            params=params,
            mock_data=self._mock_forecast(city, days),
        )
        return self._normalize_response(result, city, days)

    def _resolve_city_code(self, city: str) -> str:
        """将常见城市名转换为高德adcode，未命中时保留原值"""
        cleaned_city = str(city or "").strip()
        return self.CITY_ADCODES.get(cleaned_city, cleaned_city)

    def _normalize_response(self, result: Dict[str, Any], city: str, days: int) -> Dict[str, Any]:
        """将高德天气响应转换为项目统一天气结构"""
        if not result.get("success"):
            return result

        metadata = result.get("metadata", {}) or {}
        if metadata.get("mock"):
            return result

        data = result.get("data", {}) or {}
        if str(data.get("status")) != "1":
            reason = data.get("info") or "amap_weather_business_error"
            if self.client.mock_enabled:
                return self.client.mock_response(mock_data=self._mock_forecast(city, days), reason=reason)
            return self.client.error_response(str(reason), mock=False)

        forecasts = data.get("forecasts") or []
        if not forecasts:
            if self.client.mock_enabled:
                return self.client.mock_response(mock_data=self._mock_forecast(city, days), reason="amap_weather_empty")
            return self.client.error_response("高德天气返回为空。", mock=False)

        forecast_block = forecasts[0] or {}
        casts = forecast_block.get("casts") or []
        normalized_forecasts = [self._normalize_cast(cast) for cast in casts[:days]]
        normalized = {
            "city": forecast_block.get("city") or city,
            "adcode": forecast_block.get("adcode") or self._resolve_city_code(city),
            "province": forecast_block.get("province"),
            "report_time": forecast_block.get("reporttime"),
            "forecasts": normalized_forecasts,
        }
        return self.client.success_response(data=normalized, mock=False, metadata=metadata)

    def _normalize_cast(self, cast: Dict[str, Any]) -> Dict[str, Any]:
        """标准化高德单日天气预报"""
        day_weather = cast.get("dayweather") or "天气待定"
        night_weather = cast.get("nightweather")
        weather = day_weather if not night_weather or night_weather == day_weather else f"{day_weather}转{night_weather}"
        temperature = self._format_temperature(cast.get("nighttemp"), cast.get("daytemp"))
        wind = self._format_wind(cast.get("daywind"), cast.get("daypower"))
        return {
            "date": cast.get("date") or "日期待定",
            "week": cast.get("week"),
            "weather": weather,
            "temperature": temperature,
            "wind": wind,
            "suitable_for_outdoor": self._is_suitable_for_outdoor(weather),
        }

    def _format_temperature(self, low: Any, high: Any) -> str:
        """格式化温度范围"""
        if low in (None, "") and high in (None, ""):
            return "温度待定"
        if low in (None, ""):
            return f"{high}℃"
        if high in (None, ""):
            return f"{low}℃"
        return f"{low}-{high}℃"

    def _format_wind(self, wind: Any, power: Any) -> str:
        """格式化风力信息"""
        if wind and power:
            return f"{wind}风{power}级"
        if wind:
            return f"{wind}风"
        if power:
            return f"{power}级"
        return "风力待定"

    def _is_suitable_for_outdoor(self, weather: str) -> bool:
        """根据天气文本判断是否适合长时间户外活动"""
        bad_weather_keywords = ["雨", "雪", "雷", "暴", "冰雹", "沙尘", "雾霾"]
        return not any(keyword in str(weather) for keyword in bad_weather_keywords)

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
