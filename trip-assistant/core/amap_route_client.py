"""
高德路线距离客户端
封装路线距离能力，当前以mock fallback支持景点顺序优化
"""
from typing import Any, Dict, List, Optional

from app.config import settings
from core.external_api import ExternalAPIClient


class AMapRouteClient:
    """高德路线距离客户端"""

    DISTANCE_URL = "https://restapi.amap.com/v3/distance"

    def __init__(
        self,
        api_key: Optional[str] = None,
        request_func: Optional[Any] = None,
        mock_enabled: Optional[bool] = None,
    ):
        """初始化高德路线客户端"""
        resolved_key = settings.AMAP_API_KEY if api_key is None else api_key
        self.client = ExternalAPIClient(
            name="amap",
            api_key=resolved_key,
            timeout=settings.EXTERNAL_API_TIMEOUT,
            retry_times=settings.EXTERNAL_API_RETRY_TIMES,
            mock_enabled=mock_enabled,
            request_func=request_func,
        )

    async def distance_matrix(
        self,
        origins: List[Dict[str, Any]],
        destinations: List[Dict[str, Any]],
        mode: str = "walking",
    ) -> Dict[str, Any]:
        """计算起终点距离矩阵，未配置Key时使用mock fallback"""
        params = {
            "key": self.client.api_key,
            "origins": "|".join(self._location(place) for place in origins),
            "destination": "|".join(self._location(place) for place in destinations),
            "type": self._mode_type(mode),
        }
        return await self.client.request_json(
            method="GET",
            url=self.DISTANCE_URL,
            params=params,
            mock_data=self._mock_distance_response(origins, destinations, mode),
        )

    def _mock_distance_response(
        self,
        origins: List[Dict[str, Any]],
        destinations: List[Dict[str, Any]],
        mode: str,
    ) -> Dict[str, Any]:
        """构建路线距离mock响应"""
        results = []
        for origin_index, origin in enumerate(origins):
            for destination_index, destination in enumerate(destinations):
                distance = self._mock_distance(origin.get("name"), destination.get("name"))
                results.append({
                    "origin_id": origin.get("id") or origin_index,
                    "destination_id": destination.get("id") or destination_index,
                    "origin": origin.get("name"),
                    "destination": destination.get("name"),
                    "distance": distance,
                    "duration": self._mock_duration(distance, mode),
                })
        return {
            "status": "1",
            "info": "MOCK",
            "mode": mode,
            "results": results,
        }

    def _mock_distance(self, origin: str, destination: str) -> int:
        """返回杭州核心景点间mock距离，单位米"""
        if origin == destination:
            return 0
        pair = frozenset([origin, destination])
        distances = {
            frozenset(["西湖", "灵隐寺"]): 6200,
            frozenset(["西湖", "西溪国家湿地公园"]): 9800,
            frozenset(["西湖", "宋城"]): 14200,
            frozenset(["灵隐寺", "西溪国家湿地公园"]): 5800,
            frozenset(["灵隐寺", "宋城"]): 13200,
            frozenset(["西溪国家湿地公园", "宋城"]): 16800,
            frozenset(["鼓浪屿", "南普陀寺"]): 5200,
            frozenset(["鼓浪屿", "环岛路"]): 7600,
            frozenset(["亚龙湾", "蜈支洲岛"]): 22500,
        }
        return distances.get(pair, 10000)

    def _mock_duration(self, distance: int, mode: str) -> int:
        """根据距离和交通方式估算耗时，单位秒"""
        speed = {
            "walking": 1.4,
            "driving": 8.0,
            "transit": 5.0,
        }.get(mode, 1.4)
        return int(distance / speed)

    def _location(self, place: Dict[str, Any]) -> str:
        """提取地点坐标"""
        return str(place.get("location") or place.get("name") or "")

    def _mode_type(self, mode: str) -> str:
        """映射高德距离API类型"""
        mode_types = {
            "driving": "1",
            "walking": "3",
            "transit": "2",
        }
        return mode_types.get(mode, "3")
