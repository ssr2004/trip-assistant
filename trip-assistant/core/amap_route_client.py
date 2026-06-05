"""
高德路线距离客户端
封装路线距离能力，当前以mock fallback支持景点顺序优化
"""
from typing import Any, Dict, List, Optional

from app.config import settings
from core.cache import ExternalAPICache, create_external_api_cache
from core.external_api import ExternalAPIClient


class AMapRouteClient:
    """高德路线距离客户端"""

    DISTANCE_URL = "https://restapi.amap.com/v3/distance"

    def __init__(
        self,
        api_key: Optional[str] = None,
        request_func: Optional[Any] = None,
        mock_enabled: Optional[bool] = None,
        cache: Optional[ExternalAPICache] = None,
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
        self.cache = cache or create_external_api_cache()

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
        cache_params = self._cache_params(params)
        cached_result, cache_metadata = await self.cache.get("amap", "distance", cache_params)
        if cached_result:
            return self._with_cache_metadata(cached_result, cache_metadata)

        result = await self.client.request_json(
            method="GET",
            url=self.DISTANCE_URL,
            params=params,
            mock_data=self._mock_distance_response(origins, destinations, mode),
        )
        result = self._normalize_response(result, origins, destinations, mode)
        if result.get("success") and not (result.get("metadata", {}) or {}).get("mock"):
            write_metadata = await self.cache.set(
                provider="amap",
                resource="distance",
                params=cache_params,
                value=self._without_cache_metadata(result),
            )
            cache_metadata.update(write_metadata)
        else:
            cache_metadata["cache_write"] = False
        cache_metadata["cache_hit"] = False
        return self._with_cache_metadata(result, cache_metadata)

    def _normalize_response(
        self,
        result: Dict[str, Any],
        origins: List[Dict[str, Any]],
        destinations: List[Dict[str, Any]],
        mode: str,
    ) -> Dict[str, Any]:
        """将高德距离API响应标准化为路线工具可直接消费的距离矩阵。"""
        if not result.get("success"):
            return result
        metadata = result.get("metadata", {}) or {}
        if metadata.get("mock"):
            return result

        data = result.get("data", {}) or {}
        if str(data.get("status")) != "1":
            reason = data.get("info") or "amap_distance_business_error"
            if self.client.mock_enabled:
                return self.client.mock_response(
                    mock_data=self._mock_distance_response(origins, destinations, mode),
                    reason=str(reason),
                    metadata={"api_status": "degraded", "error_type": "business_error"},
                )
            return self.client.error_response(
                str(reason),
                mock=False,
                metadata={
                    "api_status": "failed",
                    "execution_mode": "real_api_failed",
                    "error_type": "business_error",
                },
            )

        normalized_results = []
        raw_results = data.get("results") or []
        for item in raw_results if isinstance(raw_results, list) else []:
            origin_index = self._safe_index(item.get("origin_id"), len(origins))
            destination_index = self._safe_index(item.get("dest_id") or item.get("destination_id"), len(destinations))
            origin = origins[origin_index] if origins else {}
            destination = destinations[destination_index] if destinations else {}
            distance = self._safe_int(item.get("distance"))
            normalized_results.append({
                "origin_id": origin.get("id") or origin_index,
                "destination_id": destination.get("id") or destination_index,
                "origin": origin.get("name"),
                "destination": destination.get("name"),
                "distance": distance,
                "duration": self._safe_int(item.get("duration")) or self._mock_duration(distance, mode),
            })

        if not normalized_results:
            if self.client.mock_enabled:
                return self.client.mock_response(
                    mock_data=self._mock_distance_response(origins, destinations, mode),
                    reason="amap_distance_empty",
                    metadata={"api_status": "degraded", "error_type": "empty_response"},
                )
            return self.client.error_response(
                "高德距离API返回为空。",
                mock=False,
                metadata={
                    "api_status": "failed",
                    "execution_mode": "real_api_failed",
                    "error_type": "empty_response",
                },
            )

        normalized = {
            "status": "1",
            "info": data.get("info") or "OK",
            "mode": mode,
            "results": normalized_results,
        }
        return self.client.success_response(data=normalized, mock=False, metadata=metadata)

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

    def _cache_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """构建不含敏感Key的缓存参数。"""
        return {key: value for key, value in params.items() if key != "key"}

    def _with_cache_metadata(self, result: Dict[str, Any], cache_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """将缓存元数据合并到外部API响应。"""
        merged = dict(result)
        metadata = dict(result.get("metadata", {}) or {})
        metadata.update(cache_metadata)
        merged["metadata"] = metadata
        return merged

    def _without_cache_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """构建可缓存响应，避免缓存命中信息被持久化。"""
        cached = dict(result)
        metadata = {
            key: value
            for key, value in (result.get("metadata", {}) or {}).items()
            if not key.startswith("cache_")
        }
        cached["metadata"] = metadata
        return cached

    def _safe_index(self, value: Any, size: int) -> int:
        """高德origin_id/dest_id通常为1-based，缺失或越界时回退到0。"""
        try:
            index = int(value) - 1
        except (TypeError, ValueError):
            return 0
        if index < 0 or index >= size:
            return 0
        return index

    def _safe_int(self, value: Any) -> int:
        """安全转换高德字符串数值。"""
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

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
