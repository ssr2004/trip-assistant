"""
路线距离工具
基于高德路线距离客户端或mock fallback优化景点访问顺序
"""
from typing import Any, Dict, List, Optional

from core.amap_route_client import AMapRouteClient
from tools.registry import BaseTool


class RouteTool(BaseTool):
    """路线优化工具"""

    def __init__(self, route_client: Optional[AMapRouteClient] = None):
        """初始化路线工具"""
        self.route_client = route_client or AMapRouteClient()

    @property
    def name(self) -> str:
        return "optimize_route_order"

    @property
    def description(self) -> str:
        return "根据距离优化景点游览顺序"

    async def execute(
        self,
        places: List[Dict[str, Any]] = None,
        start_place: str = None,
        mode: str = "walking",
        **kwargs,
    ) -> Dict[str, Any]:
        """优化景点顺序"""
        places = [place for place in (places or []) if isinstance(place, dict) and place.get("name")]
        if len(places) < 2:
            return self.error_result(
                error="路线优化至少需要两个景点",
                data={"ordered_places": places, "segments": [], "total_distance": 0, "total_duration": 0},
                metadata={"source": "amap_route_mock", "provider": "amap", "mock": True},
            )

        api_result = await self.route_client.distance_matrix(places, places, mode=mode)
        distances = self._extract_distances(api_result)
        ordered_places = self._greedy_order(places, distances, start_place)
        segments = self._build_segments(ordered_places, distances)

        total_distance = sum(segment["distance"] for segment in segments)
        total_duration = sum(segment["duration"] for segment in segments)
        api_metadata = api_result.get("metadata", {}) or {}
        is_mock = api_metadata.get("mock", False)

        return self.success_result(
            data={
                "ordered_places": ordered_places,
                "segments": segments,
                "total_distance": total_distance,
                "total_duration": total_duration,
                "mode": mode,
            },
            metadata={
                "source": "amap_route_mock" if is_mock else "amap_route",
                "provider": "amap",
                "mock": is_mock,
                "count": len(ordered_places),
            },
        )

    def _extract_distances(self, api_result: Dict[str, Any]) -> Dict[tuple[str, str], Dict[str, int]]:
        """提取距离矩阵"""
        data = api_result.get("data", {}) if isinstance(api_result, dict) else {}
        results = data.get("results", []) if isinstance(data, dict) else []
        distances = {}
        for item in results if isinstance(results, list) else []:
            origin = item.get("origin")
            destination = item.get("destination")
            if not origin or not destination:
                continue
            distances[(origin, destination)] = {
                "distance": int(item.get("distance") or 0),
                "duration": int(item.get("duration") or 0),
            }
        return distances

    def _greedy_order(
        self,
        places: List[Dict[str, Any]],
        distances: Dict[tuple[str, str], Dict[str, int]],
        start_place: Optional[str],
    ) -> List[Dict[str, Any]]:
        """使用最近邻贪心算法排序景点"""
        remaining = [dict(place) for place in places]
        start_index = self._start_index(remaining, start_place)
        ordered = [remaining.pop(start_index)]

        while remaining:
            current = ordered[-1]["name"]
            next_index = min(
                range(len(remaining)),
                key=lambda index: self._distance_between(current, remaining[index]["name"], distances),
            )
            ordered.append(remaining.pop(next_index))
        return ordered

    def _build_segments(
        self,
        ordered_places: List[Dict[str, Any]],
        distances: Dict[tuple[str, str], Dict[str, int]],
    ) -> List[Dict[str, Any]]:
        """构建相邻景点路线段"""
        segments = []
        for index in range(len(ordered_places) - 1):
            origin = ordered_places[index]["name"]
            destination = ordered_places[index + 1]["name"]
            distance_info = distances.get((origin, destination)) or distances.get((destination, origin)) or {}
            segments.append({
                "from": origin,
                "to": destination,
                "distance": int(distance_info.get("distance") or 0),
                "duration": int(distance_info.get("duration") or 0),
            })
        return segments

    def _start_index(self, places: List[Dict[str, Any]], start_place: Optional[str]) -> int:
        """定位起点索引"""
        if not start_place:
            return 0
        for index, place in enumerate(places):
            if start_place in place.get("name", "") or place.get("name", "") in start_place:
                return index
        return 0

    def _distance_between(
        self,
        origin: str,
        destination: str,
        distances: Dict[tuple[str, str], Dict[str, int]],
    ) -> int:
        """读取两点距离"""
        distance_info = distances.get((origin, destination)) or distances.get((destination, origin)) or {}
        return int(distance_info.get("distance") or 999999)
