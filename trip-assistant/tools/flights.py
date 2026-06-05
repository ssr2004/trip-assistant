"""Airport POI guidance tool backed by real AMap data only.

The tool keeps the historical name ``search_flights`` because the planner
already uses that tool slot for long-distance travel preparation. Its current
capability is intentionally narrower than real flight inventory: it returns
airport POIs and airport-pair guidance, never flight numbers, fares, seats, or
ticket availability.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.amap_client import AMapPOIClient
from tools.registry import BaseTool


class FlightTool(BaseTool):
    """Search real airport POIs. No mock flight inventory is produced."""

    AIRPORT_TYPES = "150104"

    def __init__(self, amap_client: Optional[AMapPOIClient] = None):
        self.amap_client = amap_client or AMapPOIClient(mock_enabled=False)

    @property
    def name(self) -> str:
        return "search_flights"

    @property
    def description(self) -> str:
        return "查询真实机场POI与机场衔接建议，不返回航班号、票价、舱位或余票"

    async def execute(self, origin: str = None, destination: str = None, date: str = None, **kwargs) -> Dict:
        if not origin or not destination:
            return self.error_result(
                error="查询机场出行建议需要提供出发地和目的地。",
                data={"flights": [], "airport_guidance": {}},
                metadata={
                    "source": "amap_airport_poi",
                    "provider": "amap",
                    "mock": False,
                    "capability": "airport_poi_route_guidance",
                    "real_flight_inventory": False,
                },
            )

        origin_result = await self._search_airports(origin)
        destination_result = await self._search_airports(destination)
        origin_airports = self._normalize_airports(origin_result, origin)
        destination_airports = self._normalize_airports(destination_result, destination)

        origin_metadata = origin_result.get("metadata", {}) if isinstance(origin_result, dict) else {}
        destination_metadata = destination_result.get("metadata", {}) if isinstance(destination_result, dict) else {}
        metadata = self._merge_external_metadata(origin_metadata, destination_metadata)

        if origin_airports and destination_airports:
            airport_pairs = self._build_airport_pairs(origin_airports, destination_airports)
            return self.success_result(
                data={
                    "origin": origin,
                    "destination": destination,
                    "date": date,
                    "flights": [],
                    "airport_guidance": {
                        "origin_airports": origin_airports,
                        "destination_airports": destination_airports,
                        "airport_pairs": airport_pairs,
                        "route_note": "以下为真实机场POI组成的出行衔接建议，不包含真实航班班次、票价、舱位或余票。",
                    },
                    "inventory_available": False,
                },
                metadata={
                    **metadata,
                    "source": "amap_airport_poi",
                    "provider": "amap",
                    "mock": False,
                    "capability": "airport_poi_route_guidance",
                    "real_flight_inventory": False,
                    "count": len(airport_pairs),
                },
            )

        missing_parts = []
        if not origin_airports:
            missing_parts.append("出发地机场POI")
        if not destination_airports:
            missing_parts.append("目的地机场POI")
        return self.error_result(
            error=f"未从真实高德POI中查询到{'、'.join(missing_parts)}，不使用mock补全。",
            data={
                "origin": origin,
                "destination": destination,
                "date": date,
                "flights": [],
                "airport_guidance": {
                    "origin_airports": origin_airports,
                    "destination_airports": destination_airports,
                    "airport_pairs": [],
                    "route_note": "没有完整真实机场POI时不生成航班或机场推荐。",
                },
                "inventory_available": False,
            },
            metadata={
                **metadata,
                "source": "amap_airport_poi",
                "provider": "amap",
                "mock": False,
                "capability": "airport_poi_route_guidance",
                "real_flight_inventory": False,
                "api_status": metadata.get("api_status") or "failed",
                "execution_mode": metadata.get("execution_mode") or "real_api_failed",
                "count": 0,
            },
        )

    async def _search_airports(self, city: str) -> Dict[str, Any]:
        return await self.amap_client.search_pois(
            city=city,
            keywords="机场",
            poi_types=self.AIRPORT_TYPES,
            extensions="all",
            offset=5,
        )

    def _normalize_airports(self, api_result: Dict[str, Any], city: str) -> List[Dict[str, Any]]:
        if not api_result.get("success"):
            return []
        metadata = api_result.get("metadata", {}) or {}
        if metadata.get("mock"):
            return []
        pois = self._extract_pois(api_result)
        return [self._normalize_airport_poi(poi, index, city) for index, poi in enumerate(pois[:3], start=1)]

    def _extract_pois(self, api_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = api_result.get("data", {}) if isinstance(api_result, dict) else {}
        pois = data.get("pois", []) if isinstance(data, dict) else []
        return pois if isinstance(pois, list) else []

    def _normalize_airport_poi(self, poi: Dict[str, Any], index: int, city: str) -> Dict[str, Any]:
        return {
            "id": poi.get("id") or index,
            "name": poi.get("name") or f"{city}机场",
            "address": poi.get("address") or poi.get("adname") or "地址待高德补充",
            "city": poi.get("cityname") or city,
            "district": poi.get("adname"),
            "category": self._normalize_category(poi.get("type")),
            "telephone": poi.get("tel") or None,
            "source": "amap",
            "poi_id": poi.get("id") or None,
        }

    def _normalize_category(self, poi_type: Any) -> str:
        if not poi_type:
            return "机场"
        return str(poi_type).split(";")[0] or "机场"

    def _build_airport_pairs(
        self,
        origin_airports: List[Dict[str, Any]],
        destination_airports: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        pairs = []
        for origin_airport in origin_airports[:2]:
            for destination_airport in destination_airports[:2]:
                pairs.append({
                    "origin_airport": origin_airport,
                    "destination_airport": destination_airport,
                    "route_type": "airport_poi_pair",
                    "source": "amap",
                    "note": "真实航班班次和票价需要单独接入航班库存API，本结果只表示机场出行衔接候选。",
                })
        return pairs

    def _merge_external_metadata(self, *metadata_items: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for item in metadata_items:
            merged.update(self.external_metadata(item or {}))
        if any((item or {}).get("api_status") == "success" for item in metadata_items):
            merged["api_status"] = "success"
            merged["execution_mode"] = "real_api"
        return merged
