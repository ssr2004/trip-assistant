"""Hotel search tool backed by real AMap hotel POI data only."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.amap_client import AMapPOIClient
from tools.registry import BaseTool


class HotelTool(BaseTool):
    """Search real hotel POIs. No mock price, inventory, or room data is produced."""

    HOTEL_TYPES = "100000"

    def __init__(self, amap_client: Optional[AMapPOIClient] = None):
        self.amap_client = amap_client or AMapPOIClient(mock_enabled=False)

    @property
    def name(self) -> str:
        return "search_hotels"

    @property
    def description(self) -> str:
        return "搜索真实酒店POI信息，不返回模拟价格、库存或房型"

    async def execute(
        self,
        location: str = None,
        checkin_date: str = None,
        checkout_date: str = None,
        preferences: List[str] = None,
        **kwargs,
    ) -> Dict:
        if not location:
            return self.error_result(
                error="查询酒店需要提供目的地或入住城市。",
                data={"hotels": []},
                metadata={
                    "source": "amap_hotel_poi",
                    "provider": "amap",
                    "mock": False,
                    "capability": "hotel_poi_search",
                    "real_inventory": False,
                },
            )

        api_result = await self.amap_client.search_pois(
            city=location,
            keywords=self._build_keywords(preferences),
            poi_types=self.HOTEL_TYPES,
            extensions="all",
            offset=10,
        )
        pois = self._extract_pois(api_result)
        api_metadata = api_result.get("metadata", {}) if isinstance(api_result, dict) else {}

        if api_result.get("success") and pois and not api_metadata.get("mock"):
            hotels = [self._normalize_hotel_poi(poi, index) for index, poi in enumerate(pois[:5], start=1)]
            return self.success_result(
                data={
                    "location": location,
                    "checkin_date": checkin_date,
                    "checkout_date": checkout_date,
                    "hotels": hotels,
                    "inventory_available": False,
                    "inventory_note": "高德POI只提供真实酒店地点信息，不提供房态、价格、房型或可订库存。",
                },
                metadata={
                    **self.external_metadata(api_metadata),
                    "source": "amap_hotel_poi",
                    "provider": "amap",
                    "mock": False,
                    "capability": "hotel_poi_search",
                    "real_inventory": False,
                    "count": len(hotels),
                },
            )

        return self.error_result(
            error=api_result.get("error") or "未从真实高德酒店POI中查询到可用酒店结果。",
            data={
                "location": location,
                "checkin_date": checkin_date,
                "checkout_date": checkout_date,
                "hotels": [],
                "inventory_available": False,
                "inventory_note": "没有真实酒店POI结果时不使用mock补全。",
            },
            metadata={
                **self.external_metadata(api_metadata),
                "source": "amap_hotel_poi",
                "provider": "amap",
                "mock": False,
                "capability": "hotel_poi_search",
                "real_inventory": False,
                "api_status": api_metadata.get("api_status") or "failed",
                "execution_mode": api_metadata.get("execution_mode") or "real_api_failed",
                "count": 0,
            },
        )

    def _build_keywords(self, preferences: Optional[List[str]]) -> str:
        keywords = ["酒店"]
        for preference in preferences or []:
            text = str(preference).strip()
            if text:
                keywords.append(text)
        return " ".join(keywords)

    def _extract_pois(self, api_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = api_result.get("data", {}) if isinstance(api_result, dict) else {}
        pois = data.get("pois", []) if isinstance(data, dict) else []
        return pois if isinstance(pois, list) else []

    def _normalize_hotel_poi(self, poi: Dict[str, Any], index: int) -> Dict[str, Any]:
        biz_ext = poi.get("biz_ext") if isinstance(poi.get("biz_ext"), dict) else {}
        rating = biz_ext.get("rating")
        return {
            "id": poi.get("id") or index,
            "name": poi.get("name") or "酒店",
            "address": poi.get("address") or poi.get("adname") or "地址待高德补充",
            "city": poi.get("cityname"),
            "district": poi.get("adname"),
            "category": self._normalize_category(poi.get("type")),
            "rating": self._normalize_rating(rating),
            "telephone": poi.get("tel") or None,
            "source": "amap",
            "poi_id": poi.get("id") or None,
        }

    def _normalize_category(self, poi_type: Any) -> str:
        if not poi_type:
            return "酒店"
        return str(poi_type).split(";")[0] or "酒店"

    def _normalize_rating(self, rating: Any) -> float | str:
        try:
            return float(rating)
        except (TypeError, ValueError):
            return "暂无评分"
