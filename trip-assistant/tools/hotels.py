"""Hotel search tool backed by real AMap hotel POI data only."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.amap_client import AMapPOIClient
from core.mcp_client import MCPClient, create_amap_mcp_client
from tools.registry import BaseTool


class HotelTool(BaseTool):
    """Search real hotel POIs. No mock price, inventory, or room data is produced."""

    HOTEL_TYPES = "100000"
    CITY_CORE_DISTRICTS = {
        "烟台": ["芝罘区", "莱山区", "福山区", "牟平区", "开发区"],
        "杭州": ["西湖区", "上城区", "拱墅区", "滨江区", "萧山区"],
        "成都": ["锦江区", "青羊区", "武侯区", "成华区", "金牛区"],
        "厦门": ["思明区", "湖里区"],
        "三亚": ["吉阳区", "天涯区", "海棠区"],
    }
    REMOTE_DISTRICT_PENALTY = {
        "烟台": ["莱州市", "海阳市", "龙口市", "招远市", "栖霞市", "蓬莱区"],
    }

    def __init__(self, amap_client: Optional[AMapPOIClient] = None, mcp_client: Optional[MCPClient] = None):
        self.amap_client = amap_client or AMapPOIClient(mock_enabled=False)
        self.mcp_client = mcp_client or create_amap_mcp_client()

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
        budget: float = None,
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

        search_keywords = self._build_search_keywords(preferences, budget)
        mcp_result = await self._search_mcp_hotels(location, search_keywords[0])
        mcp_hotels = self._extract_mcp_hotels(mcp_result)
        if mcp_result.get("success") and mcp_hotels:
            hotels = self._select_hotels(mcp_hotels, location, source="mcp_amap")
            return self._success_hotels(
                location=location,
                checkin_date=checkin_date,
                checkout_date=checkout_date,
                hotels=hotels,
                source="mcp_amap_hotel_search",
                provider="mcp_amap",
                metadata=mcp_result.get("metadata", {}),
            )

        api_results = []
        api_metadata = {}
        for keyword in search_keywords[:3]:
            api_result = await self.amap_client.search_pois(
                city=location,
                keywords=keyword,
                poi_types=self.HOTEL_TYPES,
                extensions="all",
                offset=20,
            )
            api_results.append(api_result)
            api_metadata = api_result.get("metadata", {}) if isinstance(api_result, dict) else api_metadata
            if api_result.get("success") and self._extract_pois(api_result):
                break

        pois = []
        for api_result in api_results:
            pois.extend(self._extract_pois(api_result))

        if any(result.get("success") for result in api_results) and pois and not api_metadata.get("mock"):
            hotels = self._select_hotels(pois, location, source="amap")
            return self._success_hotels(
                location=location,
                checkin_date=checkin_date,
                checkout_date=checkout_date,
                hotels=hotels,
                source="amap_hotel_poi",
                provider="amap",
                metadata=api_metadata,
            )

        return self.error_result(
            error=self._first_error(api_results) or "未从真实高德酒店POI中查询到可用酒店结果。",
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
                "mcp_attempted": bool(mcp_result),
                "mcp_error": mcp_result.get("error") if isinstance(mcp_result, dict) else None,
                "count": 0,
            },
        )

    async def _search_mcp_hotels(self, location: str, keywords: str) -> Dict[str, Any]:
        return await self.mcp_client.call_tool(
            "maps_text_search",
            {
                "keywords": keywords,
                "city": location,
                "types": self.HOTEL_TYPES,
            },
        )

    def _success_hotels(
        self,
        location: str,
        checkin_date: str,
        checkout_date: str,
        hotels: List[Dict[str, Any]],
        source: str,
        provider: str,
        metadata: Dict[str, Any],
    ) -> Dict:
        return self.success_result(
            data={
                "location": location,
                "checkin_date": checkin_date,
                "checkout_date": checkout_date,
                "hotels": hotels,
                "inventory_available": False,
                "inventory_note": "当前酒店数据来自真实MCP/POI搜索；若数据源未返回价格、房态或房型，系统不会mock补全。",
            },
            metadata={
                **self.external_metadata(metadata),
                "source": source,
                "provider": provider,
                "mock": False,
                "capability": "hotel_search",
                "real_inventory": False,
                "count": len(hotels),
                "selection_policy": "city_core_rating_relevance",
            },
        )

    def _build_search_keywords(self, preferences: Optional[List[str]], budget: Optional[float]) -> List[str]:
        keywords = ["酒店"]
        for preference in preferences or []:
            text = str(preference).strip()
            if text:
                keywords.append(text)
        if budget:
            keywords.append(self._budget_keyword(budget))
        if not any(word in " ".join(keywords) for word in ["高评分", "市中心", "商圈", "连锁", "豪华", "经济"]):
            keywords.append("高评分")
        base = " ".join(keywords)
        candidates = [
            base,
            "酒店 市中心 高评分",
            "酒店 商圈 交通便利",
            "酒店 连锁 高评分",
        ]
        return self._dedupe(candidates)

    def _budget_keyword(self, budget: float) -> str:
        try:
            value = float(budget)
        except (TypeError, ValueError):
            return ""
        if value <= 250:
            return "经济型"
        if value <= 600:
            return "舒适型"
        return "高星级"

    def _dedupe(self, values: List[str]) -> List[str]:
        result = []
        for value in values:
            text = str(value).strip()
            if text and text not in result:
                result.append(text)
        return result

    def _extract_pois(self, api_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = api_result.get("data", {}) if isinstance(api_result, dict) else {}
        pois = data.get("pois", []) if isinstance(data, dict) else []
        return pois if isinstance(pois, list) else []

    def _extract_mcp_hotels(self, mcp_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = mcp_result.get("data", {}) if isinstance(mcp_result, dict) else {}
        for key in ["pois", "hotels", "items", "results"]:
            value = data.get(key) if isinstance(data, dict) else None
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        nested_data = data.get("data") if isinstance(data, dict) else None
        if isinstance(nested_data, dict):
            for key in ["pois", "hotels", "items", "results"]:
                value = nested_data.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []

    def _select_hotels(self, pois: List[Dict[str, Any]], location: str, source: str) -> List[Dict[str, Any]]:
        normalized = []
        seen = set()
        for poi in pois:
            poi_id = poi.get("id") or poi.get("name")
            if not poi_id or poi_id in seen:
                continue
            seen.add(poi_id)
            hotel = self._normalize_hotel_poi(poi, len(normalized) + 1, source=source)
            score, reasons = self._score_hotel(poi, hotel, location)
            if score < 0:
                continue
            hotel["quality_score"] = round(score, 2)
            hotel["selection_reason"] = "、".join(reasons[:3]) or "真实高德酒店POI"
            normalized.append(hotel)
        normalized.sort(key=lambda item: item.get("quality_score", 0), reverse=True)
        for index, hotel in enumerate(normalized[:5], start=1):
            hotel["rank"] = index
        return normalized[:5]

    def _normalize_hotel_poi(self, poi: Dict[str, Any], index: int, source: str) -> Dict[str, Any]:
        biz_ext = poi.get("biz_ext") if isinstance(poi.get("biz_ext"), dict) else {}
        rating = biz_ext.get("rating") or poi.get("rating")
        return {
            "id": poi.get("id") or index,
            "name": poi.get("name") or "酒店",
            "address": poi.get("address") or poi.get("adname") or "地址待高德补充",
            "city": poi.get("cityname"),
            "district": poi.get("adname"),
            "category": self._normalize_category(poi.get("type")),
            "rating": self._normalize_rating(rating),
            "telephone": poi.get("tel") or None,
            "source": source,
            "poi_id": poi.get("id") or None,
            "price": poi.get("price") or biz_ext.get("cost") or None,
        }

    def _score_hotel(self, poi: Dict[str, Any], hotel: Dict[str, Any], location: str) -> tuple[float, List[str]]:
        score = 0.0
        reasons = []
        name = str(hotel.get("name") or "")
        address = str(hotel.get("address") or "")
        district = str(hotel.get("district") or "")
        category = str(hotel.get("category") or "")
        city = str(hotel.get("city") or "")
        city_text = location.rstrip("市")

        if city_text and (city_text in city or city_text in address or city_text in district):
            score += 2.0
            reasons.append("目的地相关")
        if district in self.CITY_CORE_DISTRICTS.get(location, []) or district in self.CITY_CORE_DISTRICTS.get(city_text, []):
            score += 2.5
            reasons.append("核心城区")
        if district in self.REMOTE_DISTRICT_PENALTY.get(location, []) or district in self.REMOTE_DISTRICT_PENALTY.get(city_text, []):
            score -= 2.5
            reasons.append("远郊区县降权")

        rating = hotel.get("rating")
        if isinstance(rating, (int, float)):
            score += max(min((rating - 3.5) * 1.4, 2.5), 0)
            if rating >= 4.5:
                reasons.append("高评分")

        if any(keyword in name for keyword in ["酒店", "饭店", "假日", "希尔顿", "万豪", "洲际", "亚朵", "全季", "桔子", "智选", "锦江"]):
            score += 1.2
            reasons.append("酒店业态明确")
        if "宾馆" in name:
            score -= 0.4
        if "住宿服务" in category or "宾馆酒店" in str(poi.get("type") or ""):
            score += 0.8
        if hotel.get("telephone"):
            score += 0.3
        if hotel.get("price"):
            score += 0.5
            reasons.append("含真实价格字段")

        return score, reasons

    def _first_error(self, results: List[Dict[str, Any]]) -> Optional[str]:
        for result in results:
            if isinstance(result, dict) and result.get("error"):
                return result.get("error")
        return None

    def _normalize_category(self, poi_type: Any) -> str:
        if not poi_type:
            return "酒店"
        return str(poi_type).split(";")[0] or "酒店"

    def _normalize_rating(self, rating: Any) -> float | str:
        try:
            return float(rating)
        except (TypeError, ValueError):
            return "暂无评分"
