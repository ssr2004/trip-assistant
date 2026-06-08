"""Train ticket search tool backed by the 12306 MCP server."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.mcp_client import MCPClient, create_12306_mcp_client
from tools.registry import BaseTool


class TrainTool(BaseTool):
    """Search real train tickets through mcp-server-12306."""

    def __init__(self, mcp_client: Optional[MCPClient] = None):
        self.mcp_client = mcp_client or create_12306_mcp_client()

    @property
    def name(self) -> str:
        return "search_trains"

    @property
    def description(self) -> str:
        return "通过12306 MCP查询真实火车/高铁车次、时间、席别、票价或余票信息"

    async def execute(self, origin: str = None, destination: str = None, date: str = None, **kwargs) -> Dict:
        if not origin or not destination:
            return self.error_result(
                error="查询火车需要提供出发地和目的地。",
                data={"trains": []},
                metadata=self._metadata(error_type="missing_route"),
            )
        if not date:
            return self.error_result(
                error="查询真实12306车次需要提供出发日期。",
                data={"origin": origin, "destination": destination, "trains": []},
                metadata=self._metadata(error_type="missing_date"),
            )

        station_context = await self._resolve_station_context(origin, destination)
        result = await self.mcp_client.call_tool(
            "query-tickets",
            {"from_station": origin, "to_station": destination, "train_date": date},
        )
        if not result.get("success"):
            return self.error_result(
                error=self._friendly_error(result.get("error") or "12306 MCP查询失败。"),
                data={"origin": origin, "destination": destination, "date": date, "trains": []},
                metadata=self._metadata(
                    result.get("metadata", {}),
                    error_type=result.get("metadata", {}).get("error_type"),
                    station_context=station_context,
                ),
            )

        payload = result.get("data") or {}
        if payload.get("success") is False:
            return self.error_result(
                error=self._friendly_error(payload.get("error") or payload.get("message") or "12306 MCP未返回有效车次。"),
                data={
                    "origin": origin,
                    "destination": destination,
                    "date": date,
                    "trains": [],
                    "station_context": station_context,
                },
                metadata=self._metadata(
                    result.get("metadata", {}),
                    count=0,
                    error_type="provider_business_error",
                    station_context=station_context,
                ),
            )

        trains = self._normalize_trains(payload)
        trains = await self._enrich_prices(origin, destination, date, trains)
        if not trains:
            return self.error_result(
                error=self._friendly_error(payload.get("message") or "12306 MCP未返回可用车次。"),
                data={
                    "origin": origin,
                    "destination": destination,
                    "date": date,
                    "trains": [],
                    "station_context": station_context,
                },
                metadata=self._metadata(result.get("metadata", {}), count=0, error_type="empty_result", station_context=station_context),
            )

        return self.success_result(
            data={
                "origin": origin,
                "destination": destination,
                "date": date,
                "trains": trains,
                "station_context": station_context,
                "data_note": "结果来自12306 MCP实时查询，不包含下单、锁票或候补能力。",
            },
            metadata=self._metadata(result.get("metadata", {}), count=len(trains), station_context=station_context),
        )

    async def _resolve_station_context(self, origin: str, destination: str) -> Dict[str, Any]:
        """Query station candidates before ticket lookup for better traceability."""
        return {
            "origin_candidates": await self._search_station_candidates(origin),
            "destination_candidates": await self._search_station_candidates(destination),
        }

    async def _search_station_candidates(self, query: str) -> List[Dict[str, Any]]:
        result = await self.mcp_client.call_tool("search-stations", {"query": query, "limit": 8})
        payload = result.get("data") or {}
        stations = payload.get("stations") if isinstance(payload, dict) else []
        if not isinstance(stations, list):
            return []
        candidates = []
        for station in stations[:8]:
            if not isinstance(station, dict):
                continue
            candidates.append({
                "name": station.get("name") or station.get("station_name") or station.get("station"),
                "code": station.get("code") or station.get("station_code") or station.get("telecode"),
                "city": station.get("city") or station.get("city_name"),
            })
        return [candidate for candidate in candidates if candidate.get("name") or candidate.get("code")]

    async def _enrich_prices(self, origin: str, destination: str, date: str, trains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not trains:
            return trains
        result = await self.mcp_client.call_tool(
            "query-ticket-price",
            {"from_station": origin, "to_station": destination, "train_date": date},
        )
        if not result.get("success"):
            return trains
        price_items = self._extract_train_items(result.get("data") or {})
        price_by_code = {
            item.get("train_code") or item.get("station_train_code") or item.get("车次"): item
            for item in price_items
            if isinstance(item, dict)
        }
        if not price_by_code:
            return trains
        enriched = []
        for train in trains:
            train_code = train.get("train_code")
            price_item = price_by_code.get(train_code)
            if not price_item:
                enriched.append(train)
                continue
            seats = dict(train.get("seats") or {})
            price = price_item.get("price") or price_item.get("prices") or price_item.get("票价")
            if price:
                seats["price"] = price
            enriched.append({**train, "seats": seats})
        return enriched

    def _normalize_trains(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_items = self._extract_train_items(payload)
        return [self._normalize_train(item, index) for index, item in enumerate(raw_items[:8], start=1)]

    def _extract_train_items(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        for key in ["trains", "tickets", "data", "items", "results"]:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if isinstance(payload.get("result"), list):
            return [item for item in payload["result"] if isinstance(item, dict)]
        return []

    def _normalize_train(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        return {
            "id": item.get("train_no") or item.get("train_code") or item.get("station_train_code") or index,
            "train_code": item.get("train_code") or item.get("station_train_code") or item.get("车次") or item.get("train_no"),
            "from_station": item.get("from_station_name") or item.get("start_station_name") or item.get("from_station") or item.get("出发站"),
            "to_station": item.get("to_station_name") or item.get("end_station_name") or item.get("to_station") or item.get("到达站"),
            "departure_time": item.get("start_time") or item.get("departure_time") or item.get("出发时间"),
            "arrival_time": item.get("arrive_time") or item.get("arrival_time") or item.get("到达时间"),
            "duration": item.get("lishi") or item.get("duration") or item.get("历时"),
            "seats": self._extract_seats(item),
            "source": "mcp_12306",
        }

    def _extract_seats(self, item: Dict[str, Any]) -> Dict[str, Any]:
        seat_aliases = {
            "business": ["business_seat", "swz_num", "商务座"],
            "first_class": ["first_class_seat", "zy_num", "一等座"],
            "second_class": ["second_class_seat", "ze_num", "二等座"],
            "hard_sleeper": ["hard_sleeper", "yw_num", "硬卧"],
            "hard_seat": ["hard_seat", "yz_num", "硬座"],
            "no_seat": ["no_seat", "wz_num", "无座"],
        }
        seats = {}
        for target, aliases in seat_aliases.items():
            for alias in aliases:
                value = item.get(alias)
                if value not in (None, "", "--"):
                    seats[target] = value
                    break
        price = item.get("price") or item.get("prices") or item.get("票价")
        if price:
            seats["price"] = price
        return seats

    def _metadata(
        self,
        mcp_metadata: Optional[Dict[str, Any]] = None,
        count: int = 0,
        error_type: Optional[str] = None,
        station_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = {
            "source": "mcp_12306",
            "provider": "mcp_12306",
            "mock": False,
            "capability": "train_ticket_search",
            "real_inventory": True,
            "count": count,
        }
        if station_context is not None:
            metadata["station_resolution_attempted"] = True
            metadata["origin_station_candidate_count"] = len(station_context.get("origin_candidates") or [])
            metadata["destination_station_candidate_count"] = len(station_context.get("destination_candidates") or [])
        for key in ["api_status", "execution_mode", "fallback_used"]:
            if (mcp_metadata or {}).get(key) is not None:
                metadata[key] = mcp_metadata[key]
        if error_type:
            metadata["error_type"] = error_type
        return metadata

    def _friendly_error(self, error: str) -> str:
        message = str(error or "")
        lower = message.lower()
        if "pywintypes" in lower or "no module named" in lower:
            return "12306 MCP本地运行环境异常，已跳过真实火车结果；可配置远程SSE MCP或修复本地依赖后重试。"
        if "input validation" in lower:
            return "12306 MCP参数校验失败，已跳过真实火车结果。"
        return message or "12306 MCP查询失败。"
