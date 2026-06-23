"""Train ticket search tool backed by the 12306 MCP server (Joooook/12306-mcp).

该 MCP 直接返回真实车次、时刻、席别余票与票价（format=json 结构化），
且能正确处理 12306 的查票接口动态轮换（queryG/queryB...），不再出现 302 反爬失败。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.mcp_client import MCPClient, create_12306_mcp_client
from tools.registry import BaseTool


# 12306 席别短码 -> 内部统一 key（与 response_builder 渲染一致）
_SEAT_SHORT_TO_KEY = {
    "swz": "business",      # 商务座
    "zy": "first_class",    # 一等座
    "ze": "second_class",   # 二等座
    "wz": "no_seat",        # 无座
    "yw": "hard_sleeper",   # 硬卧
    "rw": "soft_sleeper",   # 软卧
    "yz": "hard_seat",      # 硬座
    "rz": "soft_seat",      # 软座
}


class TrainTool(BaseTool):
    """通过 12306 MCP（Joooook/12306-mcp）查询真实火车/高铁车次、时刻、席别余票与票价。"""

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
            "get-tickets",
            {
                "date": date,
                "fromStation": origin,
                "toStation": destination,
                "format": "json",
                "limitedNum": 8,
            },
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
        items = payload.get("items") if isinstance(payload, dict) else []
        if not isinstance(items, list):
            items = []
        trains = [
            self._normalize_train(item, index)
            for index, item in enumerate(items[:8], start=1)
        ]
        trains = [train for train in trains if train]
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
                metadata=self._metadata(
                    result.get("metadata", {}),
                    count=0,
                    error_type="empty_result",
                    station_context=station_context,
                ),
            )

        return self.success_result(
            data={
                "origin": origin,
                "destination": destination,
                "date": date,
                "trains": trains,
                "station_context": station_context,
                "data_note": "结果来自12306 MCP实时查询（含真实余票与票价），不包含下单、锁票或候补能力。",
            },
            metadata=self._metadata(
                result.get("metadata", {}),
                count=len(trains),
                station_context=station_context,
            ),
        )

    async def _resolve_station_context(self, origin: str, destination: str) -> Dict[str, Any]:
        """一次性解析出发/目的地城市站点码，用于可观测性与兜底校验。"""
        try:
            result = await self.mcp_client.call_tool(
                "get-station-code-of-citys",
                {"citys": f"{origin}|{destination}"},
            )
        except Exception:
            return {"origin_candidates": [], "destination_candidates": []}
        data = result.get("data") if result.get("success") else {}
        if not isinstance(data, dict):
            data = {}
        return {
            "origin_candidates": self._city_stations(data, origin),
            "destination_candidates": self._city_stations(data, destination),
        }

    @staticmethod
    def _city_stations(data: Dict[str, Any], city: str) -> List[Dict[str, Any]]:
        entry = data.get(city) if isinstance(data.get(city), dict) else None
        if not entry:
            return []
        return [{
            "name": entry.get("station_name") or city,
            "code": entry.get("station_code"),
            "city": city,
        }]

    def _normalize_train(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        if not isinstance(item, dict):
            return {}
        return {
            "id": item.get("train_no") or item.get("start_train_code") or index,
            "train_code": item.get("start_train_code") or item.get("train_no"),
            "from_station": item.get("from_station"),
            "to_station": item.get("to_station"),
            "departure_time": item.get("start_time"),
            "arrival_time": item.get("arrive_time"),
            "duration": item.get("lishi"),
            "seats": self._extract_seats(item.get("prices") or []),
            "flags": item.get("dw_flag") or [],
            "source": "mcp_12306",
        }

    @staticmethod
    def _extract_seats(prices: List[Dict[str, Any]]) -> Dict[str, Any]:
        seats: Dict[str, Any] = {}
        for price_item in prices:
            if not isinstance(price_item, dict):
                continue
            key = _SEAT_SHORT_TO_KEY.get(price_item.get("short"))
            if not key:
                continue
            num = price_item.get("num")
            price = price_item.get("price")
            availability = (
                f"剩余{num}张"
                if str(num).isdigit()
                else (str(num) if num not in (None, "", "--") else "")
            )
            if price not in (None, "", "--"):
                seats[key] = f"{availability} ¥{price}".strip()
            elif availability:
                seats[key] = availability
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
