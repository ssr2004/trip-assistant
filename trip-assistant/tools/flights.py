"""
航班工具
提供航班搜索和预订功能
"""
from typing import Dict, List
from tools.registry import BaseTool


class FlightTool(BaseTool):
    """航班工具"""

    @property
    def name(self) -> str:
        return "search_flights"

    @property
    def description(self) -> str:
        return "搜索航班信息"

    async def execute(self, origin: str = None, destination: str = None, date: str = None, **kwargs) -> Dict:
        """
        搜索航班

        Args:
            origin: 出发地
            destination: 目的地
            date: 出发日期

        Returns:
            标准化航班搜索结果
        """
        if not origin or not destination:
            return {
                "success": False,
                "data": {"flights": []},
                "error": "查询航班需要提供出发地和目的地",
                "metadata": {
                    "source": "mock_flight_data",
                    "tool": self.name,
                },
            }

        flights = self._get_mock_flights(origin, destination, date)
        return {
            "success": True,
            "data": {
                "origin": origin,
                "destination": destination,
                "date": date,
                "flights": flights,
            },
            "error": None,
            "metadata": {
                "source": "mock_flight_data",
                "tool": self.name,
                "count": len(flights),
            },
        }

    def _get_mock_flights(self, origin: str, destination: str, date: str) -> List[Dict]:
        """获取模拟航班数据"""
        departure_date = date or "日期待定"
        mock_flights = [
            {
                "flight_no": "MU1234",
                "airline": "东方航空",
                "departure_airport": origin,
                "arrival_airport": destination,
                "departure_time": f"{departure_date} 08:00",
                "arrival_time": f"{departure_date} 10:30",
                "price": 680,
                "cabin_class": "经济舱"
            },
            {
                "flight_no": "CZ5678",
                "airline": "南方航空",
                "departure_airport": origin,
                "arrival_airport": destination,
                "departure_time": f"{departure_date} 10:30",
                "arrival_time": f"{departure_date} 13:00",
                "price": 720,
                "cabin_class": "经济舱"
            },
            {
                "flight_no": "CA9012",
                "airline": "国际航空",
                "departure_airport": origin,
                "arrival_airport": destination,
                "departure_time": f"{departure_date} 14:00",
                "arrival_time": f"{departure_date} 16:30",
                "price": 850,
                "cabin_class": "商务舱"
            }
        ]
        return mock_flights
