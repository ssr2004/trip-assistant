"""
航班工具
提供航班搜索和预订功能
"""
from typing import Dict, List, Optional
from tools.registry import BaseTool


class FlightTool(BaseTool):
    """航班工具"""

    @property
    def name(self) -> str:
        return "search_flights"

    @property
    def description(self) -> str:
        return "搜索航班信息"

    async def execute(self, origin: str = None, destination: str = None, date: str = None, **kwargs) -> List[Dict]:
        """
        搜索航班

        Args:
            origin: 出发地
            destination: 目的地
            date: 出发日期

        Returns:
            航班列表
        """
        # 这里调用真实的航班API
        # 暂时返回模拟数据
        return self._get_mock_flights(origin, destination, date)

    def _get_mock_flights(self, origin: str, destination: str, date: str) -> List[Dict]:
        """获取模拟航班数据"""
        # 模拟数据
        mock_flights = [
            {
                "flight_no": "MU1234",
                "airline": "东方航空",
                "departure_airport": origin or "郑州新郑国际机场",
                "arrival_airport": destination or "杭州萧山国际机场",
                "departure_time": f"{date} 08:00",
                "arrival_time": f"{date} 10:30",
                "price": 680,
                "cabin_class": "经济舱"
            },
            {
                "flight_no": "CZ5678",
                "airline": "南方航空",
                "departure_airport": origin or "郑州新郑国际机场",
                "arrival_airport": destination or "杭州萧山国际机场",
                "departure_time": f"{date} 10:30",
                "arrival_time": f"{date} 13:00",
                "price": 720,
                "cabin_class": "经济舱"
            },
            {
                "flight_no": "CA9012",
                "airline": "国际航空",
                "departure_airport": origin or "郑州新郑国际机场",
                "arrival_airport": destination or "杭州萧山国际机场",
                "departure_time": f"{date} 14:00",
                "arrival_time": f"{date} 16:30",
                "price": 850,
                "cabin_class": "商务舱"
            }
        ]
        return mock_flights
