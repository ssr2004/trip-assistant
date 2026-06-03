"""
酒店工具
提供酒店搜索和预订功能
"""
from typing import Dict, List
from tools.registry import BaseTool


class HotelTool(BaseTool):
    """酒店工具"""

    @property
    def name(self) -> str:
        return "search_hotels"

    @property
    def description(self) -> str:
        return "搜索酒店信息"

    async def execute(self, location: str = None, checkin_date: str = None, checkout_date: str = None, **kwargs) -> Dict:
        """
        搜索酒店

        Args:
            location: 位置
            checkin_date: 入住日期
            checkout_date: 退房日期

        Returns:
            标准化酒店搜索结果
        """
        if not location:
            return self.error_result(
                error="查询酒店需要提供目的地或入住城市",
                data={"hotels": []},
                metadata={"source": "mock_hotel_data"},
            )

        hotels = self._get_mock_hotels(location, checkin_date, checkout_date)
        return self.success_result(
            data={
                "location": location,
                "checkin_date": checkin_date,
                "checkout_date": checkout_date,
                "hotels": hotels,
            },
            metadata={
                "source": "mock_hotel_data",
                "count": len(hotels),
            },
        )

    def _get_mock_hotels(self, location: str, checkin_date: str, checkout_date: str) -> List[Dict]:
        """获取模拟酒店数据"""
        mock_hotels = [
            {
                "id": 1,
                "name": "杭州西湖国宾馆",
                "location": location,
                "address": "杭州市西湖区杨公堤18号",
                "price_per_night": 1200,
                "rating": 4.8,
                "star": 5,
                "amenities": ["免费WiFi", "游泳池", "健身房", "餐厅"]
            },
            {
                "id": 2,
                "name": "杭州西溪湿地公园酒店",
                "location": location,
                "address": "杭州市西湖区天目山路518号",
                "price_per_night": 800,
                "rating": 4.5,
                "star": 4,
                "amenities": ["免费WiFi", "餐厅", "会议室"]
            },
            {
                "id": 3,
                "name": "杭州滨江银泰喜来登大酒店",
                "location": location,
                "address": "杭州市滨江区江南大道288号",
                "price_per_night": 600,
                "rating": 4.3,
                "star": 4,
                "amenities": ["免费WiFi", "健身房", "餐厅"]
            }
        ]
        return mock_hotels
