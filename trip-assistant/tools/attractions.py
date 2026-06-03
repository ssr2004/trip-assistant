"""
景点工具
提供景点搜索和推荐功能
"""
from typing import Dict, List, Optional
from tools.registry import BaseTool


class AttractionTool(BaseTool):
    """景点工具"""

    @property
    def name(self) -> str:
        return "search_attractions"

    @property
    def description(self) -> str:
        return "搜索景点信息"

    async def execute(self, location: str = None, keywords: List[str] = None, **kwargs) -> List[Dict]:
        """
        搜索景点

        Args:
            location: 位置
            keywords: 关键词

        Returns:
            景点列表
        """
        # 这里调用真实的景点API
        # 暂时返回模拟数据
        return self._get_mock_attractions(location, keywords)

    def _get_mock_attractions(self, location: str, keywords: List[str]) -> List[Dict]:
        """获取模拟景点数据"""
        mock_attractions = [
            {
                "id": 1,
                "name": "西湖",
                "location": location or "杭州",
                "category": "自然风光",
                "description": "西湖是中国著名的风景名胜区，以其秀丽的湖光山色和众多的名胜古迹闻名于世。",
                "rating": 4.9,
                "opening_hours": "全天开放",
                "ticket_price": "免费"
            },
            {
                "id": 2,
                "name": "灵隐寺",
                "location": location or "杭州",
                "category": "历史文化",
                "description": "灵隐寺是中国佛教禅宗十大古刹之一，始建于东晋咸和元年。",
                "rating": 4.7,
                "opening_hours": "07:00-18:00",
                "ticket_price": "30元"
            },
            {
                "id": 3,
                "name": "西溪国家湿地公园",
                "location": location or "杭州",
                "category": "自然风光",
                "description": "西溪湿地是罕见的城中次生湿地，生态资源丰富，自然景观质朴。",
                "rating": 4.6,
                "opening_hours": "08:00-17:30",
                "ticket_price": "80元"
            },
            {
                "id": 4,
                "name": "宋城",
                "location": location or "杭州",
                "category": "主题公园",
                "description": "宋城是一座大型宋代文化主题公园，以《宋城千古情》演出闻名。",
                "rating": 4.5,
                "opening_hours": "10:00-21:00",
                "ticket_price": "310元"
            }
        ]

        # 如果有关键词，进行过滤
        if keywords:
            filtered = []
            for attraction in mock_attractions:
                if any(kw in attraction["name"] or kw in attraction["description"] for kw in keywords):
                    filtered.append(attraction)
            return filtered if filtered else mock_attractions

        return mock_attractions
