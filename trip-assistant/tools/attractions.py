"""
景点工具
提供景点搜索和推荐功能
"""
from typing import Dict, List, Optional

from core.amap_client import AMapPOIClient
from rag.document_adapter import RAGDocumentAdapter
from tools.registry import BaseTool


class AttractionTool(BaseTool):
    """景点工具"""

    def __init__(self, amap_client: Optional[AMapPOIClient] = None):
        """初始化景点工具"""
        self.amap_client = amap_client or AMapPOIClient()
        self.rag_adapter = RAGDocumentAdapter()

    @property
    def name(self) -> str:
        return "search_attractions"

    @property
    def description(self) -> str:
        return "搜索景点信息"

    async def execute(self, location: str = None, keywords: List[str] = None, **kwargs) -> Dict:
        """
        搜索景点

        Args:
            location: 位置
            keywords: 关键词

        Returns:
            标准化景点搜索结果
        """
        if not location:
            return self.error_result(
                error="查询景点需要提供目的地或城市",
                data={"attractions": []},
                metadata={"source": "mock_attraction_data"},
            )

        api_result = await self.amap_client.search_pois(
            city=location,
            keywords=self._build_keywords(keywords),
        )
        pois = self._extract_pois(api_result)
        if api_result.get("success") and pois:
            selected_pois = pois[:4]
            attractions = [self._normalize_poi(poi, index) for index, poi in enumerate(selected_pois, start=1)]
            rag_documents = [self.rag_adapter.from_amap_poi(poi) for poi in selected_pois]
            api_metadata = api_result.get("metadata", {}) or {}
            is_mock = api_metadata.get("mock", False)
            return self.success_result(
                data={
                    "location": location,
                    "keywords": keywords or [],
                    "attractions": attractions,
                    "rag_documents": rag_documents,
                },
                metadata={
                    "source": "amap_poi_mock" if is_mock else "amap_poi",
                    "provider": "amap",
                    "mock": is_mock,
                    "count": len(attractions),
                },
            )

        attractions = self._get_mock_attractions(location, keywords)
        return self.success_result(
            data={
                "location": location,
                "keywords": keywords or [],
                "attractions": attractions,
                "rag_documents": [],
            },
            metadata={
                "source": "mock_attraction_data",
                "provider": "local",
                "mock": True,
                "fallback_reason": api_result.get("error") or "amap_poi_empty",
                "count": len(attractions),
            },
        )

    def _build_keywords(self, keywords: Optional[List[str]]) -> str:
        """构建高德POI检索关键词"""
        cleaned_keywords = [str(keyword).strip() for keyword in (keywords or []) if str(keyword).strip()]
        if not cleaned_keywords:
            return "景点"
        return " ".join(["景点", *cleaned_keywords])

    def _extract_pois(self, api_result: Dict) -> List[Dict]:
        """从高德响应中提取POI列表"""
        data = api_result.get("data", {}) if isinstance(api_result, dict) else {}
        pois = data.get("pois", []) if isinstance(data, dict) else []
        return pois if isinstance(pois, list) else []

    def _normalize_poi(self, poi: Dict, index: int) -> Dict:
        """将高德POI标准化为项目景点结构"""
        category = self._normalize_category(poi.get("type"))
        address = poi.get("address") or poi.get("adname") or "地址待定"
        return {
            "id": poi.get("id") or index,
            "name": poi.get("name", "景点"),
            "location": poi.get("location") or poi.get("cityname") or "坐标待定",
            "address": address,
            "category": category,
            "description": f"来自高德POI：{address}。",
            "rating": self._normalize_rating(poi),
            "opening_hours": "以景区公告为准",
            "ticket_price": "待定",
            "source": "amap",
        }

    def _normalize_category(self, poi_type: str) -> str:
        """提取POI分类展示文本"""
        if not poi_type:
            return "风景名胜"
        return str(poi_type).split(";")[0] or "风景名胜"

    def _normalize_rating(self, poi: Dict) -> float:
        """提取POI评分，缺失时给出展示默认值"""
        biz_ext = poi.get("biz_ext") if isinstance(poi, dict) else {}
        rating = biz_ext.get("rating") if isinstance(biz_ext, dict) else None
        try:
            return float(rating)
        except (TypeError, ValueError):
            return 4.6

    def _get_mock_attractions(self, location: str, keywords: List[str]) -> List[Dict]:
        """获取本地兜底模拟景点数据"""
        mock_attractions = [
            {
                "id": 1,
                "name": "西湖",
                "location": location,
                "category": "自然风光",
                "description": "西湖是中国著名的风景名胜区，以其秀丽的湖光山色和众多的名胜古迹闻名于世。",
                "rating": 4.9,
                "opening_hours": "全天开放",
                "ticket_price": "免费",
                "source": "local_mock",
            },
            {
                "id": 2,
                "name": "灵隐寺",
                "location": location,
                "category": "历史文化",
                "description": "灵隐寺是中国佛教禅宗十大古刹之一，始建于东晋咸和元年。",
                "rating": 4.7,
                "opening_hours": "07:00-18:00",
                "ticket_price": "30元",
                "source": "local_mock",
            },
            {
                "id": 3,
                "name": "西溪国家湿地公园",
                "location": location,
                "category": "自然风光",
                "description": "西溪湿地是罕见的城中次生湿地，生态资源丰富，自然景观质朴。",
                "rating": 4.6,
                "opening_hours": "08:00-17:30",
                "ticket_price": "80元",
                "source": "local_mock",
            },
            {
                "id": 4,
                "name": "宋城",
                "location": location,
                "category": "主题公园",
                "description": "宋城是一座大型宋代文化主题公园，以《宋城千古情》演出闻名。",
                "rating": 4.5,
                "opening_hours": "10:00-21:00",
                "ticket_price": "310元",
                "source": "local_mock",
            }
        ]

        if keywords:
            filtered = []
            for attraction in mock_attractions:
                if any(kw in attraction["name"] or kw in attraction["description"] for kw in keywords):
                    filtered.append(attraction)
            return filtered if filtered else mock_attractions

        return mock_attractions
