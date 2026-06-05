"""
高德地图API客户端
基于外部API客户端基础设施封装POI检索能力
"""
from typing import Any, Callable, Dict, List, Optional

from app.config import settings
from core.cache import ExternalAPICache, create_external_api_cache
from core.external_api import ExternalAPIClient


class AMapPOIClient:
    """高德POI检索客户端"""

    SEARCH_URL = "https://restapi.amap.com/v3/place/text"
    SCENIC_TYPES = "110000"

    def __init__(
        self,
        api_key: Optional[str] = None,
        request_func: Optional[Callable[..., Any]] = None,
        mock_enabled: Optional[bool] = None,
        cache: Optional[ExternalAPICache] = None,
    ):
        """初始化高德POI客户端"""
        resolved_key = settings.AMAP_API_KEY if api_key is None else api_key
        self.client = ExternalAPIClient(
            name="amap",
            api_key=resolved_key,
            timeout=settings.EXTERNAL_API_TIMEOUT,
            retry_times=settings.EXTERNAL_API_RETRY_TIMES,
            mock_enabled=mock_enabled,
            request_func=request_func,
        )
        self.cache = cache or create_external_api_cache()

    def available(self) -> bool:
        """判断是否具备真实高德API调用条件"""
        return self.client.available()

    async def search_pois(
        self,
        city: str,
        keywords: str = "景点",
        offset: int = 10,
        page: int = 1,
    ) -> Dict[str, Any]:
        """检索城市景点POI，未配置Key时返回mock fallback"""
        params = {
            "key": self.client.api_key,
            "keywords": keywords or "景点",
            "city": city,
            "types": self.SCENIC_TYPES,
            "offset": offset,
            "page": page,
            "extensions": "base",
        }
        cache_params = self._cache_params(params)
        cached_result, cache_metadata = await self.cache.get("amap", "poi", cache_params)
        if cached_result:
            return self._with_cache_metadata(cached_result, cache_metadata)

        result = await self.client.request_json(
            method="GET",
            url=self.SEARCH_URL,
            params=params,
            mock_data=self._mock_poi_response(city),
        )
        result = self._normalize_response(result, city)
        if result.get("success") and not (result.get("metadata", {}) or {}).get("mock"):
            write_metadata = await self.cache.set(
                provider="amap",
                resource="poi",
                params=cache_params,
                value=self._without_cache_metadata(result),
            )
            cache_metadata.update(write_metadata)
        else:
            cache_metadata["cache_write"] = False
        cache_metadata["cache_hit"] = False
        return self._with_cache_metadata(result, cache_metadata)

    def _normalize_response(self, result: Dict[str, Any], city: str) -> Dict[str, Any]:
        """处理高德业务状态码，避免HTTP成功但业务失败被误判为真实数据。"""
        if not result.get("success"):
            return result
        metadata = result.get("metadata", {}) or {}
        if metadata.get("mock"):
            return result
        data = result.get("data", {}) or {}
        if str(data.get("status")) == "1":
            return result

        reason = data.get("info") or "amap_poi_business_error"
        if self.client.mock_enabled:
            return self.client.mock_response(
                mock_data=self._mock_poi_response(city),
                reason=str(reason),
                metadata={"api_status": "degraded", "error_type": "business_error"},
            )
        return self.client.error_response(
            str(reason),
            mock=False,
            metadata={
                "api_status": "failed",
                "execution_mode": "real_api_failed",
                "error_type": "business_error",
            },
        )

    def _cache_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """构建不含敏感Key的缓存参数"""
        return {key: value for key, value in params.items() if key != "key"}

    def _with_cache_metadata(self, result: Dict[str, Any], cache_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """将缓存元数据合并到外部API响应中"""
        merged = dict(result)
        metadata = dict(result.get("metadata", {}) or {})
        metadata.update(cache_metadata)
        merged["metadata"] = metadata
        return merged

    def _without_cache_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """构建可缓存响应，避免缓存命中信息被持久化"""
        cached = dict(result)
        metadata = {
            key: value
            for key, value in (result.get("metadata", {}) or {}).items()
            if not key.startswith("cache_")
        }
        cached["metadata"] = metadata
        return cached

    def _mock_poi_response(self, city: str) -> Dict[str, Any]:
        """构建高德POI mock响应"""
        return {
            "status": "1",
            "info": "MOCK",
            "count": str(len(self._mock_pois(city))),
            "pois": self._mock_pois(city),
        }

    def _mock_pois(self, city: str) -> List[Dict[str, Any]]:
        """按城市返回景点POI mock数据"""
        city_pois = {
            "杭州": [
                self._poi("西湖", "风景名胜", "杭州市西湖区", "120.1551,30.2741", "浙江省", "杭州市", "西湖区"),
                self._poi("灵隐寺", "风景名胜;宗教场所", "杭州市西湖区灵隐路", "120.1022,30.2400", "浙江省", "杭州市", "西湖区"),
                self._poi("西溪国家湿地公园", "风景名胜;公园广场", "杭州市西湖区天目山路", "120.0648,30.2745", "浙江省", "杭州市", "西湖区"),
                self._poi("宋城", "风景名胜;主题公园", "杭州市西湖区之江路", "120.0962,30.1607", "浙江省", "杭州市", "西湖区"),
            ],
            "成都": [
                self._poi("成都大熊猫繁育研究基地", "风景名胜", "成都市成华区熊猫大道", "104.1455,30.7396", "四川省", "成都市", "成华区"),
                self._poi("宽窄巷子", "风景名胜;历史文化", "成都市青羊区金河路", "104.0563,30.6698", "四川省", "成都市", "青羊区"),
                self._poi("武侯祠", "风景名胜;历史文化", "成都市武侯区武侯祠大街", "104.0472,30.6470", "四川省", "成都市", "武侯区"),
                self._poi("春熙路", "购物服务;城市地标", "成都市锦江区春熙路", "104.0808,30.6551", "四川省", "成都市", "锦江区"),
            ],
            "厦门": [
                self._poi("鼓浪屿", "风景名胜", "厦门市思明区鼓浪屿", "118.0679,24.4446", "福建省", "厦门市", "思明区"),
                self._poi("环岛路", "风景名胜;道路", "厦门市思明区环岛路", "118.1476,24.4290", "福建省", "厦门市", "思明区"),
                self._poi("南普陀寺", "风景名胜;宗教场所", "厦门市思明区思明南路", "118.1011,24.4437", "福建省", "厦门市", "思明区"),
                self._poi("曾厝垵", "风景名胜;特色街区", "厦门市思明区曾厝垵", "118.1319,24.4328", "福建省", "厦门市", "思明区"),
            ],
            "三亚": [
                self._poi("亚龙湾", "风景名胜;海滨", "三亚市吉阳区亚龙湾", "109.6425,18.2293", "海南省", "三亚市", "吉阳区"),
                self._poi("蜈支洲岛", "风景名胜;海岛", "三亚市海棠区蜈支洲岛", "109.7577,18.3151", "海南省", "三亚市", "海棠区"),
                self._poi("天涯海角", "风景名胜", "三亚市天涯区天涯镇", "109.3498,18.2973", "海南省", "三亚市", "天涯区"),
                self._poi("南山文化旅游区", "风景名胜;人文景观", "三亚市崖州区南山", "109.2082,18.3035", "海南省", "三亚市", "崖州区"),
            ],
        }
        return city_pois.get(city, city_pois["杭州"])

    def _poi(
        self,
        name: str,
        poi_type: str,
        address: str,
        location: str,
        province: str,
        city: str,
        district: str,
    ) -> Dict[str, Any]:
        """构建单条POI mock数据"""
        return {
            "id": f"mock-{name}",
            "name": name,
            "type": poi_type,
            "address": address,
            "location": location,
            "pname": province,
            "cityname": city,
            "adname": district,
            "biz_ext": {"rating": "4.8"},
        }
