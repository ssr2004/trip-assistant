"""
RAG文档适配器
将外部API记录转换为本地RAG可检索的文档结构
"""
from typing import Any, Dict, Optional


class RAGDocumentAdapter:
    """RAG文档适配器"""

    def from_api_record(
        self,
        record: Dict[str, Any],
        source_type: str,
        provider: str,
        title_field: str = "name",
        content_fields: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """将外部API记录转换为本地文档结构"""
        content_fields = content_fields or ["name", "description", "address", "category"]
        title = str(record.get(title_field) or record.get("title") or f"{provider}_{source_type}")
        content = self._build_markdown_content(title, record, content_fields)
        record_id = record.get("id") or record.get("uid") or record.get("code") or title

        return {
            "title": title,
            "content": content,
            "source": f"api/{provider}/{source_type}/{record_id}",
            "type": source_type,
            "metadata": {
                "provider": provider,
                "source_type": source_type,
                "record_id": str(record_id),
                "source": "external_api",
            },
        }

    def from_amap_poi(self, poi: Dict[str, Any]) -> Dict[str, Any]:
        """将高德POI记录转换为景点RAG文档"""
        biz_ext = poi.get("biz_ext") if isinstance(poi.get("biz_ext"), dict) else {}
        record = {
            "id": poi.get("id") or poi.get("uid") or poi.get("name"),
            "name": poi.get("name"),
            "type": poi.get("type"),
            "address": poi.get("address") or poi.get("adname"),
            "location": poi.get("location"),
            "province": poi.get("pname"),
            "city": poi.get("cityname"),
            "district": poi.get("adname"),
            "rating": biz_ext.get("rating"),
        }
        document = self.from_api_record(
            record=record,
            source_type="attraction",
            provider="amap",
            title_field="name",
            content_fields=["name", "type", "address", "location", "province", "city", "district", "rating"],
        )
        document["metadata"].update({
            "poi_type": str(poi.get("type") or ""),
            "city": str(poi.get("cityname") or ""),
            "district": str(poi.get("adname") or ""),
        })
        return document

    def _build_markdown_content(
        self,
        title: str,
        record: Dict[str, Any],
        content_fields: list[str],
    ) -> str:
        """构建Markdown文档内容"""
        lines = [f"## {title}"]
        for field in content_fields:
            value = record.get(field)
            if value in (None, "", []):
                continue
            lines.append(f"- {self._field_label(field)}：{value}")
        return "\n".join(lines)

    def _field_label(self, field: str) -> str:
        """字段展示名"""
        labels = {
            "name": "名称",
            "title": "标题",
            "description": "描述",
            "address": "地址",
            "category": "分类",
            "type": "类型",
            "location": "坐标",
            "province": "省份",
            "city": "城市",
            "district": "行政区",
            "rating": "评分",
            "price": "价格",
        }
        return labels.get(field, field)
