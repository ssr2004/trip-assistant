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
            "rating": "评分",
            "price": "价格",
        }
        return labels.get(field, field)
