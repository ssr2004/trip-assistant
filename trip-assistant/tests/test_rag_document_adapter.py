"""
RAG文档适配器测试
"""
from rag.document_adapter import RAGDocumentAdapter


def test_api_record_to_rag_document():
    """外部API记录可以转换为RAG文档结构"""
    adapter = RAGDocumentAdapter()

    document = adapter.from_api_record(
        record={
            "id": "poi-001",
            "name": "西湖",
            "description": "杭州代表性自然风光景点",
            "address": "杭州市西湖区",
            "category": "自然风光",
        },
        source_type="amap_poi",
        provider="amap",
    )

    assert document["title"] == "西湖"
    assert document["source"] == "api/amap/amap_poi/poi-001"
    assert document["type"] == "amap_poi"
    assert "## 西湖" in document["content"]
    assert "杭州代表性自然风光景点" in document["content"]
    assert document["metadata"]["provider"] == "amap"
    assert document["metadata"]["source_type"] == "amap_poi"
    assert document["metadata"]["record_id"] == "poi-001"
    assert document["metadata"]["source"] == "external_api"


def test_api_record_adapter_uses_fallback_title_and_record_id():
    """缺少标题和ID时适配器仍能生成稳定文档"""
    adapter = RAGDocumentAdapter()

    document = adapter.from_api_record(
        record={"description": "开放政策数据"},
        source_type="open_policy_source",
        provider="open_data",
    )

    assert document["title"] == "open_data_open_policy_source"
    assert document["source"] == "api/open_data/open_policy_source/open_data_open_policy_source"
    assert "开放政策数据" in document["content"]
