"""
高德POI到RAG文档适配测试
"""

from rag.document_adapter import RAGDocumentAdapter


def test_rag_document_adapter_converts_amap_poi_to_attraction_document():
    """高德POI记录可以转换为景点RAG文档"""
    poi = {
        "id": "mock-西湖",
        "name": "西湖",
        "type": "风景名胜;旅游景点",
        "address": "杭州市西湖区",
        "location": "120.1551,30.2741",
        "pname": "浙江省",
        "cityname": "杭州市",
        "adname": "西湖区",
        "biz_ext": {"rating": "4.8"},
    }

    document = RAGDocumentAdapter().from_amap_poi(poi)

    assert document["title"] == "西湖"
    assert document["source"] == "api/amap/attraction/mock-西湖"
    assert document["type"] == "attraction"
    assert "## 西湖" in document["content"]
    assert "- 类型：风景名胜;旅游景点" in document["content"]
    assert "- 地址：杭州市西湖区" in document["content"]
    assert "- 坐标：120.1551,30.2741" in document["content"]
    assert "- 城市：杭州市" in document["content"]
    assert document["metadata"]["provider"] == "amap"
    assert document["metadata"]["source_type"] == "attraction"
    assert document["metadata"]["source"] == "external_api"
    assert document["metadata"]["city"] == "杭州市"
    assert document["metadata"]["district"] == "西湖区"


def test_rag_document_adapter_handles_missing_amap_fields():
    """高德POI缺少部分字段时仍可生成RAG文档"""
    poi = {
        "name": "未知景点",
        "cityname": "杭州市",
    }

    document = RAGDocumentAdapter().from_amap_poi(poi)

    assert document["title"] == "未知景点"
    assert document["source"] == "api/amap/attraction/未知景点"
    assert "## 未知景点" in document["content"]
    assert "- 城市：杭州市" in document["content"]
    assert document["metadata"]["provider"] == "amap"
    assert document["metadata"]["record_id"] == "未知景点"
