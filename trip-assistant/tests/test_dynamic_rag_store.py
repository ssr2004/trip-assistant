"""
动态RAG文档存储测试
"""

from rag.document_adapter import RAGDocumentAdapter
from rag.dynamic_store import DynamicRAGStore


def _amap_poi(name="西湖", poi_id="mock-西湖"):
    return {
        "id": poi_id,
        "name": name,
        "type": "风景名胜;旅游景点",
        "address": "杭州市西湖区",
        "location": "120.1551,30.2741",
        "pname": "浙江省",
        "cityname": "杭州市",
        "adname": "西湖区",
        "biz_ext": {"rating": "4.8"},
    }


def test_dynamic_rag_store_adds_and_lists_documents():
    """动态RAG Store可以添加并列出外部文档"""
    document = RAGDocumentAdapter().from_amap_poi(_amap_poi())
    store = DynamicRAGStore()

    store.add_documents([document])

    documents = store.list_documents()
    assert len(documents) == 1
    assert documents[0]["title"] == "西湖"
    assert documents[0]["source"] == "api/amap/attraction/mock-西湖"
    assert documents[0]["type"] == "attraction"


def test_dynamic_rag_store_deduplicates_documents_by_source():
    """动态RAG Store按文档标识去重"""
    adapter = RAGDocumentAdapter()
    document = adapter.from_amap_poi(_amap_poi())
    duplicate = adapter.from_amap_poi(_amap_poi())
    store = DynamicRAGStore()

    store.add_documents([document, duplicate])

    assert len(store.list_documents()) == 1


def test_dynamic_rag_store_search_hits_external_poi_document():
    """动态RAG Store可以检索外部POI文档"""
    adapter = RAGDocumentAdapter()
    store = DynamicRAGStore()
    store.add_documents([
        adapter.from_amap_poi(_amap_poi("西湖", "mock-西湖")),
        adapter.from_amap_poi(_amap_poi("灵隐寺", "mock-灵隐寺")),
    ])

    results = store.search("西湖", top_k=3)

    assert results
    assert results[0]["title"] == "西湖"
    assert results[0]["source"] == "api/amap/attraction/mock-西湖"
    assert results[0]["type"] == "attraction"
    assert results[0]["score"] > 0
    assert "西湖" in results[0]["excerpt"]
    assert results[0]["chunk_id"]
    assert results[0]["document_id"]


def test_dynamic_rag_store_ignores_invalid_documents():
    """动态RAG Store忽略缺少关键字段的文档"""
    store = DynamicRAGStore()

    store.add_documents([{}, {"title": "缺少内容"}, None])

    assert store.list_documents() == []
    assert store.search("西湖") == []
