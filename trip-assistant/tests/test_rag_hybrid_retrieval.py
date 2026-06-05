"""RAG混合检索测试。"""

from rag.local_retriever import LocalMarkdownRetriever


class FakeEmbeddingManager:
    """用可控向量模拟语义相似度。"""

    def __init__(self):
        self.calls = 0

    def embed_batch(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            if "儿童" in text or "亲子" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def test_local_retriever_can_return_vector_only_match(tmp_path):
    """关键词缺失时，向量相似度可以召回语义相关chunk。"""
    documents_dir = tmp_path / "guides"
    documents_dir.mkdir()
    (documents_dir / "family.md").write_text(
        "# 家庭旅行\n\n亲子友好的博物馆和互动展览适合慢节奏安排。",
        encoding="utf-8",
    )
    (documents_dir / "nightlife.md").write_text(
        "# 夜间娱乐\n\n酒吧街和夜游项目适合成年人结伴体验。",
        encoding="utf-8",
    )
    fake_embedding = FakeEmbeddingManager()
    retriever = LocalMarkdownRetriever(
        embedding_manager=fake_embedding,
        vector_min_score=0.5,
    )
    documents = retriever.load_documents(documents_dir, "guide", tmp_path)

    results = retriever.search("儿童乐园", documents, terms=["儿童", "乐园"], top_k=1)

    assert results
    assert results[0]["title"] == "家庭旅行"
    assert results[0]["keyword_score"] == 0.0
    assert results[0]["vector_score"] == 1.0
    assert results[0]["retrieval_strategy"] == "vector"
    assert fake_embedding.calls == 1


def test_local_retriever_reports_hybrid_scores_for_keyword_hits(tmp_path):
    """关键词命中结果会保留关键词、向量和混合策略分数。"""
    documents_dir = tmp_path / "policies"
    documents_dir.mkdir()
    (documents_dir / "hotel.md").write_text(
        "# 酒店政策\n\n酒店入住前24小时通常可以免费取消。",
        encoding="utf-8",
    )
    retriever = LocalMarkdownRetriever(
        embedding_manager=FakeEmbeddingManager(),
        vector_min_score=0.0,
    )
    documents = retriever.load_documents(documents_dir, "policy", tmp_path)

    results = retriever.search("酒店能取消吗", documents, terms=["酒店", "取消"], top_k=1)

    assert results[0]["keyword_score"] == 1.0
    assert results[0]["vector_score"] >= 0.0
    assert results[0]["retrieval_strategy"] in {"hybrid", "keyword"}
    assert results[0]["score"] >= results[0]["keyword_score"] * retriever.keyword_weight
