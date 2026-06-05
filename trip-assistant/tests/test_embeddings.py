"""Embedding管理器测试。"""

from rag.embeddings import EmbeddingManager


class FakeEmbeddingItem:
    """模拟单条embedding响应。"""

    def __init__(self, embedding):
        self.embedding = embedding


class FakeEmbeddingResponse:
    """模拟OpenAI-compatible embedding响应。"""

    def __init__(self, embeddings):
        self.data = [FakeEmbeddingItem(embedding) for embedding in embeddings]


class FakeEmbeddingsEndpoint:
    """模拟embeddings接口。"""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        inputs = kwargs["input"] if isinstance(kwargs["input"], list) else [kwargs["input"]]
        embeddings = [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(inputs)]
        return FakeEmbeddingResponse(embeddings)


class FakeOpenAIClient:
    """模拟OpenAI-compatible客户端。"""

    def __init__(self):
        self.embeddings = FakeEmbeddingsEndpoint()


class FakeFailingEmbeddingsEndpoint:
    """模拟provider失败。"""

    def create(self, **kwargs):
        raise RuntimeError("provider failed with key sk-test-secret")


class FakeFailingOpenAIClient:
    """模拟失败OpenAI-compatible客户端。"""

    embeddings = FakeFailingEmbeddingsEndpoint()


def test_embedding_fallback_is_deterministic_without_api_key():
    """无Key时使用确定性本地向量，避免随机检索结果。"""
    manager = EmbeddingManager(api_key="", fallback_dimension=32)

    first = manager.embed("杭州西湖亲子旅行")
    second = manager.embed("杭州西湖亲子旅行")
    other = manager.embed("成都火锅美食")

    assert first == second
    assert first != other
    assert len(first) == 32
    assert manager.last_backend == "deterministic_fallback"


def test_embedding_openai_compatible_batch_and_cache():
    """有Key时通过OpenAI-compatible接口批量请求，并复用缓存。"""
    fake_client = FakeOpenAIClient()
    manager = EmbeddingManager(
        api_key="sk-test-secret",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="text-embedding-v4",
        openai_client=fake_client,
    )

    first = manager.embed_batch(["杭州攻略", "酒店取消"])
    second = manager.embed_batch(["杭州攻略", "酒店取消"])

    assert first == second
    assert fake_client.embeddings.calls == [
        {
            "model": "text-embedding-v4",
            "input": ["杭州攻略", "酒店取消"],
        }
    ]
    assert manager.last_backend == "openai_compatible"


def test_embedding_provider_error_is_sanitized_and_falls_back():
    """Provider失败时错误信息脱敏，并回退到确定性向量。"""
    manager = EmbeddingManager(
        api_key="sk-test-secret",
        openai_client=FakeFailingOpenAIClient(),
        fallback_dimension=16,
    )

    embedding = manager.embed("三亚亲子游")

    assert len(embedding) == 16
    assert manager.last_backend == "deterministic_fallback"
    assert "sk-test-secret" not in manager.last_error
    assert "***" in manager.last_error
