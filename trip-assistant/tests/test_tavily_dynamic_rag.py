"""
Tavily 自增长 Agentic RAG 测试
"""
from pathlib import Path

import pytest

from app.config import settings
from rag.dynamic_guide_generator import GeneratedGuideResult, TavilyDynamicGuideGenerator
from rag.tavily_client import TavilyClient
from tools.guide import GuideTool


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_tavily_client_sends_domain_filter_and_sanitizes_key(monkeypatch):
    """Tavily请求包含域名过滤，失败时不泄露API Key"""
    monkeypatch.setattr(settings, "TAVILY_SEARCH_ENABLED", True)
    captured = {}

    def request(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        raise RuntimeError(f"bad key {json['api_key']}")

    client = TavilyClient(api_key="secret-key", request_func=request)

    result = client.search("烟台旅游攻略", include_domains=["xiaohongshu.com"], max_results=3)

    assert result["success"] is False
    assert "secret-key" not in result["error"]
    assert captured["json"]["include_domains"] == ["xiaohongshu.com"]
    assert captured["json"]["max_results"] == 3


def test_tavily_generator_persists_filtered_chunks(tmp_path, monkeypatch):
    """Tavily结果经过长度过滤后持久化为RAG文档"""
    monkeypatch.setattr(settings, "TAVILY_SEARCH_ENABLED", True)

    class FakeClient:
        def search(self, query, max_results=None):
            return {
                "success": True,
                "data": {
                    "results": [
                        {
                            "title": "烟台三天旅游攻略 小红书",
                            "url": "https://www.xiaohongshu.com/explore/1",
                            "content": "烟台三天旅游攻略，第一天烟台山和所城里，第二天养马岛看海，第三天蓬莱阁返程。海鲜和鲅鱼水饺适合安排在市区晚餐。",
                            "raw_content": "",
                            "score": 0.92,
                        },
                        {
                            "title": "短内容",
                            "url": "https://www.xiaohongshu.com/explore/2",
                            "content": "烟台好玩",
                            "score": 0.7,
                        },
                    ]
                },
            }

    generator = TavilyDynamicGuideGenerator(
        tavily_client=FakeClient(),
        output_dir=tmp_path,
        min_chunk_length=30,
        dedupe_threshold=0.99,
    )

    result = generator.ensure_destination_guide(destination="烟台", query="烟台旅游攻略")

    assert result.generated is True
    assert result.source_count == 2
    assert result.chunks_added == 1
    assert result.chunks_filtered_short == 1
    assert result.path and result.path.exists()
    content = result.path.read_text(encoding="utf-8")
    assert "烟台Tavily搜索攻略摘要" in content
    assert "Tavily相关性：0.92" in content
    assert "养马岛" in content


def test_tavily_generator_deduplicates_existing_chunks(tmp_path, monkeypatch):
    """入库前与已有目的地文档做相似度去重"""
    monkeypatch.setattr(settings, "TAVILY_SEARCH_ENABLED", True)

    class FakeClient:
        def search(self, query, max_results=None):
            return {
                "success": True,
                "data": {
                    "results": [
                        {
                            "title": "烟台三天旅游攻略",
                            "url": "https://example.com/a",
                            "content": "烟台三天旅游攻略，第一天烟台山和所城里，第二天养马岛看海，第三天蓬莱阁返程。",
                            "score": 0.9,
                        }
                    ]
                },
            }

    existing_documents = [
        {
            "title": "烟台已有攻略",
            "content": "# 烟台已有攻略\n\n烟台三天旅游攻略，第一天烟台山和所城里，第二天养马岛看海，第三天蓬莱阁返程。",
            "source": "rag/documents/guides/yantai.md",
            "type": "guide",
        }
    ]
    generator = TavilyDynamicGuideGenerator(
        tavily_client=FakeClient(),
        output_dir=tmp_path,
        min_chunk_length=20,
        dedupe_threshold=0.8,
    )

    result = generator.ensure_destination_guide(destination="烟台", query="烟台旅游攻略", existing_documents=existing_documents)

    assert result.generated is False
    assert result.chunks_deduplicated == 1
    assert result.error == "no_new_chunks"


def test_tavily_generator_falls_back_when_domain_result_is_site_shell(tmp_path, monkeypatch):
    """配置域名只返回站点壳时，不应把低质量内容写入RAG。"""
    monkeypatch.setattr(settings, "TAVILY_SEARCH_ENABLED", True)
    monkeypatch.setattr(settings, "TAVILY_INCLUDE_DOMAINS", "xiaohongshu.com")

    calls = []

    class FakeClient:
        def search(self, query, include_domains=None, max_results=None):
            calls.append(include_domains)
            if include_domains is None:
                return {
                    "success": True,
                    "data": {
                        "results": [
                            {
                                "title": "小红书 - Xiaohongshu",
                                "url": "https://www.xiaohongshu.com/explore",
                                "raw_content": "增值电信业务经营许可证 沪公网安备 违法不良信息举报 推荐 穿搭 美食 旅行",
                                "score": 0.9,
                            }
                        ]
                    },
                }
            return {
                "success": True,
                "data": {
                    "results": [
                        {
                            "title": "烟台三天旅游攻略",
                            "url": "https://example.com/yantai-guide",
                            "content": "烟台三天旅游攻略：第一天烟台山和所城里，第二天养马岛和海边景点，第三天市区美食和返程。",
                            "score": 0.86,
                        }
                    ]
                },
            }

    generator = TavilyDynamicGuideGenerator(
        tavily_client=FakeClient(),
        output_dir=tmp_path,
        min_chunk_length=20,
        dedupe_threshold=0.99,
    )

    result = generator.ensure_destination_guide(destination="烟台", query="烟台三天旅游攻略")

    assert result.generated is True
    assert calls == [None, []]
    assert result.path and result.path.exists()
    content = result.path.read_text(encoding="utf-8")
    assert "养马岛" in content
    assert "增值电信业务经营许可证" not in content


def test_tavily_generator_uses_city_slug_for_zhengzhou(tmp_path):
    """常见中文城市应生成可读文件名，便于用户确认对应RAG文档。"""
    generator = TavilyDynamicGuideGenerator(output_dir=tmp_path)

    path = generator._target_path("郑州")

    assert path.name.startswith("zhengzhou_")
    assert path.name.endswith("_tavily_generated.md")


@pytest.mark.asyncio
async def test_guide_tool_grows_knowledge_when_initial_rag_misses(tmp_path, monkeypatch):
    """初次RAG低置信未命中时搜索入库并二次检索"""
    monkeypatch.setattr(settings, "RAG_MIN_HIT_SCORE", 0.55)
    monkeypatch.setattr(settings, "RAG_MIN_CHUNK_LENGTH", 20)
    guides_dir = tmp_path / "guides"
    generated_dir = guides_dir / "generated"
    generated_dir.mkdir(parents=True)

    class FakeGenerator:
        def ensure_destination_guide(self, destination, query="", existing_documents=None):
            path = generated_dir / "yantai_tavily_generated.md"
            path.write_text(
                "# 烟台Tavily搜索攻略摘要\n\n"
                "## 攻略片段 1：烟台三天旅游攻略\n"
                "烟台三天旅游攻略，养马岛、烟台山、所城里适合三天路线安排。\n",
                encoding="utf-8",
            )
            return GeneratedGuideResult(generated=True, path=path, source_count=1, chunks_added=1)

    tool = GuideTool(documents_dir=guides_dir, guide_generator=FakeGenerator(), auto_generate=True)

    result = await tool.execute(query="烟台三天旅行攻略", destination="烟台")

    assert result["success"] is True
    assert result["metadata"]["rag_initial_hit"] is False
    assert result["metadata"]["web_search_attempted"] is True
    assert result["metadata"]["knowledge_chunks_added"] == 1
    assert result["metadata"]["rag_second_pass_hit"] is True
    assert "养马岛" in result["data"]["answer"]


@pytest.mark.asyncio
async def test_guide_tool_ignores_generated_metadata_sections(tmp_path, monkeypatch):
    """生成文档的元数据/来源段不应排在攻略片段前面。"""
    monkeypatch.setattr(settings, "RAG_MIN_HIT_SCORE", 0.55)
    monkeypatch.setattr(settings, "RAG_MIN_CHUNK_LENGTH", 20)
    guides_dir = tmp_path / "guides"
    generated_dir = guides_dir / "generated"
    generated_dir.mkdir(parents=True)
    (generated_dir / "yantai_tavily_generated.md").write_text(
        "# 烟台Tavily搜索攻略摘要\n\n"
        "## 元数据\n"
        "- 目的地：烟台\n"
        "- 查询词：烟台三天旅行攻略\n\n"
        "## 搜索来源\n"
        "1. [source](https://example.com)\n\n"
        "## 攻略片段 1：烟台三天旅行攻略\n"
        "烟台三天旅行攻略：第一天烟台山和所城里，第二天养马岛看海边景点，第三天市区美食和返程。\n",
        encoding="utf-8",
    )

    class FailingGenerator:
        def ensure_destination_guide(self, *args, **kwargs):
            raise AssertionError("existing guide should hit")

    tool = GuideTool(documents_dir=guides_dir, guide_generator=FailingGenerator(), auto_generate=True)

    result = await tool.execute(query="烟台三天旅行攻略", destination="烟台")

    assert result["metadata"]["rag_initial_hit"] is True
    assert result["metadata"]["web_search_attempted"] is False
    assert result["data"]["sources"][0]["section"].startswith("攻略片段")
    assert "养马岛" in result["data"]["answer"]
    assert "元数据" not in result["data"]["answer"]


@pytest.mark.asyncio
async def test_guide_tool_creates_canonical_file_when_old_generated_slug_hits(tmp_path, monkeypatch):
    """旧版 destination_ 文件命中时，也要补生成规范城市slug文件。"""
    monkeypatch.setattr(settings, "RAG_MIN_HIT_SCORE", 0.55)
    monkeypatch.setattr(settings, "RAG_MIN_CHUNK_LENGTH", 20)
    guides_dir = tmp_path / "guides"
    generated_dir = guides_dir / "generated"
    generated_dir.mkdir(parents=True)
    (generated_dir / "destination_old_tavily_generated.md").write_text(
        "# 郑州Tavily搜索攻略摘要\n\n"
        "## 攻略片段 1：郑州三天旅行攻略\n"
        "郑州三天旅行攻略：第一天二七纪念塔，第二天河南博物院，第三天少林寺方向一日游。\n",
        encoding="utf-8",
    )

    class CanonicalGenerator:
        def __init__(self):
            self.called = False

        def target_path_for(self, destination):
            return generated_dir / "zhengzhou_canonical_tavily_generated.md"

        def ensure_destination_guide(self, destination, query="", existing_documents=None):
            self.called = True
            assert existing_documents == []
            path = self.target_path_for(destination)
            path.write_text(
                "# 郑州Tavily搜索攻略摘要\n\n"
                "## 攻略片段 1：郑州三天旅行攻略\n"
                "郑州三天旅行攻略：河南博物院、二七纪念塔和嵩山少林寺适合组合安排。\n",
                encoding="utf-8",
            )
            return GeneratedGuideResult(generated=True, path=path, source_count=1, chunks_added=1)

    generator = CanonicalGenerator()
    tool = GuideTool(documents_dir=guides_dir, guide_generator=generator, auto_generate=True)

    result = await tool.execute(query="郑州三天旅行攻略", destination="郑州")

    assert generator.called is True
    assert result["metadata"]["rag_initial_hit"] is True
    assert result["metadata"]["knowledge_chunks_added"] == 1
    assert (generated_dir / "zhengzhou_canonical_tavily_generated.md").exists()


@pytest.mark.asyncio
async def test_guide_tool_does_not_search_when_initial_rag_hits(tmp_path, monkeypatch):
    """已有高置信目的地攻略时不触发Tavily搜索"""
    monkeypatch.setattr(settings, "RAG_MIN_HIT_SCORE", 0.2)
    monkeypatch.setattr(settings, "RAG_MIN_CHUNK_LENGTH", 20)
    guides_dir = tmp_path / "guides"
    guides_dir.mkdir()
    (guides_dir / "yantai.md").write_text(
        "# 烟台旅游攻略\n\n## 三天路线\n烟台三天旅行可以安排烟台山、所城里、养马岛和海鲜美食。",
        encoding="utf-8",
    )

    class FailingGenerator:
        def ensure_destination_guide(self, *args, **kwargs):
            raise AssertionError("should not search")

    tool = GuideTool(documents_dir=guides_dir, guide_generator=FailingGenerator(), auto_generate=True)

    result = await tool.execute(query="烟台三天旅行攻略", destination="烟台")

    assert result["metadata"]["rag_initial_hit"] is True
    assert result["metadata"]["web_search_attempted"] is False
    assert "养马岛" in result["data"]["answer"]


@pytest.mark.asyncio
async def test_guide_tool_degrades_when_tavily_unavailable(tmp_path):
    """Tavily失败时明确降级，不编造攻略"""
    guides_dir = tmp_path / "guides"
    guides_dir.mkdir()

    class EmptyGenerator:
        def ensure_destination_guide(self, destination, query="", existing_documents=None):
            return GeneratedGuideResult(generated=False, error="missing_api_key")

    tool = GuideTool(documents_dir=guides_dir, guide_generator=EmptyGenerator(), auto_generate=True)

    result = await tool.execute(query="烟台三天旅行攻略", destination="烟台")

    assert result["success"] is True
    assert result["data"]["sources"] == []
    assert "暂未检索到烟台的本地攻略文档" in result["data"]["answer"]
    assert result["metadata"]["web_search_attempted"] is True
    assert result["metadata"]["knowledge_growth_error"] == "missing_api_key"
