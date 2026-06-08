"""
RAG结构化检索测试
"""
import pytest

from core.response_builder import ResponseBuilder
from rag.local_retriever import LocalMarkdownRetriever
from tools.guide import GuideTool
from tools.policy import PolicyTool


def test_local_markdown_retriever_loads_title_and_searches(tmp_path):
    """本地Markdown检索器可以加载标题并返回结构化片段"""
    documents_dir = tmp_path / "policies"
    documents_dir.mkdir()
    document_path = documents_dir / "ticket.md"
    document_path.write_text(
        "# 测试退票政策\n\n退票需要根据起飞时间收取手续费。\n改签可以按同舱规则办理。",
        encoding="utf-8",
    )

    retriever = LocalMarkdownRetriever()
    documents = retriever.load_documents(documents_dir, "policy", tmp_path)
    results = retriever.search("机票退票", documents, ["退票", "手续费"], top_k=1)

    assert documents[0]["title"] == "测试退票政策"
    assert documents[0]["source"] == "policies/ticket.md"
    assert results[0]["title"] == "测试退票政策"
    assert results[0]["type"] == "policy"
    assert results[0]["score"] > 0
    assert results[0]["matched_terms"] == ["退票", "手续费"]
    assert "退票" in results[0]["excerpt"]
    assert results[0]["content"] == results[0]["excerpt"]


@pytest.mark.asyncio
async def test_policy_tool_returns_structured_sources():
    """政策工具返回标题、命中词、分数和引用片段"""
    tool = PolicyTool()

    result = await tool.execute(query="机票能退吗")

    assert result["success"] is True
    sources = result["data"]["sources"]
    assert sources
    first_source = sources[0]
    assert first_source["title"] == "机票退改签政策"
    assert first_source["type"] == "policy"
    assert first_source["score"] > 0
    assert first_source["matched_terms"]
    assert first_source["excerpt"]
    assert first_source["content"] == first_source["excerpt"]
    assert "《机票退改签政策》" in result["data"]["answer"]


@pytest.mark.asyncio
async def test_guide_tool_returns_structured_sources(tmp_path):
    """攻略工具返回结构化攻略来源"""
    guides_dir = tmp_path / "guides"
    guides_dir.mkdir()
    (guides_dir / "hangzhou.md").write_text(
        "# 杭州旅游攻略\n\n## 三天路线\n杭州三天旅行可以安排西湖、灵隐寺和宋城。",
        encoding="utf-8",
    )
    tool = GuideTool(documents_dir=guides_dir, auto_generate=False)

    result = await tool.execute(query="杭州三天旅行攻略", destination="杭州")

    assert result["success"] is True
    assert result["data"]["destination"] == "杭州"
    assert "杭州" in result["data"]["answer"]
    sources = result["data"]["sources"]
    assert sources
    first_source = sources[0]
    assert first_source["title"] == "杭州旅游攻略"
    assert first_source["type"] == "guide"
    assert first_source["score"] > 0
    assert "杭州" in first_source["matched_terms"]
    assert first_source["excerpt"]


@pytest.mark.asyncio
async def test_guide_tool_does_not_cross_destination_documents(tmp_path):
    """目的地没有本地攻略时不能引用其他城市攻略"""
    guides_dir = tmp_path / "guides"
    guides_dir.mkdir()
    (guides_dir / "hangzhou.md").write_text(
        "# 杭州旅游攻略\n\n杭州三天旅行可以安排西湖、灵隐寺和宋城。",
        encoding="utf-8",
    )
    tool = GuideTool(documents_dir=guides_dir, auto_generate=False)

    result = await tool.execute(query="烟台三天旅行攻略", destination="烟台")

    assert result["success"] is True
    assert result["data"]["sources"] == []
    assert "暂未检索到烟台的本地攻略文档" in result["data"]["answer"]
    assert "杭州旅游攻略" not in result["data"]["answer"]
    assert result["metadata"]["destination_filter_applied"] is True
    assert result["metadata"]["destination_document_count"] == 0


def test_policy_response_displays_source_title():
    """政策回复展示结构化来源标题"""
    builder = ResponseBuilder()

    response = builder.build(
        intent={"intent": "policy_query"},
        task_results=[
            {
                "task": {"task_type": "tool_call", "tool": "retrieve_policy", "name": "检索政策文档"},
                "success": True,
                "result": {
                    "success": True,
                    "data": {
                        "query": "机票能退吗",
                        "answer": "根据本地政策文档《机票退改签政策》，机票可按规则退改。",
                        "sources": [
                            {
                                "title": "机票退改签政策",
                                "source": "rag/documents/policies/flight_policy.md",
                                "type": "policy",
                                "score": 1.0,
                                "matched_terms": ["机票"],
                                "excerpt": "## 机票退改签政策",
                            }
                        ],
                    },
                    "error": None,
                    "metadata": {"tool": "retrieve_policy"},
                },
                "error": None,
            }
        ],
    )

    assert "资料来源" in response
    assert "机票退改签政策：rag/documents/policies/flight_policy.md" in response


def test_guide_response_displays_source_title():
    """完整旅行规划只展示攻略规划依据，不输出RAG原文"""
    builder = ResponseBuilder()
    raw_guide_text = "这是很长的攻略原文，不应该在完整旅行规划里直接展示。导游小西免费咨询，图片链接 abee_b.jpg。"

    response = builder.build(
        intent={"intent": "travel_plan", "entities": {"origin": "郑州", "destination": "杭州", "duration": 3}},
        task_results=[
            {
                "task": {"task_type": "tool_call", "tool": "generate_itinerary", "name": "生成旅行行程"},
                "success": True,
                "result": {
                    "success": True,
                    "data": {
                        "origin": "郑州",
                        "destination": "杭州",
                        "duration": 3,
                        "itinerary": [],
                    },
                    "error": None,
                    "metadata": {"tool": "generate_itinerary"},
                },
                "error": None,
            },
            {
                "task": {"task_type": "tool_call", "tool": "retrieve_guide", "name": "检索旅行攻略"},
                "success": True,
                "result": {
                    "success": True,
                    "data": {
                        "query": "杭州三天旅行攻略",
                        "destination": "杭州",
                        "answer": raw_guide_text,
                        "planning_insights": {
                            "highlights": ["西湖", "灵隐寺"],
                            "decision_basis": ["优先围绕西湖和灵隐寺组织游览顺序"],
                            "route_hints": ["西湖和灵隐寺适合分成两天安排。"],
                            "source_briefs": [
                                {
                                    "title": "杭州旅游攻略",
                                    "source": "rag/documents/guides/hangzhou_guide.md",
                                }
                            ],
                        },
                        "sources": [
                            {
                                "title": "杭州旅游攻略",
                                "source": "rag/documents/guides/hangzhou_guide.md",
                                "type": "guide",
                                "score": 1.0,
                                "matched_terms": ["杭州"],
                                "excerpt": "## 杭州旅游攻略",
                            }
                        ],
                    },
                    "error": None,
                    "metadata": {"tool": "retrieve_guide"},
                },
                "error": None,
            },
        ],
    )

    assert "攻略与注意事项" not in response
    assert "规划采用的攻略要点" not in response
    assert "规划决策" not in response
    assert "杭州旅游攻略：rag/documents/guides/hangzhou_guide.md" not in response
    assert raw_guide_text not in response
    assert "导游小西免费咨询" not in response
