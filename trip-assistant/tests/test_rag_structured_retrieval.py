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
async def test_guide_tool_returns_structured_sources():
    """攻略工具返回结构化攻略来源"""
    tool = GuideTool()

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
    """完整旅行规划中的攻略来源展示标题"""
    builder = ResponseBuilder()

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
                        "answer": "根据本地攻略文档《杭州旅游攻略》，为您检索到杭州相关建议。",
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

    assert "七、攻略与注意事项" in response
    assert "杭州旅游攻略：rag/documents/guides/hangzhou_guide.md" in response
