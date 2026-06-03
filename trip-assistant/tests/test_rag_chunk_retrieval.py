"""
RAG文档分块检索测试
"""
import pytest

from core.intent import IntentParser
from core.planner import TaskPlanner
from core.response_builder import ResponseBuilder
from rag.local_retriever import LocalMarkdownRetriever
from tools.guide import GuideTool
from tools.policy import PolicyTool


def test_local_markdown_retriever_splits_document_by_sections(tmp_path):
    """Markdown文档可以按章节切分为结构化chunk"""
    documents_dir = tmp_path / "policies"
    documents_dir.mkdir()
    document_path = documents_dir / "flight_policy.md"
    document_path.write_text(
        "## 机票退改签政策\n\n"
        "### 退票规定\n\n"
        "起飞前可以申请退票。\n"
        "退票会根据时间收取手续费。\n\n"
        "### 改签规定\n\n"
        "同舱改签需要遵守航司规则。",
        encoding="utf-8",
    )

    retriever = LocalMarkdownRetriever()
    documents = retriever.load_documents(documents_dir, "policy", tmp_path)
    chunks = retriever.split_documents(documents, max_lines=4)

    assert len(chunks) >= 2
    first_chunk = chunks[0]
    assert first_chunk["chunk_id"] == "policy-flight_policy-0"
    assert first_chunk["document_id"] == "flight_policy"
    assert first_chunk["title"] == "机票退改签政策"
    assert first_chunk["section"] == "退票规定"
    assert first_chunk["chunk_index"] == 0
    assert first_chunk["source"] == "policies/flight_policy.md"


def test_chunk_search_returns_section_metadata(tmp_path):
    """chunk级检索结果包含章节和chunk元数据"""
    documents_dir = tmp_path / "policies"
    documents_dir.mkdir()
    (documents_dir / "hotel_policy.md").write_text(
        "## 酒店取消与入住政策\n\n"
        "### 免费取消\n\n"
        "酒店入住前24小时通常可以免费取消。\n"
        "### 入住与退房\n\n"
        "入住时间通常为14点以后。",
        encoding="utf-8",
    )

    retriever = LocalMarkdownRetriever()
    documents = retriever.load_documents(documents_dir, "policy", tmp_path)
    results = retriever.search("酒店能取消吗", documents, ["酒店", "取消"], top_k=1)

    assert results
    assert results[0]["title"] == "酒店取消与入住政策"
    assert results[0]["section"] == "免费取消"
    assert results[0]["chunk_id"] == "policy-hotel_policy-0"
    assert results[0]["document_id"] == "hotel_policy"
    assert results[0]["chunk_index"] == 0
    assert "取消" in results[0]["matched_terms"]


@pytest.mark.asyncio
async def test_policy_tool_hits_refund_chunk():
    """机票退票问题可以命中退票规定chunk"""
    tool = PolicyTool()

    result = await tool.execute(query="机票能退吗")

    assert result["success"] is True
    first_source = result["data"]["sources"][0]
    assert first_source["title"] == "机票退改签政策"
    assert first_source["section"] == "退票规定"
    assert first_source["chunk_id"]
    assert "退" in first_source["matched_terms"] or "退票" in first_source["matched_terms"]
    assert "退票规定" in result["data"]["answer"]


@pytest.mark.asyncio
async def test_policy_tool_hits_hotel_policy_document():
    """酒店取消问题可以命中酒店政策文档"""
    tool = PolicyTool()

    result = await tool.execute(query="酒店能取消吗")

    assert result["success"] is True
    first_source = result["data"]["sources"][0]
    assert first_source["title"] == "酒店取消与入住政策"
    assert first_source["type"] == "policy"
    assert "取消" in first_source["matched_terms"]
    assert "酒店取消与入住政策" in result["data"]["answer"]


@pytest.mark.asyncio
async def test_policy_tool_hits_attraction_ticket_policy_document():
    """景点门票退改问题可以命中门票政策文档"""
    tool = PolicyTool()

    result = await tool.execute(query="景点门票可以退吗")

    assert result["success"] is True
    first_source = result["data"]["sources"][0]
    assert first_source["title"] == "景点门票退改政策"
    assert first_source["type"] == "policy"
    assert "门票" in first_source["matched_terms"]
    assert "景点门票退改政策" in result["data"]["answer"]


@pytest.mark.asyncio
async def test_guide_tool_hits_chengdu_food_guide():
    """成都美食问题可以命中成都攻略"""
    tool = GuideTool()

    result = await tool.execute(query="成都有什么好吃的", destination="成都")

    assert result["success"] is True
    first_source = result["data"]["sources"][0]
    assert first_source["title"] == "成都旅游攻略"
    assert first_source["type"] == "guide"
    assert "成都" in first_source["matched_terms"]
    assert "成都旅游攻略" in result["data"]["answer"]


@pytest.mark.asyncio
async def test_guide_tool_hits_sanya_three_day_guide():
    """三亚三天行程问题可以命中三亚攻略"""
    tool = GuideTool()

    result = await tool.execute(query="三亚三天怎么安排", destination="三亚")

    assert result["success"] is True
    first_source = result["data"]["sources"][0]
    assert first_source["title"] == "三亚旅游攻略"
    assert first_source["type"] == "guide"
    assert "三亚" in first_source["matched_terms"]
    assert "三亚旅游攻略" in result["data"]["answer"]


def test_intent_and_planner_support_guide_query():
    """攻略类自然语言问题会规划到攻略检索工具"""
    parser = IntentParser()
    planner = TaskPlanner()

    intent = parser.parse("成都有什么好吃的")
    tasks = planner.plan(intent, {"query": "成都有什么好吃的"})

    assert intent["intent"] == "guide_query"
    assert intent["entities"]["destination"] == "成都"
    assert tasks[0]["tool"] == "retrieve_guide"
    assert tasks[0]["params"]["destination"] == "成都"


def test_intent_policy_query_handles_hotel_cancellation():
    """酒店取消类问题优先识别为政策查询"""
    parser = IntentParser()

    intent = parser.parse("酒店能取消吗")

    assert intent["intent"] == "policy_query"


def test_response_builder_displays_title_and_section_reference():
    """响应构建器展示文档标题和章节标题"""
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
                        "query": "酒店能取消吗",
                        "answer": "根据本地政策文档《酒店取消与入住政策》的“免费取消”部分，可参考以下内容。",
                        "sources": [
                            {
                                "title": "酒店取消与入住政策",
                                "section": "免费取消",
                                "source": "rag/documents/policies/hotel_policy.md",
                                "type": "policy",
                                "score": 1.0,
                                "matched_terms": ["酒店", "取消"],
                                "excerpt": "酒店入住前24小时通常可以免费取消。",
                                "chunk_id": "policy-hotel_policy-0",
                                "document_id": "hotel_policy",
                                "chunk_index": 0,
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

    assert "酒店取消与入住政策 / 免费取消：rag/documents/policies/hotel_policy.md" in response
