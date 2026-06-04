"""
意图识别LLM fallback测试
"""
import pytest

from core.intent import IntentParser
from core.llm import LLMResponse


class FakeLLMClient:
    """可控LLM客户端"""

    def __init__(
        self,
        content: str = "",
        success: bool = True,
        available: bool = True,
        contents: list[str] | None = None,
        successes: list[bool] | None = None,
    ):
        self.content = content
        self.success = success
        self.available = available
        self.contents = contents or []
        self.successes = successes or []
        self.calls = 0
        self.last_request = None

    async def chat(self, request):
        self.calls += 1
        self.last_request = request
        content = self.contents[self.calls - 1] if self.calls <= len(self.contents) else self.content
        success = self.successes[self.calls - 1] if self.calls <= len(self.successes) else self.success
        return LLMResponse(
            success=success,
            content=content,
            error=None if success else "mock llm failed",
            metadata={"mock": True, "model": "fake-model", "provider": "fake"},
        )


@pytest.mark.asyncio
async def test_parse_async_uses_rule_result_when_llm_unavailable():
    """没有LLM能力时继续使用规则解析结果"""
    llm_client = FakeLLMClient(available=False)
    parser = IntentParser(llm_client=llm_client)

    result = await parser.parse_async("我想随便聊聊")

    assert result["intent"] == "general_chat"
    assert llm_client.calls == 0


@pytest.mark.asyncio
async def test_parse_async_skips_llm_for_high_confidence_rule_result():
    """规则高置信度时不调用LLM"""
    llm_client = FakeLLMClient(available=True)
    parser = IntentParser(llm_client=llm_client)

    result = await parser.parse_async("我要从郑州去杭州玩三天，预算3000，6月10日出发")

    assert result["intent"] == "travel_plan"
    assert result["metadata"]["source"] == "rule"
    assert result["entities"]["origin"] == "郑州"
    assert result["entities"]["destination"] == "杭州"
    assert result["entities"]["duration"] == 3
    assert llm_client.calls == 0


@pytest.mark.asyncio
async def test_parse_async_uses_llm_for_general_chat_travel_need():
    """规则识别为general_chat时可通过LLM补充为旅行规划"""
    llm_content = """
    {
      "intent": "travel_plan",
      "entities": {
        "origin": null,
        "destination": null,
        "departure_date": null,
        "return_date": null,
        "duration": 3,
        "budget": 3000,
        "travelers": null,
        "preferences": ["海边", "放松", "慢节奏"]
      },
      "confidence": 0.82,
      "missing_slots": ["origin", "destination", "departure_date"],
      "followup_question": "请问您准备从哪个城市出发？"
    }
    """
    llm_client = FakeLLMClient(content=llm_content)
    parser = IntentParser(llm_client=llm_client)

    result = await parser.parse_async("我想找个海边城市放松三天，预算3000")

    assert result["intent"] == "travel_plan"
    assert result["entities"]["duration"] == 3
    assert result["entities"]["budget"] == 3000
    assert "海边" in result["entities"]["preferences"]
    assert "destination" in result["missing_slots"]
    assert result["metadata"]["source"] == "llm"
    assert llm_client.calls == 1
    assert llm_client.last_request.response_format == "json_object"


@pytest.mark.asyncio
async def test_parse_async_supports_markdown_json_from_llm():
    """LLM返回Markdown JSON代码块时也可以解析"""
    llm_content = """```json
{
  "intent": "policy_query",
  "entities": {
    "origin": null,
    "destination": null,
    "departure_date": null,
    "return_date": null,
    "duration": null,
    "budget": null,
    "travelers": null,
    "preferences": []
  },
  "confidence": 0.7,
  "missing_slots": [],
  "followup_question": null
}
```"""
    llm_client = FakeLLMClient(content=llm_content)
    parser = IntentParser(llm_client=llm_client)

    result = await parser.parse_async("我想问一下退订这方面")

    assert result["intent"] == "policy_query"
    assert result["missing_slots"] == []


@pytest.mark.asyncio
async def test_parse_async_falls_back_when_llm_returns_invalid_json():
    """LLM返回非法JSON时回退规则结果"""
    llm_client = FakeLLMClient(content="这不是JSON")
    parser = IntentParser(llm_client=llm_client)

    result = await parser.parse_async("我想找个地方散散心")

    assert result["intent"] == "general_chat"
    assert result["metadata"]["source"] == "rule_fallback"
    assert result["metadata"]["llm_error_type"] == "json_parse_failed"
    assert result["metadata"]["json_repair_attempted"] is True
    assert result["metadata"]["json_repair_success"] is False
    assert llm_client.calls == 2


@pytest.mark.asyncio
async def test_parse_async_records_llm_error_type_when_call_fails():
    """LLM调用失败时记录错误分类并回退规则结果"""
    llm_client = FakeLLMClient(success=False)
    parser = IntentParser(llm_client=llm_client)

    result = await parser.parse_async("我想找个地方散散心")

    assert result["intent"] == "general_chat"
    assert result["metadata"]["source"] == "rule_fallback"
    assert result["metadata"]["llm_error_type"] is None
    assert llm_client.calls == 1


@pytest.mark.asyncio
async def test_parse_async_falls_back_when_llm_schema_invalid():
    """LLM返回结构不符合模型时回退规则结果"""
    llm_client = FakeLLMClient(content='{"intent": "unknown", "entities": {}}')
    parser = IntentParser(llm_client=llm_client)

    result = await parser.parse_async("我想找个地方散散心")

    assert result["intent"] == "general_chat"
    assert result["metadata"]["source"] == "rule_fallback"
    assert result["metadata"]["llm_error_type"] == "schema_validation_failed"
    assert result["metadata"]["json_repair_attempted"] is True
    assert result["metadata"]["json_repair_success"] is False
    assert llm_client.calls == 2


@pytest.mark.asyncio
async def test_parse_async_repairs_malformed_json_once():
    """Malformed JSON can be repaired once before falling back."""
    repaired_content = """
    {
      "intent": "travel_plan",
      "entities": {
        "origin": null,
        "destination": "杭州",
        "departure_date": null,
        "return_date": null,
        "duration": 2,
        "budget": null,
        "travelers": null,
        "preferences": ["美食"]
      },
      "confidence": 0.76,
      "missing_slots": ["origin", "departure_date"],
      "followup_question": null
    }
    """
    llm_client = FakeLLMClient(contents=["{bad json", repaired_content])
    parser = IntentParser(llm_client=llm_client)

    result = await parser.parse_async("帮我找个周末美食旅行地")

    assert result["intent"] == "travel_plan"
    assert result["entities"]["destination"] == "杭州"
    assert result["metadata"]["source"] == "llm"
    assert result["metadata"]["json_repair_attempted"] is True
    assert result["metadata"]["json_repair_success"] is True
    assert llm_client.calls == 2
    assert llm_client.last_request.metadata["repair_for"] == "structured_json"


@pytest.mark.asyncio
async def test_parse_async_repairs_schema_invalid_json_once():
    """Schema-invalid JSON can be repaired once before falling back."""
    repaired_content = """
    {
      "intent": "policy_query",
      "entities": {
        "origin": null,
        "destination": null,
        "departure_date": null,
        "return_date": null,
        "duration": null,
        "budget": null,
        "travelers": null,
        "preferences": []
      },
      "confidence": 0.68,
      "missing_slots": [],
      "followup_question": null
    }
    """
    llm_client = FakeLLMClient(contents=['{"intent": "unknown", "entities": {}}', repaired_content])
    parser = IntentParser(llm_client=llm_client)

    result = await parser.parse_async("我想找个地方散散心")

    assert result["intent"] == "policy_query"
    assert result["metadata"]["source"] == "llm"
    assert result["metadata"]["json_repair_attempted"] is True
    assert result["metadata"]["json_repair_success"] is True
    assert llm_client.calls == 2


@pytest.mark.asyncio
async def test_parse_async_does_not_leak_content_when_repair_fails():
    """Repair failure metadata should not expose raw malformed model output."""
    llm_client = FakeLLMClient(contents=["not json with sk-test-secret", "still not json"])
    parser = IntentParser(llm_client=llm_client)

    result = await parser.parse_async("随便推荐一个地方散散心")

    assert result["metadata"]["source"] == "rule_fallback"
    assert result["metadata"]["json_repair_attempted"] is True
    assert "sk-test-secret" not in str(result)
    assert llm_client.calls == 2
