"""Tests for robust structured JSON parsing from LLM output."""
import pytest

from core.llm import LLMResponse
from core.llm.json_repair import (
    JSON_PARSE_FAILED,
    SCHEMA_VALIDATION_FAILED,
    parse_llm_json_object,
    parse_or_repair_json_object,
)


class FakeRepairLLMClient:
    def __init__(self, content: str = "", success: bool = True):
        self.content = content
        self.success = success
        self.calls = 0
        self.last_request = None

    async def chat(self, request):
        self.calls += 1
        self.last_request = request
        return LLMResponse(
            success=self.success,
            content=self.content,
            error=None if self.success else "repair failed",
            metadata={"error_type": "provider_error"} if not self.success else {},
        )


def test_parse_llm_json_object_supports_markdown_fence():
    content = """```json
{"ok": true, "value": 1}
```"""

    assert parse_llm_json_object(content) == {"ok": True, "value": 1}


def test_parse_llm_json_object_extracts_first_json_object():
    content = 'Here is the result: {"ok": true, "nested": {"value": "x"}} done.'

    assert parse_llm_json_object(content) == {"ok": True, "nested": {"value": "x"}}


@pytest.mark.asyncio
async def test_parse_or_repair_json_object_repairs_parse_failure():
    llm_client = FakeRepairLLMClient(content='{"ok": true}')

    result = await parse_or_repair_json_object(
        "not json",
        llm_client=llm_client,
        validator=lambda data: data if data.get("ok") is True else None,
        repair_instructions="Return {'ok': true}.",
    )

    assert result.success is True
    assert result.repair_attempted is True
    assert result.repair_success is True
    assert result.data == {"ok": True}
    assert llm_client.calls == 1


@pytest.mark.asyncio
async def test_parse_or_repair_json_object_repairs_schema_failure():
    llm_client = FakeRepairLLMClient(content='{"ok": true}')

    def validator(data):
        if data.get("ok") is not True:
            raise ValueError("schema invalid")
        return data

    result = await parse_or_repair_json_object(
        '{"ok": false}',
        llm_client=llm_client,
        validator=validator,
        repair_instructions="Return {'ok': true}.",
    )

    assert result.success is True
    assert result.repair_attempted is True
    assert result.repair_success is True
    assert llm_client.last_request.metadata["repair_failure_type"] == SCHEMA_VALIDATION_FAILED


@pytest.mark.asyncio
async def test_parse_or_repair_json_object_preserves_failure_when_repair_fails():
    llm_client = FakeRepairLLMClient(success=False)

    result = await parse_or_repair_json_object(
        "not json",
        llm_client=llm_client,
        validator=lambda data: data,
        repair_instructions="Return a valid object.",
    )

    assert result.success is False
    assert result.error_type == JSON_PARSE_FAILED
    assert result.repair_attempted is True
    assert result.repair_success is False
    assert result.repair_error_type == "provider_error"
