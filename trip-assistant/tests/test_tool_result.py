"""
工具返回结构模型测试
"""
from typing import Any

import pytest

from models.tool import ToolResult
from tools.registry import BaseTool, ToolRegistry


class DummyTool(BaseTool):
    """测试用工具"""

    @property
    def name(self) -> str:
        return "dummy_tool"

    @property
    def description(self) -> str:
        return "测试工具"

    async def execute(self, **kwargs) -> Any:
        return self.success_result(data={"value": kwargs.get("value", 1)}, metadata={"source": "dummy"})


class ModelReturningTool(DummyTool):
    """直接返回ToolResult的测试工具"""

    @property
    def name(self) -> str:
        return "model_tool"

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            data={"value": kwargs.get("value", 2)},
            error=None,
            metadata={"source": "model", "tool": self.name},
        )


def test_tool_result_defaults_and_to_dict():
    """ToolResult提供稳定的默认结构和字典转换"""
    result = ToolResult(success=True)

    data = result.to_dict()
    assert data == {
        "success": True,
        "data": {},
        "error": None,
        "metadata": {},
    }


def test_base_tool_success_result_adds_tool_metadata():
    """BaseTool成功结果自动补充工具名"""
    tool = DummyTool()

    result = tool.success_result(data={"items": [1]}, metadata={"source": "unit_test"})

    assert result["success"] is True
    assert result["data"] == {"items": [1]}
    assert result["error"] is None
    assert result["metadata"]["source"] == "unit_test"
    assert result["metadata"]["tool"] == "dummy_tool"


def test_base_tool_error_result_adds_tool_metadata():
    """BaseTool失败结果自动补充错误信息和工具名"""
    tool = DummyTool()

    result = tool.error_result(error="执行失败", data={"items": []}, metadata={"source": "unit_test"})

    assert result["success"] is False
    assert result["data"] == {"items": []}
    assert result["error"] == "执行失败"
    assert result["metadata"]["source"] == "unit_test"
    assert result["metadata"]["tool"] == "dummy_tool"


@pytest.mark.asyncio
async def test_tool_registry_normalizes_tool_result_model():
    """工具注册表可以把ToolResult模型规范化为dict"""
    registry = ToolRegistry()
    registry.register(ModelReturningTool())

    result = await registry.execute("model_tool", {"value": 9})

    assert isinstance(result, dict)
    assert result["success"] is True
    assert result["data"]["value"] == 9
    assert result["metadata"]["tool"] == "model_tool"


@pytest.mark.asyncio
async def test_existing_tool_result_shape_is_unchanged():
    """现有工具对外返回结构保持不变"""
    registry = ToolRegistry()

    result = await registry.execute("search_flights", {"origin": "郑州", "destination": "杭州"})

    assert set(result.keys()) == {"success", "data", "error", "metadata"}
    assert result["success"] is True
    assert result["error"] is None
    assert result["metadata"]["tool"] == "search_flights"
    assert result["metadata"]["source"] == "mock_flight_data"
    assert len(result["data"]["flights"]) == 3
