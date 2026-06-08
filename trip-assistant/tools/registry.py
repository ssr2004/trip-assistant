"""
工具注册表
管理所有可用工具
"""
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from models.tool import ToolResult


EXTERNAL_OBSERVABILITY_KEYS = {
    "api_status",
    "execution_mode",
    "fallback_reason",
    "fallback_used",
    "mock_reason",
    "error_type",
    "attempt_count",
    "retry_count",
    "cache_enabled",
    "cache_hit",
    "cache_backend",
    "cache_ttl",
    "cache_write",
    "cache_error",
}


class BaseTool(ABC):
    """工具基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """执行工具"""
        pass

    def success_result(
        self,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构建标准成功工具结果"""
        return ToolResult(
            success=True,
            data=data,
            error=None,
            metadata=self._build_metadata(metadata),
        ).to_dict()

    def error_result(
        self,
        error: str,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构建标准失败工具结果"""
        return ToolResult(
            success=False,
            data=data or {},
            error=error,
            metadata=self._build_metadata(metadata),
        ).to_dict()

    def _build_metadata(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """补充工具元数据"""
        merged_metadata = dict(metadata or {})
        merged_metadata.setdefault("tool", self.name)
        return merged_metadata

    def external_metadata(self, api_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """提取外部API脱敏观测字段，供工具结果和trace透传。"""
        metadata = {}
        for key in EXTERNAL_OBSERVABILITY_KEYS:
            value = (api_metadata or {}).get(key)
            if value is not None:
                metadata[key] = value
        if (api_metadata or {}).get("source"):
            metadata["external_source"] = api_metadata["source"]
        return metadata


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        """初始化工具注册表"""
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """注册默认工具"""
        from tools.flights import FlightTool
        from tools.hotels import HotelTool
        from tools.attractions import AttractionTool
        from tools.policy import PolicyTool
        from tools.guide import GuideTool
        from tools.itinerary import ItineraryTool
        from tools.routes import RouteTool
        from tools.weather import WeatherTool
        from tools.trains import TrainTool

        self.register(FlightTool())
        self.register(TrainTool())
        self.register(HotelTool())
        self.register(AttractionTool())
        self.register(PolicyTool())
        self.register(GuideTool())
        self.register(ItineraryTool())
        self.register(RouteTool())
        self.register(WeatherTool())

    def register(self, tool: BaseTool):
        """
        注册工具

        Args:
            tool: 工具实例
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """
        获取工具

        Args:
            name: 工具名称

        Returns:
            工具实例
        """
        return self._tools.get(name)

    async def execute(self, tool_name: str, params: Dict) -> Any:
        """
        执行工具

        Args:
            tool_name: 工具名称
            params: 工具参数

        Returns:
            执行结果
        """
        tool = self.get(tool_name)
        if not tool:
            raise ValueError(f"工具不存在: {tool_name}")

        result = await tool.execute(**params)
        return self._normalize_result(result)

    def _normalize_result(self, result: Any) -> Any:
        """规范化工具返回结果，兼容ToolResult和普通dict"""
        if isinstance(result, ToolResult):
            return result.to_dict()
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result

    def list_tools(self) -> List[Dict]:
        """列出所有工具"""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self._tools.values()
        ]
