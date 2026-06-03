"""
工具返回结果模型
定义工具层统一返回协议
"""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """工具标准返回结果"""

    success: bool = Field(..., description="工具是否执行成功")
    data: Dict[str, Any] = Field(default_factory=dict, description="工具返回数据")
    error: Optional[str] = Field(None, description="错误信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="工具元数据")

    def to_dict(self) -> Dict[str, Any]:
        """转换为普通字典，兼容现有响应构建逻辑"""
        return self.model_dump()
