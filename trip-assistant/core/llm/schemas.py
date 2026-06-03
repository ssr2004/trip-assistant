"""
LLM调用数据模型
定义统一的LLM请求和响应结构
"""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


MessageRole = Literal["system", "user", "assistant"]
ResponseFormat = Literal["text", "json_object"]


class LLMMessage(BaseModel):
    """LLM消息"""

    role: MessageRole = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")


class LLMRequest(BaseModel):
    """LLM请求"""

    messages: List[LLMMessage] = Field(default_factory=list, description="消息列表")
    model: Optional[str] = Field(None, description="模型名称，默认使用配置中的模型")
    temperature: Optional[float] = Field(None, description="采样温度，默认使用配置中的温度")
    response_format: ResponseFormat = Field("text", description="响应格式")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="调用上下文元信息")


class LLMResponse(BaseModel):
    """LLM响应"""

    success: bool = Field(..., description="调用是否成功")
    content: str = Field("", description="模型输出内容")
    error: Optional[str] = Field(None, description="失败原因")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="响应元信息")
