"""
任务规划数据模型
定义Agent任务规划阶段输出的结构化任务
"""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


TaskType = Literal[
    "ask_user",
    "tool_call",
    "recommend_destination",
    "generate_itinerary",
    "dynamic_rag_query",
    "revise_itinerary",
]


class PlanningTask(BaseModel):
    """规划任务"""

    task_id: str = Field(..., description="任务ID")
    task_type: TaskType = Field(..., description="任务类型")
    name: str = Field(..., description="任务名称")
    priority: int = Field(..., description="执行优先级")
    tool: Optional[str] = Field(None, description="需要调用的工具名称")
    params: Dict[str, Any] = Field(default_factory=dict, description="任务参数")
    reason: str = Field("", description="规划该任务的原因")
    depends_on: List[str] = Field(default_factory=list, description="依赖任务ID列表")


class TaskPlan(BaseModel):
    """任务计划"""

    intent: str = Field(..., description="意图类型")
    tasks: List[PlanningTask] = Field(default_factory=list, description="规划出的任务列表")
    need_user_input: bool = Field(False, description="是否需要用户补充信息")
    summary: str = Field("", description="任务计划摘要")
