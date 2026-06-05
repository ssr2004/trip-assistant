"""
Agent状态定义
定义LangGraph工作流的状态结构
"""
from typing import Any, TypedDict, List, Dict, Optional, Annotated, NotRequired
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


TASK_RESULT_REQUIRED_FIELDS = {"task", "success", "result", "error", "meta"}

TASK_RESULT_META_FIELDS = {
    "duration_ms",
    "execution_mode",
    "error_type",
    "result_summary",
    "failure_category",
    "recoverable",
    "degraded",
    "fallback_used",
    "recovery_strategy",
    "degradation_reason",
    "dependency_ids",
    "resolved_dependencies",
    "missing_dependencies",
    "failed_dependencies",
    "dependency_error_count",
    "dependency_context_keys",
}


class TaskResultMeta(TypedDict, total=False):
    """Executor-owned task observability and recovery metadata."""

    duration_ms: int
    execution_mode: str
    error_type: Optional[str]
    result_summary: Optional[str]
    failure_category: Optional[str]
    recoverable: bool
    degraded: bool
    fallback_used: bool
    recovery_strategy: str
    degradation_reason: str
    dependency_ids: List[str]
    resolved_dependencies: List[str]
    missing_dependencies: List[str]
    failed_dependencies: List[str]
    dependency_error_count: int
    dependency_context_keys: List[str]


class TaskResult(TypedDict):
    """Executor output consumed by ResponseBuilder, artifacts and trace."""

    task: Dict[str, Any]
    success: bool
    result: Any
    error: Optional[str]
    meta: TaskResultMeta


class AgentState(TypedDict):
    """Agent状态"""

    # 消息历史
    messages: Annotated[List[BaseMessage], add_messages]

    # 解析的意图
    intent: Optional[Dict]

    # 规划的任务列表
    tasks: List[Dict]

    # 任务执行结果
    task_results: List[TaskResult]

    planner_metadata: NotRequired[Dict]

    # RAG检索上下文
    rag_context: List[Dict]

    # 记忆系统上下文
    memory_context: Dict

    # 动态RAG上下文
    dynamic_rag_context: NotRequired[Dict]

    # 会话ID
    session_id: Optional[str]
