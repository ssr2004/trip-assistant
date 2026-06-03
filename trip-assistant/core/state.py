"""
Agent状态定义
定义LangGraph工作流的状态结构
"""
from typing import TypedDict, List, Dict, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """Agent状态"""

    # 消息历史
    messages: Annotated[List[BaseMessage], add_messages]

    # 解析的意图
    intent: Optional[Dict]

    # 规划的任务列表
    tasks: List[Dict]

    # 任务执行结果
    task_results: List[Dict]

    # RAG检索上下文
    rag_context: List[Dict]

    # 记忆系统上下文
    memory_context: Dict
