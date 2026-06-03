"""
旅行Agent核心模块
实现主Agent逻辑，协调各个子模块
"""
from typing import Dict, List, Optional, AsyncGenerator
import uuid

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.state import AgentState
from core.planner import TaskPlanner
from core.memory.manager import MemoryManager
from core.intent import IntentParser
from rag.retriever import RAGRetriever
from tools.registry import ToolRegistry


class TravelAgent:
    """旅行Agent主类"""

    def __init__(self):
        """初始化Agent"""
        self.intent_parser = IntentParser()
        self.task_planner = TaskPlanner()
        self.memory_manager = MemoryManager()
        self.rag_retriever = RAGRetriever()
        self.tool_registry = ToolRegistry()

        # 构建工作流图
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建LangGraph工作流"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("parse_intent", self._parse_intent)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("plan_tasks", self._plan_tasks)
        workflow.add_node("execute_tasks", self._execute_tasks)
        workflow.add_node("generate_response", self._generate_response)

        # 定义边
        workflow.set_entry_point("parse_intent")
        workflow.add_edge("parse_intent", "retrieve_context")
        workflow.add_edge("retrieve_context", "plan_tasks")
        workflow.add_edge("plan_tasks", "execute_tasks")
        workflow.add_edge("execute_tasks", "generate_response")
        workflow.add_edge("generate_response", END)

        # 编译图
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def _parse_intent(self, state: AgentState) -> Dict:
        """解析用户意图"""
        message = state["messages"][-1].content
        intent = self.intent_parser.parse(message)
        return {"intent": intent}

    async def _retrieve_context(self, state: AgentState) -> Dict:
        """检索相关上下文"""
        message = state["messages"][-1].content
        intent = state.get("intent")

        # 从RAG检索相关文档
        rag_results = self.rag_retriever.retrieve(message, top_k=3)

        # 从记忆系统检索相关信息
        memory_context = self.memory_manager.retrieve(message)

        return {
            "rag_context": rag_results,
            "memory_context": memory_context
        }

    async def _plan_tasks(self, state: AgentState) -> Dict:
        """规划执行任务"""
        intent = state.get("intent")
        context = {
            "rag": state.get("rag_context", []),
            "memory": state.get("memory_context", {})
        }

        tasks = self.task_planner.plan(intent, context)
        return {"tasks": tasks}

    async def _execute_tasks(self, state: AgentState) -> Dict:
        """执行任务"""
        tasks = state.get("tasks", [])
        results = []

        for task in tasks:
            tool_name = task.get("tool")
            params = task.get("params", {})

            # 调用对应工具
            result = await self.tool_registry.execute(tool_name, params)
            results.append({
                "task": task,
                "result": result
            })

        return {"task_results": results}

    async def _generate_response(self, state: AgentState) -> Dict:
        """生成最终回复"""
        # 这里需要调用LLM生成回复
        # 暂时返回简单格式
        task_results = state.get("task_results", [])
        intent = state.get("intent")

        # 构建回复
        response = self._build_response(intent, task_results)

        # 保存到记忆
        self.memory_manager.save(
            user_message=state["messages"][-1].content,
            ai_message=response
        )

        return {"messages": [AIMessage(content=response)]}

    def _build_response(self, intent: dict, task_results: list) -> str:
        """构建回复内容"""
        # 简单的回复构建逻辑
        if not task_results:
            return "抱歉，我无法处理您的请求。"

        response_parts = []
        for result in task_results:
            if result.get("result"):
                response_parts.append(str(result["result"]))

        return "\n\n".join(response_parts) if response_parts else "处理完成"

    async def arun(self, message: str, session_id: str) -> str:
        """异步运行Agent"""
        config = {"configurable": {"thread_id": session_id}}

        # 获取或创建初始状态
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "intent": None,
            "tasks": [],
            "task_results": [],
            "rag_context": [],
            "memory_context": {}
        }

        # 运行工作流
        result = await self.graph.ainvoke(initial_state, config)

        # 返回最后一条AI消息
        if result.get("messages"):
            return result["messages"][-1].content
        return "处理完成"

    async def astream(self, message: str, session_id: str) -> AsyncGenerator[str, None]:
        """流式运行Agent"""
        config = {"configurable": {"thread_id": session_id}}

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "intent": None,
            "tasks": [],
            "task_results": [],
            "rag_context": [],
            "memory_context": {}
        }

        # 流式运行
        async for event in self.graph.astream(initial_state, config):
            # 解析事件，生成流式输出
            if "messages" in event:
                for msg in event["messages"]:
                    if isinstance(msg, AIMessage):
                        yield msg.content

    def get_history(self, session_id: str) -> List[Dict]:
        """获取对话历史"""
        return self.memory_manager.get_history(session_id)

    def clear_history(self, session_id: str):
        """清除对话历史"""
        self.memory_manager.clear_history(session_id)
