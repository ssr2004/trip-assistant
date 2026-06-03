"""
旅行Agent核心模块
实现主Agent逻辑，协调各个子模块
"""
from typing import Dict, List, AsyncGenerator, Any

from langchain_core.messages import HumanMessage, AIMessage
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

        # 从RAG检索相关文档
        rag_results = self.rag_retriever.retrieve(message, top_k=3)

        # 从记忆系统检索相关信息
        memory_context = self.memory_manager.retrieve(message)

        return {
            "rag_context": rag_results,
            "memory_context": memory_context,
        }

    async def _plan_tasks(self, state: AgentState) -> Dict:
        """规划执行任务"""
        intent = state.get("intent")
        user_query = state["messages"][-1].content
        context = {
            "query": user_query,
            "rag": state.get("rag_context", []),
            "memory": state.get("memory_context", {}),
        }

        tasks = self.task_planner.plan(intent, context)
        return {"tasks": tasks}

    async def _execute_tasks(self, state: AgentState) -> Dict:
        """执行任务"""
        tasks = state.get("tasks", [])
        results = []

        for task in sorted(tasks, key=lambda item: item.get("priority", 99)):
            task_type = task.get("task_type", "tool_call")

            if task_type == "ask_user":
                results.append(self._build_internal_result(task, task.get("params", {})))
                continue

            if task_type == "recommend_destination":
                results.append(self._build_internal_result(task, self._recommend_destinations(task)))
                continue

            tool_name = task.get("tool")
            params = task.get("params", {})

            if not tool_name:
                results.append(self._build_error_result(task, "任务缺少工具名称"))
                continue

            try:
                result = await self.tool_registry.execute(tool_name, params)
                results.append({
                    "task": task,
                    "success": True,
                    "result": result,
                    "error": None,
                })
            except Exception as exc:
                results.append(self._build_error_result(task, str(exc)))

        return {"task_results": results}

    async def _generate_response(self, state: AgentState) -> Dict:
        """生成最终回复"""
        task_results = state.get("task_results", [])
        intent = state.get("intent")

        response = self._build_response(intent, task_results)

        # 保存到记忆
        self.memory_manager.save(
            user_message=state["messages"][-1].content,
            ai_message=response,
        )

        return {"messages": [AIMessage(content=response)]}

    def _build_internal_result(self, task: Dict, data: Any) -> Dict:
        """构建内部任务结果"""
        return {
            "task": task,
            "success": True,
            "result": data,
            "error": None,
        }

    def _build_error_result(self, task: Dict, error: str) -> Dict:
        """构建失败任务结果"""
        return {
            "task": task,
            "success": False,
            "result": None,
            "error": error,
        }

    def _recommend_destinations(self, task: Dict) -> Dict:
        """目的地推荐占位实现，后续接入LLM和真实数据"""
        params = task.get("params", {})
        preferences = params.get("preferences", [])
        budget = params.get("budget")

        candidates = [
            {"city": "杭州", "reason": "自然风光、人文景点和美食都比较均衡，适合初次旅行规划。"},
            {"city": "成都", "reason": "美食丰富、节奏轻松，适合休闲和低强度旅行。"},
            {"city": "厦门", "reason": "海边城市，适合放松、拍照和情侣出行。"},
        ]

        if "海边" in preferences or "自然风光" in preferences:
            candidates.insert(0, {"city": "三亚", "reason": "海滨度假属性强，适合想看海和放松的用户。"})

        return {
            "budget": budget,
            "preferences": preferences,
            "candidates": candidates[:3],
            "message": "我先根据您的预算、时间和偏好推荐几个候选目的地，您确认后我可以继续规划航班、酒店和行程。",
        }

    def _build_response(self, intent: dict, task_results: list) -> str:
        """构建回复内容"""
        if not task_results:
            return "我已经理解您的需求，但当前暂时没有需要执行的工具任务。"

        ask_user_result = self._find_result_by_task_type(task_results, "ask_user")
        if ask_user_result:
            return ask_user_result["result"].get("question", "请补充一下旅行需求中的关键信息。")

        destination_result = self._find_result_by_task_type(task_results, "recommend_destination")
        if destination_result:
            return self._format_destination_recommendations(destination_result["result"])

        response_parts = []
        for item in task_results:
            task = item.get("task", {})
            name = task.get("name", task.get("tool", "任务"))
            if item.get("success"):
                response_parts.append(f"【{name}】\n{item.get('result')}")
            else:
                response_parts.append(f"【{name}】执行失败：{item.get('error')}")

        return "\n\n".join(response_parts) if response_parts else "处理完成"

    def _find_result_by_task_type(self, task_results: list, task_type: str) -> Dict | None:
        """按任务类型查找结果"""
        for result in task_results:
            task = result.get("task", {})
            if task.get("task_type") == task_type and result.get("success"):
                return result
        return None

    def _format_destination_recommendations(self, result: Dict) -> str:
        """格式化目的地推荐结果"""
        lines = [result.get("message", "为您推荐以下候选目的地：")]
        for index, candidate in enumerate(result.get("candidates", []), start=1):
            lines.append(f"{index}. {candidate['city']}：{candidate['reason']}")
        lines.append("\n您可以选择其中一个城市，我再继续为您规划交通、酒店和每日行程。")
        return "\n".join(lines)

    async def arun(self, message: str, session_id: str) -> str:
        """异步运行Agent"""
        config = {"configurable": {"thread_id": session_id}}

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "intent": None,
            "tasks": [],
            "task_results": [],
            "rag_context": [],
            "memory_context": {},
        }

        result = await self.graph.ainvoke(initial_state, config)

        if result.get("messages"):
            return result["messages"][-1].content
        return "处理完成"

    async def astream(self, message: str, session_id: str) -> AsyncGenerator[str, None]:
        """流式运行Agent"""
        response = await self.arun(message, session_id)
        yield response

    def get_history(self, session_id: str) -> List[Dict]:
        """获取对话历史"""
        return self.memory_manager.get_history(session_id)

    def clear_history(self, session_id: str):
        """清除对话历史"""
        self.memory_manager.clear_history(session_id)
