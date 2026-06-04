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
from core.response_builder import ResponseBuilder
from rag.dynamic_store import DynamicRAGStore
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
        self.dynamic_rag_store = DynamicRAGStore()
        self.dynamic_rag_stores = {"default": self.dynamic_rag_store}
        self.tool_registry = ToolRegistry()
        self.response_builder = ResponseBuilder()

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
        intent = await self.intent_parser.parse_async(message)
        return {"intent": intent}

    async def _retrieve_context(self, state: AgentState) -> Dict:
        """检索相关上下文"""
        message = state["messages"][-1].content

        # 从RAG检索相关文档
        rag_results = self.rag_retriever.retrieve(message, top_k=3)

        # 从记忆系统检索相关信息
        session_id = state.get("session_id", "default")
        memory_context = self.memory_manager.retrieve(message, session_id)

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

        tasks = await self.task_planner.plan_async(intent, context)
        return {"tasks": tasks}

    async def _execute_tasks(self, state: AgentState) -> Dict:
        """执行任务"""
        tasks = state.get("tasks", [])
        results = []

        result_by_task_id = {}

        for task in sorted(tasks, key=lambda item: item.get("priority", 99)):
            task_type = task.get("task_type", "tool_call")

            if task_type == "ask_user":
                task_result = self._build_internal_result(task, task.get("params", {}))
                results.append(task_result)
                self._index_task_result(task_result, result_by_task_id)
                continue

            if task_type == "recommend_destination":
                task_result = self._build_internal_result(task, self._recommend_destinations(task))
                results.append(task_result)
                self._index_task_result(task_result, result_by_task_id)
                continue

            if task_type == "dynamic_rag_query":
                task_result = self._build_internal_result(
                    task,
                    self._query_dynamic_rag(task, state.get("session_id", "default")),
                )
                results.append(task_result)
                self._index_task_result(task_result, result_by_task_id)
                continue

            tool_name = task.get("tool")
            params = self._inject_dependency_context(
                task=task,
                params=task.get("params", {}),
                result_by_task_id=result_by_task_id,
            )

            if not tool_name:
                task_result = self._build_error_result(task, "任务缺少工具名称")
                results.append(task_result)
                self._index_task_result(task_result, result_by_task_id)
                continue

            try:
                result = await self.tool_registry.execute(tool_name, params)
                result = self._normalize_tool_result(result)
                tool_success = result.get("success", True) if isinstance(result, dict) else True
                tool_error = result.get("error") if isinstance(result, dict) else None
                task_result = {
                    "task": task,
                    "success": tool_success,
                    "result": result,
                    "error": tool_error,
                }
                results.append(task_result)
                self._index_task_result(task_result, result_by_task_id)
            except Exception as exc:
                task_result = self._build_error_result(task, str(exc))
                results.append(task_result)
                self._index_task_result(task_result, result_by_task_id)

        messages = state.get("messages", [])
        query = messages[-1].content if messages else ""
        dynamic_rag_context = self._collect_dynamic_rag_context(
            results,
            query,
            state.get("session_id", "default"),
        )
        return {"task_results": results, "dynamic_rag_context": dynamic_rag_context}

    async def _generate_response(self, state: AgentState) -> Dict:
        """生成最终回复"""
        task_results = state.get("task_results", [])
        intent = state.get("intent")
        memory_context = state.get("memory_context", {})

        response = self.response_builder.build(intent, task_results, memory_context)

        # 保存到记忆
        self.memory_manager.save(
            user_message=state["messages"][-1].content,
            ai_message=response,
            session_id=state.get("session_id", "default"),
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

    def _index_task_result(self, task_result: Dict, result_by_task_id: Dict[str, Dict]) -> None:
        """按task_id索引任务结果，供后续依赖任务读取"""
        task = task_result.get("task", {})
        task_id = task.get("task_id")
        if task_id:
            result_by_task_id[task_id] = task_result

    def _normalize_tool_result(self, result: Any) -> Any:
        """规范化工具结果，兼容ToolResult/Pydantic模型和普通dict"""
        if hasattr(result, "to_dict"):
            return result.to_dict()
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result

    def _collect_dynamic_rag_context(self, task_results: List[Dict], query: str, session_id: str) -> Dict:
        """收集工具返回的外部RAG文档并写入动态存储"""
        store = self._get_dynamic_rag_store(session_id)
        documents = []
        for task_result in task_results:
            data = self._extract_result_data(task_result)
            if not isinstance(data, dict):
                continue
            rag_documents = data.get("rag_documents") or []
            if isinstance(rag_documents, list):
                documents.extend(item for item in rag_documents if isinstance(item, dict))

        if documents:
            store.add_documents(documents)

        sources = store.search(query, top_k=3) if query else []
        return {
            "documents": documents,
            "sources": sources,
            "document_count": len(store.list_documents()),
        }

    def _get_dynamic_rag_store(self, session_id: str) -> DynamicRAGStore:
        """获取会话级动态RAG存储"""
        key = session_id or "default"
        if key not in self.dynamic_rag_stores:
            self.dynamic_rag_stores[key] = DynamicRAGStore()
        self.dynamic_rag_store = self.dynamic_rag_stores[key]
        return self.dynamic_rag_stores[key]

    def _query_dynamic_rag(self, task: Dict, session_id: str) -> Dict:
        """基于会话动态RAG文档回答追问"""
        query = (task.get("params", {}) or {}).get("query", "")
        store = self._get_dynamic_rag_store(session_id)
        documents = store.list_documents()
        if not documents:
            return {
                "query": query,
                "answer": "我暂时没有在本轮对话的外部景点数据中找到相关信息。您可以先让我查询目的地景点，例如“杭州有什么好玩的”。",
                "sources": [],
                "documents": [],
            }

        if self._is_dynamic_list_query(query):
            selected_documents = documents[:5]
            answer = self._build_dynamic_list_answer(selected_documents, include_source="来源" in query)
            return {
                "query": query,
                "answer": answer,
                "sources": selected_documents[:3],
                "documents": selected_documents,
            }

        sources = store.search(query, top_k=3)
        if not sources:
            return {
                "query": query,
                "answer": "我暂时没有在本轮对话的外部景点数据中找到相关信息。您可以换一个景点名称或先查询目的地景点。",
                "sources": [],
                "documents": documents,
            }

        matched_documents = self._match_dynamic_documents(sources, documents)
        return {
            "query": query,
            "answer": self._build_dynamic_detail_answer(matched_documents or sources),
            "sources": sources,
            "documents": documents,
        }

    def _is_dynamic_list_query(self, query: str) -> bool:
        """判断是否询问刚才推荐的景点列表或来源"""
        return any(keyword in query for keyword in ["刚才推荐", "刚刚推荐", "推荐的景点", "这些景点", "有哪些"])

    def _build_dynamic_list_answer(self, documents: List[Dict], include_source: bool = False) -> str:
        """构建动态景点列表回答"""
        lines = ["根据刚才推荐的外部景点数据，目前记录的景点有："]
        for index, document in enumerate(documents, start=1):
            line = f"{index}. {document.get('title', '景点')}"
            if include_source:
                line += f"：{document.get('source', '外部景点数据')}"
            lines.append(line)
        return "\n".join(lines)

    def _match_dynamic_documents(self, sources: List[Dict], documents: List[Dict]) -> List[Dict]:
        """根据检索片段匹配完整动态文档"""
        matched = []
        for source in sources:
            document_id = source.get("document_id")
            source_path = source.get("source")
            title = source.get("title")
            for document in documents:
                if document in matched:
                    continue
                if (
                    document.get("document_id") == document_id
                    or document.get("source") == source_path
                    or document.get("title") == title
                ):
                    matched.append(document)
                    break
        return matched

    def _build_dynamic_detail_answer(self, sources: List[Dict]) -> str:
        """构建动态景点详情回答"""
        lines = ["根据刚才推荐的外部景点数据，我查到："]
        for source in sources[:3]:
            title = source.get("title") or "相关景点"
            excerpt = source.get("content") or source.get("excerpt") or "暂无更多详情。"
            lines.append(f"\n{title}：")
            for line in excerpt.splitlines():
                cleaned = line.strip().lstrip("#").strip().lstrip("-").strip()
                if cleaned and cleaned != title:
                    lines.append(f"- {cleaned}")
        return "\n".join(lines)

    def _inject_dependency_context(
        self,
        task: Dict,
        params: Dict,
        result_by_task_id: Dict[str, Dict],
    ) -> Dict:
        """将depends_on声明的前置任务结果注入到当前工具参数"""
        depends_on = task.get("depends_on", []) or []
        if not depends_on:
            return params or {}

        merged_params = dict(params or {})
        dependency_context = self._build_dependency_context(depends_on, result_by_task_id)
        existing_context = merged_params.get("context")
        if isinstance(existing_context, dict):
            dependency_context = {**dependency_context, **existing_context}
        merged_params["context"] = dependency_context
        return merged_params

    def _build_dependency_context(
        self,
        depends_on: List[str],
        result_by_task_id: Dict[str, Dict],
    ) -> Dict:
        """构建供后续任务使用的依赖上下文"""
        context = {
            "flights": [],
            "hotels": [],
            "attractions": [],
            "guide": None,
            "raw_results": {},
            "errors": {},
        }

        for dependency in depends_on:
            task_result = result_by_task_id.get(dependency)
            if not task_result:
                context["errors"][dependency] = "依赖任务尚未执行或不存在"
                continue

            task = task_result.get("task", {})
            tool_name = task.get("tool")
            context["raw_results"][dependency] = {
                "task": task,
                "success": task_result.get("success", False),
                "result": task_result.get("result"),
                "error": task_result.get("error"),
            }

            if not task_result.get("success", False):
                context["errors"][dependency] = task_result.get("error") or "依赖任务执行失败"

            data = self._extract_result_data(task_result)
            if tool_name == "search_flights":
                context["flights"] = self._safe_list(data, "flights")
            elif tool_name == "search_hotels":
                context["hotels"] = self._safe_list(data, "hotels")
            elif tool_name == "search_attractions":
                context["attractions"] = self._safe_list(data, "attractions")
            elif tool_name == "retrieve_guide":
                context["guide"] = data if isinstance(data, dict) else None

        return context

    def _extract_result_data(self, task_result: Dict) -> Any:
        """提取标准工具结果中的data，兼容内部任务结果"""
        result = task_result.get("result")
        if isinstance(result, dict) and "data" in result:
            return result.get("data")
        return result

    def _safe_list(self, data: Any, key: str) -> List:
        """从工具data中安全提取列表字段"""
        if not isinstance(data, dict):
            return []
        value = data.get(key, [])
        return value if isinstance(value, list) else []

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
            "dynamic_rag_context": {},
            "session_id": session_id,
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
        """清除对话历史和会话动态RAG文档"""
        self.memory_manager.clear_history(session_id)
        self.dynamic_rag_stores.pop(session_id or "default", None)
