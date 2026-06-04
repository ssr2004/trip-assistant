"""
旅行Agent核心模块
实现主Agent逻辑，协调各个子模块
"""
import copy
import re
from typing import Dict, List, AsyncGenerator, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.state import AgentState
from core.planner import TaskPlanner
from core.memory.manager import MemoryManager
from core.intent import IntentParser
from core.response_builder import ResponseBuilder
from core.artifacts import normalize_chat_artifacts
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
        self.session_itinerary_contexts = {}
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

            if task_type == "revise_itinerary":
                task_result = self._build_internal_result(
                    task,
                    await self._revise_itinerary(task, state.get("session_id", "default")),
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
        session_id = state.get("session_id", "default")
        dynamic_rag_context = self._collect_dynamic_rag_context(
            results,
            query,
            session_id,
        )
        self._collect_itinerary_context(results, session_id)
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

    def _collect_itinerary_context(self, task_results: List[Dict], session_id: str) -> None:
        """从完整旅行规划结果中收集会话级行程上下文"""
        itinerary_data = None
        attractions = []
        rag_documents = []

        for task_result in task_results:
            task = task_result.get("task", {})
            data = self._extract_result_data(task_result)
            if not isinstance(data, dict):
                continue
            if task.get("tool") == "generate_itinerary" and data.get("itinerary"):
                itinerary_data = data
            if task.get("tool") == "search_attractions":
                attractions = self._safe_list(data, "attractions")
                rag_documents = self._safe_list(data, "rag_documents")

        if not itinerary_data:
            return

        key = session_id or "default"
        self.session_itinerary_contexts[key] = {
            "origin": itinerary_data.get("origin"),
            "destination": itinerary_data.get("destination"),
            "duration": itinerary_data.get("duration"),
            "budget": itinerary_data.get("budget"),
            "travelers": itinerary_data.get("travelers"),
            "preferences": itinerary_data.get("preferences", []),
            "itinerary": copy.deepcopy(itinerary_data.get("itinerary", [])),
            "budget_summary": copy.deepcopy(itinerary_data.get("budget_summary", {})),
            "attractions": copy.deepcopy(attractions),
            "rag_documents": copy.deepcopy(rag_documents),
        }

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

    async def _revise_itinerary(self, task: Dict, session_id: str) -> Dict:
        """基于会话历史行程执行规则型修订"""
        query = (task.get("params", {}) or {}).get("query", "")
        context = self.session_itinerary_contexts.get(session_id or "default")
        if not context:
            return {
                "success": False,
                "query": query,
                "message": "我暂时还没有可调整的历史行程。您可以先让我生成一个完整旅行计划，例如“我要从郑州去杭州玩三天”。",
                "itinerary": [],
                "sources": [],
            }

        revised_context = copy.deepcopy(context)
        itinerary = revised_context.get("itinerary", [])
        attraction_names = self._known_attraction_names(revised_context)
        target_names = self._extract_revision_attractions(query, attraction_names)
        target_day = self._extract_revision_day(query)

        if self._is_weather_adjustment_revision(query):
            result = await self._apply_weather_adjustment_revision(query, itinerary, revised_context)
        elif self._is_route_optimization_revision(query):
            result = await self._apply_route_optimization_revision(query, itinerary, revised_context, target_day)
        elif self._is_replace_revision(query):
            result = self._apply_replace_revision(query, itinerary, revised_context, target_names)
        elif self._is_remove_revision(query):
            result = self._apply_remove_revision(query, itinerary, target_names)
        elif target_day:
            result = self._apply_move_revision(query, itinerary, target_names, target_day)
        else:
            result = {
                "success": False,
                "action": "unknown",
                "summary": "我已收到调整请求，但暂时只支持安排到某一天、移除景点和替换景点。",
            }

        revised_context["itinerary"] = itinerary
        if result.get("success"):
            self.session_itinerary_contexts[session_id or "default"] = revised_context

        return {
            "success": result.get("success", False),
            "query": query,
            "action": result.get("action"),
            "summary": result.get("summary"),
            "message": result.get("message"),
            "itinerary": itinerary,
            "route_summary": result.get("route_summary"),
            "weather_summary": result.get("weather_summary"),
            "sources": ["上一轮旅行方案", "会话动态景点数据"],
        }

    def _known_attraction_names(self, context: Dict) -> List[str]:
        """获取当前会话已知景点名称"""
        names = []
        for attraction in context.get("attractions", []) or []:
            name = attraction.get("name")
            if name and name not in names:
                names.append(name)
        for document in context.get("rag_documents", []) or []:
            title = document.get("title")
            if title and title not in names:
                names.append(title)
        for day in context.get("itinerary", []) or []:
            for activity in day.get("activities", []) or []:
                for name in ["西湖", "灵隐寺", "西溪国家湿地公园", "宋城"]:
                    if name in activity and name not in names:
                        names.append(name)
        return names

    def _extract_revision_attractions(self, query: str, attraction_names: List[str]) -> List[str]:
        """从修订请求中提取景点名"""
        return [name for name in sorted(attraction_names, key=len, reverse=True) if name in query]

    def _extract_revision_day(self, query: str) -> Optional[int]:
        """从修订请求中提取目标天数"""
        digit_match = re.search(r"第?(\d+)天", query)
        if digit_match:
            return int(digit_match.group(1))
        chinese_numbers = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}
        chinese_match = re.search(r"第?([一二两三四五])天", query)
        if chinese_match:
            return chinese_numbers.get(chinese_match.group(1))
        return None

    def _is_remove_revision(self, query: str) -> bool:
        """判断是否为移除景点请求"""
        return any(keyword in query for keyword in ["不要去", "不想去", "删掉", "删除", "移除"])

    def _is_replace_revision(self, query: str) -> bool:
        """判断是否为替换景点请求"""
        return any(keyword in query for keyword in ["换一个", "替换"])

    def _is_route_optimization_revision(self, query: str) -> bool:
        """判断是否为路线顺序优化请求"""
        return any(keyword in query for keyword in ["排一下顺序", "排序", "按距离", "按路线", "路线顺序", "优化路线", "优化一下"])

    def _is_weather_adjustment_revision(self, query: str) -> bool:
        """判断是否为天气感知行程调整请求"""
        return any(keyword in query for keyword in ["如果下雨", "下雨怎么办", "雨天", "避开雨天", "下雨的话", "降雨"])

    async def _apply_weather_adjustment_revision(self, query: str, itinerary: List[Dict], context: Dict) -> Dict:
        """根据天气预报调整行程"""
        destination = context.get("destination") or "杭州"
        weather_result = await self.tool_registry.execute(
            "get_weather_forecast",
            {"city": destination, "days": context.get("duration") or len(itinerary) or 3},
        )
        if not weather_result.get("success"):
            return {"success": False, "action": "weather_adjust", "message": weather_result.get("error") or "天气查询失败。"}

        weather_data = weather_result.get("data", {})
        forecasts = weather_data.get("forecasts", []) or []
        adjusted_days = []
        for index, forecast in enumerate(forecasts[:len(itinerary)], start=1):
            if forecast.get("suitable_for_outdoor") is False:
                day = self._find_day(itinerary, index)
                if not day:
                    continue
                self._replace_outdoor_with_rainy_activities(day, destination)
                adjusted_days.append({
                    "day": index,
                    "date": forecast.get("date"),
                    "weather": forecast.get("weather"),
                    "temperature": forecast.get("temperature"),
                    "advice": "不适合长时间户外活动，已优先调整为室内或低强度安排。",
                })

        if not adjusted_days:
            return {
                "success": True,
                "action": "weather_adjust",
                "summary": "天气整体适合户外游览，暂不需要大幅调整行程。",
                "weather_summary": {
                    "city": weather_data.get("city") or destination,
                    "adjusted_days": [],
                    "forecasts": forecasts,
                },
            }

        return {
            "success": True,
            "action": "weather_adjust",
            "summary": "已根据雨天天气将部分户外安排调整为室内或低强度活动。",
            "weather_summary": {
                "city": weather_data.get("city") or destination,
                "adjusted_days": adjusted_days,
                "forecasts": forecasts,
            },
        }

    def _replace_outdoor_with_rainy_activities(self, day: Dict, destination: str) -> None:
        """将雨天不适合户外的活动替换为室内或低强度安排"""
        rainy_replacements = self._rainy_day_replacements(destination)
        outdoor_keywords = ["西湖", "西溪", "湿地", "海边", "亚龙湾", "户外", "散步", "游览城市核心景区"]
        activities = day.get("activities", []) or []
        kept = [activity for activity in activities if not any(keyword in str(activity) for keyword in outdoor_keywords)]
        for replacement in rainy_replacements:
            if replacement not in kept:
                kept.append(replacement)
            if len(kept) >= max(len(activities), 3):
                break
        day["activities"] = kept
        note = day.get("notes", "")
        rain_note = " 已根据雨天天气调整为室内或低强度安排。"
        if rain_note.strip() not in note:
            day["notes"] = f"{note}{rain_note}".strip()

    def _rainy_day_replacements(self, destination: str) -> List[str]:
        """按城市返回雨天替代活动"""
        replacements = {
            "杭州": ["浙江省博物馆", "中国茶叶博物馆", "杭州特色美食体验", "湖滨银泰室内休闲"],
            "成都": ["四川博物院", "成都特色美食体验", "太古里室内休闲"],
            "厦门": ["厦门博物馆", "咖啡馆休闲", "闽南特色美食体验"],
            "三亚": ["酒店休闲", "海鲜美食体验", "室内免税店购物"],
        }
        return replacements.get(destination, ["城市博物馆", "当地美食体验", "室内休闲"])

    async def _apply_route_optimization_revision(
        self,
        query: str,
        itinerary: List[Dict],
        context: Dict,
        target_day: Optional[int],
    ) -> Dict:
        """按路线距离优化某天景点顺序"""
        day = self._select_route_optimization_day(query, itinerary, context, target_day)
        if not day:
            return {"success": False, "action": "route_optimize", "message": "当前行程中没有可优化路线的景点安排。"}

        places = self._places_for_day(day, context)
        if len(places) < 2:
            return {"success": False, "action": "route_optimize", "message": "该天至少需要两个带位置的景点才能优化顺序。"}

        route_result = await self.tool_registry.execute(
            "optimize_route_order",
            {
                "places": places,
                "start_place": places[0].get("name"),
                "mode": "walking",
            },
        )
        if not route_result.get("success"):
            return {"success": False, "action": "route_optimize", "message": route_result.get("error") or "路线优化失败。"}

        route_data = route_result.get("data", {})
        ordered_names = [place.get("name") for place in route_data.get("ordered_places", []) if place.get("name")]
        day["activities"] = self._replace_day_places(day.get("activities", []), ordered_names, places)
        return {
            "success": True,
            "action": "route_optimize",
            "summary": f"已按距离优化第{day.get('day')}天景点顺序。",
            "route_summary": {
                "day": day.get("day"),
                "ordered_places": ordered_names,
                "segments": route_data.get("segments", []),
                "total_distance": route_data.get("total_distance", 0),
                "total_duration": route_data.get("total_duration", 0),
            },
        }

    def _apply_move_revision(self, query: str, itinerary: List[Dict], target_names: List[str], target_day: int) -> Dict:
        """将景点移动到指定天数"""
        if not target_names:
            return {"success": False, "action": "move", "message": "没有识别到需要安排的景点名称。"}
        target_name = target_names[0]
        target = self._find_day(itinerary, target_day)
        if not target:
            return {"success": False, "action": "move", "message": f"当前行程中没有第{target_day}天。"}

        self._remove_activity_by_name(itinerary, target_name)
        self._append_activity(target, target_name)
        return {
            "success": True,
            "action": "move",
            "summary": f"已将{target_name}安排到第{target_day}天。",
        }

    def _apply_remove_revision(self, query: str, itinerary: List[Dict], target_names: List[str]) -> Dict:
        """从行程中移除景点"""
        if not target_names:
            return {"success": False, "action": "remove", "message": "没有识别到需要移除的景点名称。"}
        removed_names = []
        for name in target_names:
            if self._remove_activity_by_name(itinerary, name):
                removed_names.append(name)
        if not removed_names:
            return {"success": False, "action": "remove", "message": "当前行程中没有找到要移除的景点。"}
        return {
            "success": True,
            "action": "remove",
            "summary": f"已从行程中移除{'、'.join(removed_names)}。",
        }

    def _apply_replace_revision(self, query: str, itinerary: List[Dict], context: Dict, target_names: List[str]) -> Dict:
        """替换行程中的景点"""
        if not target_names:
            return {"success": False, "action": "replace", "message": "没有识别到需要替换的景点名称。"}
        removed_name = target_names[0]
        replacement = self._select_replacement_attraction(query, context, removed_name)
        if not replacement:
            return {"success": False, "action": "replace", "message": "暂时没有找到合适的替换景点。"}

        replaced = self._replace_activity_by_name(itinerary, removed_name, replacement)
        if not replaced:
            self._remove_activity_by_name(itinerary, removed_name)
            first_day = self._find_day(itinerary, 1)
            if first_day:
                self._append_activity(first_day, replacement)
        return {
            "success": True,
            "action": "replace",
            "summary": f"已移除{removed_name}，并替换为{replacement}。",
        }

    def _select_replacement_attraction(self, query: str, context: Dict, removed_name: str) -> Optional[str]:
        """根据偏好选择替换景点"""
        attractions = context.get("attractions", []) or []
        natural_keywords = ["自然风光", "风景名胜", "湿地", "西湖", "西溪"]
        for attraction in attractions:
            name = attraction.get("name")
            if not name or name == removed_name:
                continue
            category = attraction.get("category", "")
            description = attraction.get("description", "")
            if "自然风光" in query and not any(keyword in f"{name}{category}{description}" for keyword in natural_keywords):
                continue
            return name
        for name in self._known_attraction_names(context):
            if name != removed_name:
                return name
        return None

    def _select_route_optimization_day(
        self,
        query: str,
        itinerary: List[Dict],
        context: Dict,
        target_day: Optional[int],
    ) -> Optional[Dict]:
        """选择需要做路线优化的行程日"""
        if target_day:
            return self._find_day(itinerary, target_day)
        days_with_counts = [(day, len(self._places_for_day(day, context))) for day in itinerary]
        candidates = [(day, count) for day, count in days_with_counts if count >= 2]
        if not candidates:
            return None
        for day, count in candidates:
            if day.get("day") == 2:
                return day
        return max(candidates, key=lambda item: item[1])[0]

    def _places_for_day(self, day: Dict, context: Dict) -> List[Dict]:
        """从某天行程中提取可路线优化的景点"""
        known_places = self._known_places(context)
        places = []
        for activity in day.get("activities", []) or []:
            activity_text = str(activity)
            for place in known_places:
                name = place.get("name")
                if name and name in activity_text and name not in [item.get("name") for item in places]:
                    places.append(place)
        return places

    def _known_places(self, context: Dict) -> List[Dict]:
        """获取带名称和坐标的已知景点"""
        places = []
        for attraction in context.get("attractions", []) or []:
            name = attraction.get("name")
            if not name:
                continue
            places.append({
                "name": name,
                "location": attraction.get("location"),
                "category": attraction.get("category"),
            })
        return places

    def _replace_day_places(self, activities: List[str], ordered_names: List[str], places: List[Dict]) -> List[str]:
        """在保留非景点活动的同时替换某天景点顺序"""
        place_names = [place.get("name") for place in places]
        non_place_activities = [
            activity for activity in activities
            if not any(name and name in str(activity) for name in place_names)
        ]
        return [*non_place_activities, *ordered_names]

    def _find_day(self, itinerary: List[Dict], day_number: int) -> Optional[Dict]:
        """按天数查找行程日"""
        for day in itinerary:
            if day.get("day") == day_number:
                return day
        return None

    def _remove_activity_by_name(self, itinerary: List[Dict], attraction_name: str) -> bool:
        """从所有天的活动中移除包含指定景点名的活动"""
        removed = False
        for day in itinerary:
            activities = day.get("activities", []) or []
            kept = [activity for activity in activities if attraction_name not in str(activity)]
            if len(kept) != len(activities):
                removed = True
            day["activities"] = kept
        return removed

    def _replace_activity_by_name(self, itinerary: List[Dict], old_name: str, new_name: str) -> bool:
        """将包含旧景点名的活动替换为新景点名"""
        for day in itinerary:
            activities = day.get("activities", []) or []
            for index, activity in enumerate(activities):
                if old_name in str(activity):
                    activities[index] = new_name
                    return True
        return False

    def _append_activity(self, day: Dict, activity_name: str) -> None:
        """向某天添加活动并避免重复"""
        activities = day.setdefault("activities", [])
        if not any(activity_name in str(activity) for activity in activities):
            activities.append(activity_name)

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
        result = await self.arun_with_artifacts(message, session_id)
        return result["response"]

    async def arun_with_artifacts(self, message: str, session_id: str) -> Dict[str, Any]:
        """异步运行Agent，并返回可供前端展示的结构化结果"""
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

        response = "处理完成"
        if result.get("messages"):
            response = result["messages"][-1].content

        return {
            "response": response,
            "artifacts": self._build_response_artifacts(result.get("task_results", [])),
        }

    def _build_response_artifacts(self, task_results: List[Dict]) -> Dict[str, Any]:
        """从任务结果中提取前端可视化展示数据"""
        artifacts: Dict[str, Any] = {}

        for task_result in task_results or []:
            task = task_result.get("task", {}) or {}
            task_type = task.get("task_type")
            tool_name = task.get("tool")
            data = self._extract_result_data(task_result)
            if not isinstance(data, dict):
                continue

            if tool_name == "generate_itinerary" and data.get("itinerary"):
                artifacts["itinerary"] = self._build_itinerary_artifact(data, title="每日行程")

            if tool_name == "search_attractions" and data.get("attractions"):
                artifacts["attractions"] = {
                    "location": data.get("location"),
                    "items": data.get("attractions", [])[:4],
                    "sources": data.get("rag_documents", [])[:3],
                }

            if tool_name == "get_weather_forecast" and data.get("forecasts"):
                artifacts["weather"] = {
                    "city": data.get("city"),
                    "forecasts": data.get("forecasts", []),
                    "travel_advice": data.get("travel_advice", []),
                }

            if tool_name == "optimize_route_order" and data.get("segments"):
                artifacts["route"] = self._build_route_artifact(data)

            if task_type == "revise_itinerary":
                if data.get("itinerary"):
                    artifacts["itinerary"] = self._build_itinerary_artifact(data, title="调整后的每日行程")
                if data.get("route_summary"):
                    artifacts["route"] = self._build_route_artifact(data.get("route_summary", {}))
                if data.get("weather_summary"):
                    artifacts["weather_adjustment"] = data.get("weather_summary")

        return normalize_chat_artifacts(artifacts)

    def _build_itinerary_artifact(self, data: Dict[str, Any], title: str) -> Dict[str, Any]:
        """构建行程展示数据"""
        return {
            "title": title,
            "origin": data.get("origin"),
            "destination": data.get("destination"),
            "duration": data.get("duration"),
            "budget": data.get("budget"),
            "summary": data.get("summary"),
            "days": copy.deepcopy(data.get("itinerary", [])),
            "budget_summary": copy.deepcopy(data.get("budget_summary", {})),
        }

    def _build_route_artifact(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """构建路线展示数据"""
        return {
            "day": data.get("day"),
            "ordered_places": copy.deepcopy(data.get("ordered_places", [])),
            "segments": copy.deepcopy(data.get("segments", [])),
            "total_distance": data.get("total_distance", 0),
            "total_duration": data.get("total_duration", 0),
            "mode": data.get("mode"),
        }

    async def astream(self, message: str, session_id: str) -> AsyncGenerator[str, None]:
        """流式运行Agent"""
        response = await self.arun(message, session_id)
        yield response

    def get_history(self, session_id: str) -> List[Dict]:
        """获取对话历史"""
        return self.memory_manager.get_history(session_id)

    def clear_history(self, session_id: str):
        """清除对话历史、会话动态RAG文档和行程上下文"""
        self.memory_manager.clear_history(session_id)
        key = session_id or "default"
        self.dynamic_rag_stores.pop(key, None)
        self.session_itinerary_contexts.pop(key, None)
