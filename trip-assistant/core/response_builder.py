"""
Agent响应生成模块
将意图、任务和工具结果聚合为面向用户的结构化中文回复
"""
from typing import Any, Dict, List, Optional


class ResponseBuilder:
    """旅行助手响应构建器"""

    def build(self, intent: Dict, task_results: List[Dict], memory_context: Dict = None) -> str:
        """
        构建最终回复

        Args:
            intent: 结构化意图
            task_results: 任务执行结果
            memory_context: 记忆上下文，可包含长期用户偏好

        Returns:
            面向用户的中文回复
        """
        memory_context = memory_context or {}
        if not task_results:
            return "我已经理解您的需求，但当前还缺少可执行的旅行任务。您可以补充出发地、目的地、日期或预算等信息。"

        ask_user_result = self._find_result_by_task_type(task_results, "ask_user")
        if ask_user_result:
            return self._format_followup(ask_user_result)

        destination_result = self._find_result_by_task_type(task_results, "recommend_destination")
        if destination_result:
            return self._format_destination_recommendations(destination_result.get("result", {}))

        intent_type = (intent or {}).get("intent")
        if intent_type == "travel_plan":
            return self._format_travel_plan(intent or {}, task_results, memory_context)
        if intent_type == "policy_query":
            return self._format_policy_response(task_results)
        if intent_type == "guide_query":
            return self._format_guide_query_response(task_results)
        if intent_type == "dynamic_knowledge_query":
            return self._format_dynamic_knowledge_response(task_results)
        if intent_type == "itinerary_revision":
            return self._format_itinerary_revision_response(task_results)
        if intent_type == "weather_query":
            return self._format_weather_response(task_results)
        if intent_type == "flight_search":
            return self._format_single_tool_response("航班推荐", task_results, "search_flights")
        if intent_type == "hotel_search":
            return self._format_single_tool_response("酒店推荐", task_results, "search_hotels")
        if intent_type == "attraction_search":
            return self._format_single_tool_response("景点推荐", task_results, "search_attractions")

        return self._format_generic_response(task_results)

    def _format_followup(self, result: Dict) -> str:
        """格式化追问回复"""
        data = result.get("result", {}) or {}
        question = data.get("question") or "请补充一下旅行需求中的关键信息。"
        missing_slots = data.get("missing_slots", [])

        lines = [question]
        if missing_slots:
            readable_slots = [self._slot_label(slot) for slot in missing_slots]
            lines.append(f"\n需要补充的信息：{'、'.join(readable_slots)}。")
        return "\n".join(lines)

    def _format_destination_recommendations(self, result: Dict) -> str:
        """格式化目的地推荐结果"""
        lines = [result.get("message", "根据您的预算、时间和偏好，我先为您推荐以下候选目的地：")]
        budget = result.get("budget")
        preferences = result.get("preferences") or []

        if budget or preferences:
            overview = []
            if budget:
                overview.append(f"预算约{budget:g}元")
            if preferences:
                overview.append(f"偏好：{'、'.join(preferences)}")
            lines.append(f"\n参考条件：{'；'.join(overview)}。")

        for index, candidate in enumerate(result.get("candidates", []), start=1):
            city = candidate.get("city", "候选目的地")
            reason = candidate.get("reason", "适合当前旅行需求。")
            lines.append(f"{index}. {city}：{reason}")

        lines.append("\n您可以选择其中一个城市，我再继续为您规划交通、酒店和每日行程。")
        return "\n".join(lines)

    def _format_travel_plan(self, intent: Dict, task_results: List[Dict], memory_context: Dict = None) -> str:
        """格式化完整旅行规划回复"""
        memory_context = memory_context or {}
        entities = intent.get("entities", {}) or {}
        itinerary_result = self._find_result_by_tool(task_results, "generate_itinerary")
        itinerary_data = self._result_data(itinerary_result) if itinerary_result else {}

        origin = itinerary_data.get("origin") or entities.get("origin") or "出发地"
        destination = itinerary_data.get("destination") or entities.get("destination") or "目的地"
        duration = itinerary_data.get("duration") or entities.get("duration") or 3
        budget = itinerary_data.get("budget") or entities.get("budget")
        travelers = itinerary_data.get("travelers") or entities.get("travelers")
        preferences = itinerary_data.get("preferences") or entities.get("preferences") or []
        departure_date = entities.get("departure_date")

        lines = [f"已为您规划{origin}到{destination}的{duration}天旅行方案。"]
        memory_note = self._format_memory_preference_note(memory_context)
        lines.extend(self._format_overview(
            origin,
            destination,
            departure_date,
            duration,
            budget,
            travelers,
            preferences,
            memory_note,
        ))

        flights_result = self._find_result_by_tool(task_results, "search_flights")
        flights = self._tool_items(flights_result, "flights") if flights_result else []
        if flights:
            lines.extend(self._format_flights(flights, section_title="二、航班推荐"))

        hotels_result = self._find_result_by_tool(task_results, "search_hotels")
        hotels = self._tool_items(hotels_result, "hotels") if hotels_result else []
        if hotels:
            lines.extend(self._format_hotels(hotels, section_title="三、酒店推荐"))

        attractions_result = self._find_result_by_tool(task_results, "search_attractions")
        attractions = self._tool_items(attractions_result, "attractions") if attractions_result else []
        attraction_sources = self._tool_items(attractions_result, "rag_documents") if attractions_result else []
        if attractions:
            lines.extend(self._format_attractions(attractions, section_title="四、景点推荐", sources=attraction_sources))

        itinerary = itinerary_data.get("itinerary", [])
        if itinerary:
            lines.extend(self._format_itinerary(itinerary, section_title="五、每日行程"))

        budget_summary = itinerary_data.get("budget_summary")
        if budget_summary:
            lines.extend(self._format_budget(budget_summary, section_title="六、预算估算"))

        guide_result = self._find_result_by_tool(task_results, "retrieve_guide")
        guide_data = self._result_data(guide_result) if guide_result else {}
        if guide_data:
            lines.extend(self._format_guide(guide_data, section_title="七、攻略与注意事项"))

        error_lines = self._format_errors(task_results)
        if error_lines:
            lines.extend(["", "八、执行提示", *error_lines])

        lines.append("\n以上方案基于当前本地模拟数据和攻略文档生成，后续接入真实API和LLM后，可进一步给出实时价格、天气、路线和更个性化的安排。")
        return "\n".join(lines)

    def _format_policy_response(self, task_results: List[Dict]) -> str:
        """格式化政策查询回复"""
        policy_result = self._find_result_by_tool(task_results, "retrieve_policy")
        data = self._result_data(policy_result) if policy_result else {}
        answer = data.get("answer")
        sources = data.get("sources") or []

        lines = ["关于您的政策问题，我查到以下信息："]
        lines.append(answer or "暂未检索到明确政策答案，建议补充更具体的问题。")
        if sources:
            lines.append("\n资料来源：")
            for source in sources[:3]:
                lines.append(self._format_source_reference(source, "本地政策文档"))
        return "\n".join(lines)

    def _format_guide_query_response(self, task_results: List[Dict]) -> str:
        """格式化攻略知识查询回复"""
        guide_result = self._find_result_by_tool(task_results, "retrieve_guide")
        data = self._result_data(guide_result) if guide_result else {}
        answer = data.get("answer")
        sources = data.get("sources") or []

        lines = ["关于您的旅行攻略问题，我查到以下信息："]
        lines.append(answer or "暂未检索到明确攻略答案，建议补充更具体的目的地或玩法需求。")
        if sources:
            lines.append("\n资料来源：")
            for source in sources[:3]:
                lines.append(self._format_source_reference(source, "本地攻略文档"))
        return "\n".join(lines)

    def _format_dynamic_knowledge_response(self, task_results: List[Dict]) -> str:
        """格式化动态外部知识追问回复"""
        result = self._find_result_by_task_type(task_results, "dynamic_rag_query")
        data = result.get("result", {}) if result else {}
        answer = data.get("answer") or "我暂时没有在本轮对话的外部景点数据中找到相关信息。"
        sources = data.get("sources") or []

        lines = [answer]
        if sources:
            lines.append("\n资料来源：")
            for source in sources[:3]:
                lines.append(self._format_source_reference(source, "外部景点数据"))
        return "\n".join(lines)

    def _format_itinerary_revision_response(self, task_results: List[Dict]) -> str:
        """格式化行程修订回复"""
        result = self._find_result_by_task_type(task_results, "revise_itinerary")
        data = result.get("result", {}) if result else {}
        if not data.get("success"):
            return data.get("message") or "我暂时还没有可调整的历史行程。您可以先让我生成一个完整旅行计划。"

        lines = [f"已根据您的要求调整行程：{data.get('summary', '行程已更新。')}"]
        itinerary = data.get("itinerary") or []
        if itinerary:
            lines.extend(self._format_itinerary(itinerary, section_title="调整后的每日行程"))

        route_summary = data.get("route_summary")
        if route_summary:
            lines.extend(self._format_route_summary(route_summary))

        weather_summary = data.get("weather_summary")
        if weather_summary:
            lines.extend(self._format_weather_adjustment_summary(weather_summary))

        sources = data.get("sources") or []
        if sources:
            lines.append("资料依据：")
            for source in sources[:3]:
                lines.append(f"- {source}")
        return "\n".join(lines)

    def _format_route_summary(self, route_summary: Dict) -> List[str]:
        """格式化路线优化摘要"""
        lines = ["", "路线优化摘要："]
        for segment in (route_summary.get("segments") or [])[:5]:
            distance_km = self._format_distance(segment.get("distance", 0))
            duration_text = self._format_duration(segment.get("duration", 0))
            lines.append(f"- {segment.get('from')} → {segment.get('to')}：约{distance_km}，约{duration_text}")
        total_distance = self._format_distance(route_summary.get("total_distance", 0))
        total_duration = self._format_duration(route_summary.get("total_duration", 0))
        lines.append(f"- 总距离：约{total_distance}，预计用时约{total_duration}")
        return lines

    def _format_weather_adjustment_summary(self, weather_summary: Dict) -> List[str]:
        """格式化天气感知行程调整摘要"""
        lines = ["", "天气调整依据："]
        adjusted_days = weather_summary.get("adjusted_days") or []
        if adjusted_days:
            for item in adjusted_days[:5]:
                lines.append(
                    f"- 第{item.get('day')}天：{item.get('weather')}，{item.get('temperature', '温度待定')}，{item.get('advice')}"
                )
            lines.append("- 已将部分户外安排替换为室内或低强度活动。")
        else:
            lines.append("- 当前天气整体适合户外游览，暂不需要大幅调整。")
        return lines

    def _format_weather_response(self, task_results: List[Dict]) -> str:
        """格式化天气查询回复"""
        result = self._find_result_by_tool(task_results, "get_weather_forecast")
        data = self._result_data(result) if result else {}
        if not data:
            return "天气查询暂未成功，请补充城市或稍后再试。"

        city = data.get("city") or "目的地"
        forecasts = data.get("forecasts") or []
        advice = data.get("travel_advice") or []
        lines = [f"{city}未来{len(forecasts)}天天气："]
        for index, forecast in enumerate(forecasts, start=1):
            lines.append(
                f"{index}. {forecast.get('date', '日期待定')}：{forecast.get('weather', '天气待定')}，"
                f"{forecast.get('temperature', '温度待定')}，{forecast.get('wind', '风力待定')}。"
            )
            if index <= len(advice):
                lines.append(f"   建议：{advice[index - 1]}")
        return "\n".join(lines)

    def _format_single_tool_response(self, title: str, task_results: List[Dict], tool_name: str) -> str:
        """格式化单工具查询回复"""
        result = self._find_result_by_tool(task_results, tool_name)

        if not result or not result.get("success"):
            error = result.get("error") if result else "未能获取工具结果"
            return f"{title}查询暂未成功：{error}。"

        if tool_name == "search_flights":
            flights = self._tool_items(result, "flights")
            return "\n".join([f"为您找到以下{title}：", *self._format_flights(flights, section_title=None)])
        if tool_name == "search_hotels":
            hotels = self._tool_items(result, "hotels")
            return "\n".join([f"为您找到以下{title}：", *self._format_hotels(hotels, section_title=None)])
        if tool_name == "search_attractions":
            attractions = self._tool_items(result, "attractions")
            sources = self._tool_items(result, "rag_documents")
            return "\n".join([f"为您找到以下{title}：", *self._format_attractions(attractions, section_title=None, sources=sources)])

        return self._format_generic_response(task_results)

    def _format_generic_response(self, task_results: List[Dict]) -> str:
        """兜底格式化回复"""
        lines = ["已完成当前旅行任务："]
        for result in task_results:
            task = result.get("task", {})
            task_name = task.get("name") or task.get("tool") or "旅行任务"
            if result.get("success"):
                lines.append(f"- {task_name}：已完成。")
            else:
                lines.append(f"- {task_name}：执行失败，原因是{result.get('error')}。")
        return "\n".join(lines)

    def _format_source_reference(self, source: Dict, default_source: str) -> str:
        """格式化结构化RAG来源，优先展示标题、章节和来源路径"""
        title = source.get("title")
        section = source.get("section")
        source_name = source.get("source") or default_source
        if title and section and section != title:
            return f"- {title} / {section}：{source_name}"
        if title:
            return f"- {title}：{source_name}"
        return f"- {source_name}"

    def _format_memory_preference_note(self, memory_context: Dict) -> Optional[str]:
        """格式化长期记忆偏好说明，仅用于完整旅行规划"""
        preferences = (memory_context or {}).get("preferences") or (memory_context or {}).get("user_preferences") or {}
        if not isinstance(preferences, dict):
            return None

        display_preferences = []
        for field in [
            "travel_styles",
            "hotel_preferences",
            "transport_preferences",
            "attraction_preferences",
            "food_preferences",
            "raw_preferences",
        ]:
            values = preferences.get(field, [])
            if not isinstance(values, list):
                continue
            for value in values:
                label = self._memory_preference_label(value)
                if label and label not in display_preferences:
                    display_preferences.append(label)

        budget_preference = preferences.get("budget_preference")
        if budget_preference:
            label = self._memory_preference_label(budget_preference)
            if label and label not in display_preferences:
                display_preferences.append(label)

        if not display_preferences:
            return None
        return f"已结合您偏好的{'、'.join(display_preferences[:4])}进行安排。"

    def _memory_preference_label(self, value: Any) -> Optional[str]:
        """将记忆偏好标签转为更自然的展示文本"""
        if not value:
            return None
        labels = {
            "地铁附近": "地铁附近住宿",
            "交通方便": "交通便利住宿",
            "经济型酒店": "经济型住宿",
            "高星级酒店": "高星级住宿",
            "经济型": "经济预算",
            "中档": "中档预算",
            "舒适型": "舒适体验",
            "豪华型": "高端体验",
            "自然风光": "自然风光体验",
            "人文历史": "人文历史体验",
            "当地美食": "当地美食体验",
        }
        return labels.get(str(value), str(value))

    def _format_overview(
        self,
        origin: str,
        destination: str,
        departure_date: Optional[str],
        duration: int,
        budget: Optional[float],
        travelers: Optional[int],
        preferences: List[str],
        memory_note: Optional[str] = None,
    ) -> List[str]:
        """格式化出行概览"""
        lines = ["", "一、出行概览"]
        lines.append(f"- 路线：{origin} → {destination}")
        if departure_date:
            lines.append(f"- 出发日期：{departure_date}")
        lines.append(f"- 行程天数：{duration}天")
        if travelers:
            lines.append(f"- 出行人数：{travelers}人")
        if budget:
            lines.append(f"- 预算：约{budget:g}元")
        if preferences:
            lines.append(f"- 偏好：{'、'.join(preferences)}")
        if memory_note:
            lines.append(f"- 个性化参考：{memory_note}")
        return lines

    def _format_flights(self, flights: List[Dict], section_title: Optional[str]) -> List[str]:
        """格式化航班列表"""
        lines = ["", section_title] if section_title else []
        for index, flight in enumerate((flights or [])[:3], start=1):
            departure_time = self._clean_time(flight.get("departure_time"))
            arrival_time = self._clean_time(flight.get("arrival_time"))
            price = flight.get("price")
            price_text = f"，约{price:g}元" if isinstance(price, (int, float)) else ""
            lines.append(
                f"{index}. {flight.get('airline', '航空公司')} {flight.get('flight_no', '')}："
                f"{flight.get('departure_airport', '出发机场')} → {flight.get('arrival_airport', '到达机场')}，"
                f"{departure_time} - {arrival_time}，{flight.get('cabin_class', '舱位待定')}{price_text}。"
            )
        return lines

    def _format_hotels(self, hotels: List[Dict], section_title: Optional[str]) -> List[str]:
        """格式化酒店列表"""
        lines = ["", section_title] if section_title else []
        for index, hotel in enumerate((hotels or [])[:3], start=1):
            amenities = hotel.get("amenities") or []
            amenities_text = f"，设施：{'、'.join(amenities[:3])}" if amenities else ""
            lines.append(
                f"{index}. {hotel.get('name', '酒店')}：{hotel.get('address', hotel.get('location', '地址待定'))}，"
                f"约{hotel.get('price_per_night', '价格待定')}元/晚，评分{hotel.get('rating', '暂无')}{amenities_text}。"
            )
        return lines

    def _format_attractions(
        self,
        attractions: List[Dict],
        section_title: Optional[str],
        sources: Optional[List[Dict]] = None,
    ) -> List[str]:
        """格式化景点列表，并展示外部POI的RAG来源"""
        lines = ["", section_title] if section_title else []
        for index, attraction in enumerate((attractions or [])[:4], start=1):
            lines.append(
                f"{index}. {attraction.get('name', '景点')}：{attraction.get('category', '类型待定')}，"
                f"评分{attraction.get('rating', '暂无')}，门票{attraction.get('ticket_price', '待定')}。"
                f"{attraction.get('description', '')}"
            )
        if sources:
            lines.append("资料来源：")
            for source in sources[:3]:
                lines.append(self._format_source_reference(source, "外部景点数据"))
        return lines

    def _format_itinerary(self, itinerary: List[Dict], section_title: str) -> List[str]:
        """格式化每日行程"""
        lines = ["", section_title]
        for day in itinerary:
            activities = day.get("activities") or []
            lines.append(f"Day {day.get('day')}：{day.get('title', '行程安排')}")
            if activities:
                lines.append(f"- 安排：{' → '.join(activities)}")
            if day.get("notes"):
                lines.append(f"- 说明：{day['notes']}")
        return lines

    def _format_budget(self, budget_summary: Dict, section_title: str) -> List[str]:
        """格式化预算估算"""
        lines = ["", section_title]
        input_budget = budget_summary.get("input_budget")
        if input_budget:
            lines.append(f"- 用户预算：约{input_budget:g}元")
        lines.append(f"- 住宿估算：约{budget_summary.get('estimated_hotel', 0):g}元")
        lines.append(f"- 餐饮估算：约{budget_summary.get('estimated_food', 0):g}元")
        lines.append(f"- 市内交通估算：约{budget_summary.get('estimated_local_transport', 0):g}元")
        lines.append(f"- 景点门票估算：约{budget_summary.get('estimated_attractions', 0):g}元")
        lines.append(f"- 不含机票合计：约{budget_summary.get('estimated_total_without_flight', 0):g}元")
        if budget_summary.get("budget_note"):
            lines.append(f"- 预算建议：{budget_summary['budget_note']}")
        return lines

    def _format_guide(self, guide_data: Dict, section_title: str) -> List[str]:
        """格式化攻略信息"""
        lines = ["", section_title]
        answer = guide_data.get("answer")
        if answer:
            lines.append(self._compact_multiline(answer, max_lines=8))

        sources = guide_data.get("sources") or []
        if sources:
            lines.append("资料来源：")
            for source in sources[:3]:
                lines.append(self._format_source_reference(source, "本地攻略文档"))
        return lines

    def _format_errors(self, task_results: List[Dict]) -> List[str]:
        """格式化失败任务"""
        lines = []
        for result in task_results:
            if result.get("success"):
                raw_result = result.get("result")
                if isinstance(raw_result, dict) and raw_result.get("success") is False:
                    task = result.get("task", {})
                    lines.append(f"- {task.get('name', '任务')}：{raw_result.get('error', '工具返回失败')}。")
                continue
            task = result.get("task", {})
            lines.append(f"- {task.get('name', '任务')}：{result.get('error', '执行失败')}。")
        return lines

    def _find_result_by_task_type(self, task_results: List[Dict], task_type: str) -> Optional[Dict]:
        """按任务类型查找结果"""
        for result in task_results:
            task = result.get("task", {})
            if task.get("task_type") == task_type and result.get("success"):
                return result
        return None

    def _find_result_by_tool(self, task_results: List[Dict], tool_name: str) -> Optional[Dict]:
        """按工具名称查找结果"""
        for result in task_results:
            task = result.get("task", {})
            if task.get("tool") == tool_name:
                return result
        return None

    def _result_data(self, result: Optional[Dict]) -> Any:
        """获取工具业务数据，兼容旧列表返回和新标准返回"""
        if not result or not result.get("success"):
            return []

        raw_result = result.get("result")
        if isinstance(raw_result, dict) and "data" in raw_result:
            return raw_result.get("data") or {}
        return raw_result

    def _tool_items(self, result: Optional[Dict], data_key: str) -> List[Dict]:
        """获取列表型工具数据，兼容标准结构和旧列表结构"""
        data = self._result_data(result)
        if isinstance(data, dict):
            items = data.get(data_key, [])
            return items if isinstance(items, list) else []
        if isinstance(data, list):
            return data
        return []

    def _slot_label(self, slot: str) -> str:
        """槽位中文名"""
        labels = {
            "origin": "出发地",
            "destination": "目的地",
            "departure_date": "出发日期",
            "duration": "旅行天数",
            "budget": "预算",
            "travelers": "出行人数",
        }
        return labels.get(slot, slot)

    def _clean_time(self, value: Any) -> str:
        """清理模拟数据中的空日期展示"""
        if value is None:
            return "时间待定"
        text = str(value)
        return text.replace("None ", "").strip() or "时间待定"

    def _format_distance(self, distance: Any) -> str:
        """格式化米制距离"""
        try:
            meters = float(distance)
        except (TypeError, ValueError):
            meters = 0
        if meters >= 1000:
            return f"{meters / 1000:.1f}公里"
        return f"{meters:.0f}米"

    def _format_duration(self, duration: Any) -> str:
        """格式化秒制耗时"""
        try:
            seconds = int(duration)
        except (TypeError, ValueError):
            seconds = 0
        minutes = max(round(seconds / 60), 1) if seconds else 0
        return f"{minutes}分钟"

    def _compact_multiline(self, text: str, max_lines: int) -> str:
        """压缩多行文本，避免攻略段落过长"""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) <= max_lines:
            return "\n".join(lines)
        return "\n".join(lines[:max_lines] + ["……"])
