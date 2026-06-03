"""
任务规划器
根据用户意图规划执行任务
"""
from typing import Dict, List
from models.intent import TravelIntent


class TaskPlanner:
    """任务规划器"""

    def __init__(self):
        """初始化规划器"""
        # 任务模板
        self.task_templates = {
            "travel_plan": [
                {"tool": "search_flights", "priority": 1},
                {"tool": "search_hotels", "priority": 2},
                {"tool": "search_attractions", "priority": 3},
                {"tool": "generate_itinerary", "priority": 4}
            ],
            "flight_search": [
                {"tool": "search_flights", "priority": 1}
            ],
            "hotel_search": [
                {"tool": "search_hotels", "priority": 1}
            ],
            "attraction_search": [
                {"tool": "search_attractions", "priority": 1}
            ],
            "policy_query": [
                {"tool": "retrieve_policy", "priority": 1}
            ]
        }

    def plan(self, intent: Dict, context: Dict) -> List[Dict]:
        """
        根据意图规划任务

        Args:
            intent: 意图信息
            context: 上下文信息（RAG、记忆等）

        Returns:
            任务列表
        """
        intent_type = intent.get("intent", "general_chat")
        entities = intent.get("entities", {})

        # 获取任务模板
        template = self.task_templates.get(intent_type, [])

        # 填充任务参数
        tasks = []
        for task_template in template:
            task = {
                "tool": task_template["tool"],
                "priority": task_template["priority"],
                "params": self._build_params(task_template["tool"], entities, context)
            }
            tasks.append(task)

        return tasks

    def _build_params(self, tool_name: str, entities: Dict, context: Dict) -> Dict:
        """构建工具参数"""
        params = {}

        if tool_name == "search_flights":
            params = {
                "origin": entities.get("origin"),
                "destination": entities.get("destination"),
                "date": entities.get("departure_date")
            }

        elif tool_name == "search_hotels":
            params = {
                "location": entities.get("destination"),
                "checkin_date": entities.get("departure_date"),
                "checkout_date": entities.get("return_date")
            }

        elif tool_name == "search_attractions":
            params = {
                "location": entities.get("destination"),
                "keywords": entities.get("preferences", [])
            }

        elif tool_name == "generate_itinerary":
            params = {
                "destination": entities.get("destination"),
                "duration": entities.get("duration", 3),
                "preferences": entities.get("preferences", [])
            }

        elif tool_name == "retrieve_policy":
            # 从上下文中获取查询内容
            params = {
                "query": context.get("query", "")
            }

        return params
