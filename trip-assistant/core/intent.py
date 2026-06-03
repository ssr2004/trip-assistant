"""
意图识别模块
解析用户输入的自然语言，提取意图和实体
"""
from typing import Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.config import settings
from models.intent import Intent, TravelIntent


class IntentParser:
    """意图解析器"""

    def __init__(self):
        """初始化意图解析器"""
        self.parser = JsonOutputParser(pydantic_object=TravelIntent)
        self.prompt = self._create_prompt()

    def _create_prompt(self) -> ChatPromptTemplate:
        """创建提示模板"""
        return ChatPromptTemplate.from_messages([
            ("system", """你是一个旅行助手的意图识别模块。
请分析用户输入，提取以下信息：

1. intent: 意图类型
   - travel_plan: 旅行规划（需要航班+酒店+行程）
   - flight_search: 航班搜索
   - hotel_search: 酒店搜索
   - attraction_search: 景点搜索
   - policy_query: 政策查询
   - general_chat: 闲聊

2. entities: 实体信息
   - origin: 出发地
   - destination: 目的地
   - departure_date: 出发日期
   - return_date: 返回日期
   - duration: 旅行天数
   - budget: 预算
   - preferences: 偏好（如经济舱、靠窗等）

3. confidence: 置信度（0-1）

{format_instructions}"""),
            ("user", "{input}")
        ])

    def parse(self, user_input: str) -> Dict:
        """
        解析用户输入

        Args:
            user_input: 用户输入的文本

        Returns:
            解析后的意图字典
        """
        # 简单的规则匹配（后续可替换为LLM）
        intent = self._rule_based_parse(user_input)

        # 如果规则匹配失败，使用LLM
        if intent["confidence"] < 0.5:
            intent = self._llm_parse(user_input)

        return intent

    def _rule_based_parse(self, text: str) -> Dict:
        """基于规则的意图识别"""
        text = text.lower()

        # 关键词映射
        keywords = {
            "travel_plan": ["去", "旅游", "旅行", "玩", "行程", "计划"],
            "flight_search": ["航班", "飞机", "机票", "飞"],
            "hotel_search": ["酒店", "住宿", "宾馆", "住"],
            "attraction_search": ["景点", "玩", "观光", "游览"],
            "policy_query": ["退票", "改签", "政策", "规定", "能退吗"]
        }

        # 检测意图
        detected_intent = "general_chat"
        max_score = 0

        for intent_type, kws in keywords.items():
            score = sum(1 for kw in kws if kw in text)
            if score > max_score:
                max_score = score
                detected_intent = intent_type

        # 提取实体
        entities = self._extract_entities(text)

        return {
            "intent": detected_intent,
            "entities": entities,
            "confidence": min(0.3 + max_score * 0.2, 0.9)
        }

    def _extract_entities(self, text: str) -> Dict:
        """提取实体"""
        import re

        entities = {
            "origin": None,
            "destination": None,
            "departure_date": None,
            "return_date": None,
            "duration": None,
            "budget": None,
            "preferences": []
        }

        # 城市名称匹配（简化版）
        cities = ["北京", "上海", "广州", "深圳", "成都", "杭州", "西安", "重庆",
                  "郑州", "武汉", "长沙", "南京", "苏州", "厦门", "青岛", "大连",
                  "三亚", "昆明", "大理", "丽江", "桂林", "贵阳", "哈尔滨"]

        found_cities = [city for city in cities if city in text]
        if len(found_cities) >= 2:
            entities["origin"] = found_cities[0]
            entities["destination"] = found_cities[1]
        elif len(found_cities) == 1:
            entities["destination"] = found_cities[0]

        # 日期匹配
        date_pattern = r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)'
        dates = re.findall(date_pattern, text)
        if dates:
            entities["departure_date"] = dates[0]
        if len(dates) > 1:
            entities["return_date"] = dates[1]

        # 天数匹配
        duration_pattern = r'(\d+)\s*[天日]'
        duration_match = re.search(duration_pattern, text)
        if duration_match:
            entities["duration"] = int(duration_match.group(1))

        return entities

    def _llm_parse(self, user_input: str) -> Dict:
        """使用LLM进行意图识别"""
        # 这里需要调用LLM
        # 暂时返回默认值
        return {
            "intent": "general_chat",
            "entities": {
                "origin": None,
                "destination": None,
                "departure_date": None,
                "return_date": None,
                "duration": None,
                "budget": None,
                "preferences": []
            },
            "confidence": 0.3
        }
