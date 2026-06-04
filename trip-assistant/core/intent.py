"""
意图识别模块
解析用户输入的自然语言，提取意图、实体和缺失槽位
"""
from datetime import date, timedelta
import json
import re
from typing import Dict, List, Optional

from core.llm import LLMClient, LLMMessage, LLMRequest
from core.llm.prompts import INTENT_FALLBACK_SYSTEM_PROMPT
from models.intent import TravelIntent


class IntentParser:
    """意图解析器"""

    FALLBACK_CONFIDENCE_THRESHOLD = 0.55

    CITIES = [
        "北京", "上海", "广州", "深圳", "成都", "杭州", "西安", "重庆",
        "郑州", "武汉", "长沙", "南京", "苏州", "厦门", "青岛", "大连",
        "三亚", "昆明", "大理", "丽江", "桂林", "贵阳", "哈尔滨",
        "天津", "沈阳", "济南", "福州", "南昌", "合肥", "宁波", "无锡",
    ]

    CHINESE_NUMBERS = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    REQUIRED_SLOTS = {
        "travel_plan": ["origin", "destination", "departure_date", "duration"],
        "flight_search": ["origin", "destination"],
        "hotel_search": ["destination"],
        "attraction_search": ["destination"],
        "policy_query": [],
        "guide_query": [],
        "dynamic_knowledge_query": [],
        "itinerary_revision": [],
        "general_chat": [],
    }

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """初始化意图解析器"""
        self.llm_client = llm_client or LLMClient()

    def parse(self, user_input: str) -> Dict:
        """
        解析用户输入

        Args:
            user_input: 用户输入的文本

        Returns:
            解析后的意图字典
        """
        normalized_text = self._normalize_text(user_input)
        intent = self._detect_intent(normalized_text)
        entities = self._extract_entities(normalized_text, intent)
        missing_slots = self._detect_missing_slots(intent, entities)
        followup_question = self._build_followup_question(missing_slots, entities)

        parsed = TravelIntent(
            intent=intent,
            entities=entities,
            confidence=self._calculate_confidence(intent, entities, missing_slots),
            missing_slots=missing_slots,
            followup_question=followup_question,
        )
        return parsed.model_dump()

    async def parse_async(self, user_input: str) -> Dict:
        """
        异步解析用户输入，规则优先，必要时尝试LLM fallback

        Args:
            user_input: 用户输入的文本

        Returns:
            解析后的意图字典
        """
        rule_result = self.parse(user_input)
        if not self._should_use_llm_fallback(rule_result):
            return rule_result

        llm_result = await self._llm_fallback(user_input, rule_result)
        return llm_result or rule_result

    def _normalize_text(self, text: str) -> str:
        """标准化输入文本"""
        return text.strip().replace("，", ",").replace("。", ".")

    def _detect_intent(self, text: str) -> str:
        """基于规则识别旅行意图"""
        if any(keyword in text for keyword in [
            "退票", "改签", "取消政策", "退改", "政策", "规定", "能退吗", "能取消吗", "可以取消吗",
            "门票可以退", "门票能退", "退款", "退房", "航班延误",
        ]):
            return "policy_query"

        if self._is_itinerary_revision(text):
            return "itinerary_revision"

        if self._is_dynamic_knowledge_query(text):
            return "dynamic_knowledge_query"

        if any(keyword in text for keyword in ["航班", "飞机", "机票"]):
            return "flight_search"

        if self._has_route_expression(text) and any(keyword in text for keyword in ["去", "到", "玩", "游"]):
            return "travel_plan"

        if self._is_guide_query(text):
            return "guide_query"

        if any(keyword in text for keyword in ["酒店", "住宿", "宾馆", "民宿", "住哪里", "住哪"]):
            return "hotel_search"

        if re.search(r".+飞.+", text) and not any(keyword in text for keyword in ["玩", "旅游", "旅行", "行程"]):
            return "flight_search"

        if any(keyword in text for keyword in ["景点", "好玩", "玩什么", "有什么玩", "观光", "游览"]):
            return "attraction_search"

        if any(keyword in text for keyword in ["旅行", "旅游", "行程", "计划", "安排", "规划"]):
            return "travel_plan"

        return "general_chat"

    def _is_guide_query(self, text: str) -> bool:
        """判断是否为目的地攻略知识查询"""
        if not self._find_cities(text):
            return False
        guide_keywords = [
            "攻略", "怎么玩", "怎么安排", "有什么好吃", "美食", "好吃", "海边", "拍照",
            "适合情侣", "适合亲子", "三天", "3天", "路线", "注意事项",
        ]
        return any(keyword in text for keyword in guide_keywords)

    def _is_dynamic_knowledge_query(self, text: str) -> bool:
        """判断是否为基于本轮外部动态知识的追问"""
        context_keywords = ["刚才", "刚刚", "上面", "前面", "刚才推荐", "这些景点", "这个景点", "推荐的景点"]
        detail_keywords = ["在哪里", "地址", "位置", "坐标", "来源", "有哪些", "是什么", "在哪"]
        poi_names = ["西湖", "灵隐寺", "西溪", "宋城", "鼓浪屿", "亚龙湾", "武侯祠", "宽窄巷子"]

        if any(keyword in text for keyword in context_keywords) and any(keyword in text for keyword in detail_keywords):
            return True
        if any(name in text for name in poi_names) and any(keyword in text for keyword in detail_keywords):
            return True
        return False

    def _is_itinerary_revision(self, text: str) -> bool:
        """判断是否为基于历史行程的计划修订请求"""
        revision_keywords = [
            "安排到", "安排在", "放到", "放在", "换一个", "替换", "不要去", "不想去",
            "删掉", "删除", "移除", "重新安排", "调整一下", "排一下顺序", "排序",
        ]
        poi_keywords = ["西湖", "灵隐寺", "西溪", "宋城", "景点", "行程"]
        return any(keyword in text for keyword in revision_keywords) and any(keyword in text for keyword in poi_keywords)

    def _extract_entities(self, text: str, intent: str) -> Dict:
        """提取旅行实体"""
        origin, destination = self._extract_route(text, intent)

        entities = {
            "origin": origin,
            "destination": destination,
            "departure_date": self._extract_departure_date(text),
            "return_date": None,
            "duration": self._extract_duration(text),
            "budget": self._extract_budget(text),
            "travelers": self._extract_travelers(text),
            "preferences": self._extract_preferences(text),
        }
        return entities

    def _extract_route(self, text: str, intent: str) -> tuple[Optional[str], Optional[str]]:
        """提取出发地和目的地"""
        route_patterns = [
            r"从(?P<origin>[^,，。\s]{2,6})(?:出发)?(?:去|到|飞往|前往)(?P<destination>[^,，。\s]{2,6})",
            r"(?P<origin>[^,，。\s]{2,6})(?:到|飞|去)(?P<destination>[^,，。\s]{2,6})",
        ]

        for pattern in route_patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            origin = self._match_city(match.group("origin"))
            destination = self._match_city(match.group("destination"))
            if origin and destination:
                return origin, destination

        found_cities = self._find_cities(text)
        if len(found_cities) >= 2:
            return found_cities[0], found_cities[1]
        if len(found_cities) == 1 and intent in {"hotel_search", "attraction_search", "travel_plan", "guide_query", "dynamic_knowledge_query", "itinerary_revision"}:
            return None, found_cities[0]
        return None, None

    def _match_city(self, text: str) -> Optional[str]:
        """从短文本中匹配城市名"""
        for city in self.CITIES:
            if city in text:
                return city
        return None

    def _find_cities(self, text: str) -> List[str]:
        """按文本出现顺序查找城市"""
        matches = []
        for city in self.CITIES:
            index = text.find(city)
            if index >= 0:
                matches.append((index, city))
        return [city for _, city in sorted(matches, key=lambda item: item[0])]

    def _has_route_expression(self, text: str) -> bool:
        """判断是否包含路线表达"""
        return bool(re.search(r".+(?:到|去|飞).+", text))

    def _extract_departure_date(self, text: str) -> Optional[str]:
        """提取出发日期"""
        full_date = re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?", text)
        if full_date:
            year, month, day = full_date.groups()
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

        month_day = re.search(r"(?<!\d)(\d{1,2})月(\d{1,2})日?", text)
        if month_day:
            month, day = month_day.groups()
            current_year = date.today().year
            return f"{current_year:04d}-{int(month):02d}-{int(day):02d}"

        relative_dates = {
            "今天": 0,
            "明天": 1,
            "后天": 2,
            "大后天": 3,
            "下周": 7,
        }
        for keyword, days in relative_dates.items():
            if keyword in text:
                target = date.today() + timedelta(days=days)
                return target.isoformat()

        return None

    def _extract_duration(self, text: str) -> Optional[int]:
        """提取旅行天数"""
        digit_duration = re.search(r"(?<![月\d])(\d+)\s*(?:天|日)(?:游|旅行|旅游|行程|玩)?", text)
        if digit_duration:
            return int(digit_duration.group(1))

        nights_pattern = re.search(r"(\d+)\s*天\s*(\d+)\s*晚", text)
        if nights_pattern:
            return int(nights_pattern.group(1))

        chinese_duration = re.search(r"([一二两三四五六七八九十]+)\s*(?:天|日)(?:游|旅行|旅游|行程|玩)?", text)
        if chinese_duration:
            return self._chinese_number_to_int(chinese_duration.group(1))

        chinese_nights = re.search(r"([一二两三四五六七八九十]+)\s*天\s*([一二两三四五六七八九十]+)\s*晚", text)
        if chinese_nights:
            return self._chinese_number_to_int(chinese_nights.group(1))

        return None

    def _extract_budget(self, text: str) -> Optional[float]:
        """提取预算金额"""
        patterns = [
            r"预算\s*(\d+(?:\.\d+)?)\s*(?:元|块|人民币)?",
            r"(\d+(?:\.\d+)?)\s*(?:元|块|人民币)?\s*(?:以内|以下|左右)",
            r"人均\s*(\d+(?:\.\d+)?)\s*(?:元|块|人民币)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        return None

    def _extract_travelers(self, text: str) -> Optional[int]:
        """提取出行人数"""
        digit_match = re.search(r"(\d+)\s*(?:个人|人|位)", text)
        if digit_match:
            return int(digit_match.group(1))

        chinese_match = re.search(r"([一二两三四五六七八九十]+)\s*(?:个人|人|位)", text)
        if chinese_match:
            return self._chinese_number_to_int(chinese_match.group(1))

        return None

    def _extract_preferences(self, text: str) -> List[str]:
        """提取显式偏好"""
        preference_keywords = [
            "靠窗", "靠过道", "经济舱", "商务舱", "头等舱", "不要红眼航班",
            "便宜", "性价比", "舒适", "地铁附近", "离景点近", "自然风光",
            "人文", "亲子", "慢节奏", "美食", "高评分", "干净",
        ]
        return [keyword for keyword in preference_keywords if keyword in text]

    def _chinese_number_to_int(self, text: str) -> Optional[int]:
        """将简单中文数字转换为整数，支持一到十九"""
        if not text:
            return None
        if text in self.CHINESE_NUMBERS:
            return self.CHINESE_NUMBERS[text]
        if text.startswith("十"):
            suffix = text[1:]
            return 10 + self.CHINESE_NUMBERS.get(suffix, 0)
        if "十" in text:
            prefix, suffix = text.split("十", 1)
            return self.CHINESE_NUMBERS.get(prefix, 0) * 10 + self.CHINESE_NUMBERS.get(suffix, 0)
        return None

    def _should_use_llm_fallback(self, rule_result: Dict) -> bool:
        """判断是否需要尝试LLM fallback"""
        if not self.llm_client.available:
            return False

        intent = rule_result.get("intent")
        confidence = rule_result.get("confidence", 0)
        return intent == "general_chat" or confidence < self.FALLBACK_CONFIDENCE_THRESHOLD

    async def _llm_fallback(self, user_input: str, rule_result: Dict) -> Optional[Dict]:
        """调用LLM补充意图解析，失败时返回None"""
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=INTENT_FALLBACK_SYSTEM_PROMPT),
                LLMMessage(
                    role="user",
                    content=(
                        "请解析下面的旅行需求，并返回严格JSON。\n"
                        "JSON字段必须包含：intent、entities、confidence、missing_slots、followup_question。\n"
                        "intent只能是 travel_plan、flight_search、hotel_search、attraction_search、policy_query、guide_query、dynamic_knowledge_query、itinerary_revision、general_chat。\n"
                        "entities必须包含 origin、destination、departure_date、return_date、duration、budget、travelers、preferences。\n"
                        f"用户输入：{user_input}\n"
                        f"规则解析结果：{json.dumps(rule_result, ensure_ascii=False)}"
                    ),
                ),
            ],
            response_format="json_object",
            metadata={"fallback_for": "intent_parse"},
        )
        response = await self.llm_client.chat(request)
        if not response.success:
            return None

        parsed_json = self._parse_llm_json(response.content)
        if not parsed_json:
            return None

        try:
            validated = TravelIntent.model_validate(parsed_json)
        except Exception:
            return None

        return self._normalize_llm_intent(validated)

    def _parse_llm_json(self, content: str) -> Optional[Dict]:
        """解析LLM返回的JSON，兼容Markdown代码块"""
        if not content:
            return None

        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def _normalize_llm_intent(self, intent: TravelIntent) -> Dict:
        """规范化LLM解析结果，补齐缺失槽位和追问"""
        result = intent.model_dump()
        entities = result.get("entities", {}) or {}
        missing_slots = self._detect_missing_slots(result.get("intent"), entities)
        result["missing_slots"] = missing_slots
        result["followup_question"] = result.get("followup_question") or self._build_followup_question(
            missing_slots,
            entities,
        )
        result["confidence"] = round(max(result.get("confidence", 0), self.FALLBACK_CONFIDENCE_THRESHOLD), 2)
        return result

    def _detect_missing_slots(self, intent: str, entities: Dict) -> List[str]:
        """识别当前意图缺失的关键槽位"""
        required_slots = self.REQUIRED_SLOTS.get(intent, [])
        return [slot for slot in required_slots if not entities.get(slot)]

    def _build_followup_question(self, missing_slots: List[str], entities: Dict) -> Optional[str]:
        """根据缺失槽位生成追问"""
        if not missing_slots:
            return None

        destination = entities.get("destination") or "目的地"
        origin = entities.get("origin") or "出发地"

        questions = {
            "origin": f"请问您准备从哪个城市出发去{destination}？",
            "destination": f"请问您计划去哪个城市旅行？",
            "departure_date": f"请问您计划什么时候从{origin}出发？我可以根据出发日期为您查询更合适的航班和酒店。",
            "duration": f"请问您计划在{destination}玩几天？这样我可以为您安排更合理的行程。",
        }
        return questions.get(missing_slots[0])

    def _calculate_confidence(self, intent: str, entities: Dict, missing_slots: List[str]) -> float:
        """计算规则解析置信度"""
        if intent == "general_chat":
            return 0.4

        filled_slots = sum(1 for value in entities.values() if value)
        score = 0.55 + min(filled_slots * 0.08, 0.32) - min(len(missing_slots) * 0.08, 0.24)
        return round(max(0.1, min(score, 0.95)), 2)
