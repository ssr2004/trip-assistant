"""
行程生成工具
根据目的地、天数、预算和偏好生成基础旅行行程
"""
from typing import Dict, List

from tools.registry import BaseTool


class ItineraryTool(BaseTool):
    """行程生成工具"""

    @property
    def name(self) -> str:
        return "generate_itinerary"

    @property
    def description(self) -> str:
        return "生成结构化旅行行程"

    async def execute(
        self,
        origin: str = None,
        destination: str = None,
        duration: int = 3,
        budget: float = None,
        travelers: int = None,
        preferences: List[str] = None,
        **kwargs,
    ) -> Dict:
        """
        生成基础行程

        Args:
            origin: 出发地
            destination: 目的地
            duration: 旅行天数
            budget: 预算
            travelers: 出行人数
            preferences: 用户偏好

        Returns:
            标准化行程生成结果
        """
        destination = destination or "目的地"
        duration = duration or 3
        preferences = preferences or []

        itinerary = self._build_itinerary(destination, duration, preferences)
        budget_summary = self._build_budget_summary(duration, budget, travelers)

        return {
            "success": True,
            "data": {
                "origin": origin,
                "destination": destination,
                "duration": duration,
                "budget": budget,
                "travelers": travelers,
                "preferences": preferences,
                "itinerary": itinerary,
                "budget_summary": budget_summary,
                "summary": self._build_summary(origin, destination, duration, budget),
            },
            "error": None,
            "metadata": {
                "source": "template_itinerary_generator",
                "tool": self.name,
            },
        }

    def _build_itinerary(self, destination: str, duration: int, preferences: List[str]) -> List[Dict]:
        """构建每日行程"""
        if destination == "杭州":
            base_days = [
                {
                    "day": 1,
                    "title": "抵达杭州与西湖初体验",
                    "activities": ["抵达杭州", "办理酒店入住", "游览西湖", "湖滨步行街晚餐"],
                    "notes": "第一天以适应节奏为主，不安排过高强度行程。",
                },
                {
                    "day": 2,
                    "title": "灵隐寺与西溪湿地",
                    "activities": ["灵隐寺", "飞来峰", "西溪国家湿地公园", "品尝杭帮菜"],
                    "notes": "兼顾历史文化和自然风光，适合完整游玩日。",
                },
                {
                    "day": 3,
                    "title": "宋城或城市休闲后返程",
                    "activities": ["宋城", "杭州特色伴手礼", "整理返程"],
                    "notes": "根据返程时间选择宋城演出或轻松购物。",
                },
            ]
        else:
            base_days = [
                {
                    "day": 1,
                    "title": f"抵达{destination}与城市初体验",
                    "activities": [f"抵达{destination}", "办理酒店入住", "游览城市核心景区", "品尝当地特色美食"],
                    "notes": "第一天以抵达和轻松游览为主。",
                },
                {
                    "day": 2,
                    "title": f"{destination}深度游览",
                    "activities": ["热门景点游览", "当地文化体验", "特色餐饮", "夜间休闲"],
                    "notes": "安排较完整的景点和文化体验。",
                },
                {
                    "day": 3,
                    "title": f"{destination}休闲与返程",
                    "activities": ["轻松游览", "购买伴手礼", "整理返程"],
                    "notes": "最后一天降低强度，预留返程时间。",
                },
            ]

        if "慢节奏" in preferences:
            for day in base_days:
                day["notes"] += " 已根据慢节奏偏好减少高强度安排。"

        if duration <= len(base_days):
            return base_days[:duration]

        itinerary = base_days[:]
        for day_index in range(len(base_days) + 1, duration + 1):
            itinerary.append({
                "day": day_index,
                "title": f"{destination}自由探索日",
                "activities": ["根据兴趣选择小众景点", "体验当地美食", "休闲散步"],
                "notes": "该日作为弹性安排，可根据天气和体力调整。",
            })
        return itinerary

    def _build_budget_summary(self, duration: int, budget: float, travelers: int) -> Dict:
        """构建预算摘要"""
        travelers = travelers or 1
        estimated_hotel = max(duration - 1, 1) * 600
        estimated_food = duration * 180 * travelers
        estimated_local_transport = duration * 80 * travelers
        estimated_attractions = duration * 120 * travelers
        estimated_total_without_flight = (
            estimated_hotel + estimated_food + estimated_local_transport + estimated_attractions
        )

        return {
            "input_budget": budget,
            "travelers": travelers,
            "estimated_hotel": estimated_hotel,
            "estimated_food": estimated_food,
            "estimated_local_transport": estimated_local_transport,
            "estimated_attractions": estimated_attractions,
            "estimated_total_without_flight": estimated_total_without_flight,
            "budget_note": self._build_budget_note(budget, estimated_total_without_flight),
        }

    def _build_budget_note(self, budget: float, estimated_total: float) -> str:
        """生成预算说明"""
        if budget is None:
            return "用户未提供明确预算，当前为基础估算，后续可结合航班和酒店价格细化。"
        if budget >= estimated_total:
            return "预算基本可以覆盖当地住宿、餐饮、交通和景点开销，机票费用需另行结合航班结果计算。"
        return "预算可能偏紧，建议优先选择经济型酒店、免费景点和公共交通。"

    def _build_summary(self, origin: str, destination: str, duration: int, budget: float) -> str:
        """生成行程摘要"""
        route = f"从{origin}出发前往{destination}" if origin else f"前往{destination}"
        budget_text = f"，预算约{budget:g}元" if budget else ""
        return f"已为您生成{route}的{duration}天旅行行程{budget_text}。"
