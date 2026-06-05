"""
用户偏好抽取器
基于规则从用户自然语言中抽取旅行长期偏好
"""
from datetime import datetime
from typing import Dict, List, Optional

from models.memory import UserPreference


class PreferenceExtractor:
    """规则型用户偏好抽取器"""

    TRAVEL_STYLE_PATTERNS: Dict[str, List[str]] = {
        "慢节奏": ["慢节奏", "慢一点", "不要太赶", "别太赶", "不想太赶", "不赶", "轻松", "休闲", "不要太累", "别太累"],
        "深度游": ["深度游", "深度体验", "深度旅行"],
        "亲子": ["亲子", "带孩子", "带娃", "小朋友"],
        "情侣": ["情侣", "约会", "浪漫"],
        "少走路": ["少走路", "不想走太多", "别走太多", "老人", "腿脚不方便"],
    }

    HOTEL_PATTERNS: Dict[str, List[str]] = {
        "地铁附近": ["地铁附近", "靠近地铁", "离地铁近", "地铁口", "近地铁"],
        "交通方便": ["交通方便", "交通便利", "出行方便"],
        "安静": ["安静", "不要吵", "别太吵"],
        "经济型酒店": ["经济型酒店", "便宜酒店", "酒店便宜", "住宿便宜"],
        "高星级酒店": ["高星级", "五星", "五星级", "高端酒店", "豪华酒店"],
    }

    TRANSPORT_PATTERNS: Dict[str, List[str]] = {
        "高铁优先": ["高铁优先", "优先高铁", "坐高铁", "高铁出行"],
        "飞机优先": ["飞机优先", "优先飞机", "坐飞机", "航班优先"],
        "少换乘": ["少换乘", "不想换乘", "不要换乘", "少转车"],
        "直达": ["直达", "不要中转", "不中转"],
    }

    ATTRACTION_PATTERNS: Dict[str, List[str]] = {
        "自然风光": ["自然风光", "风景", "山水", "湖", "森林", "湿地"],
        "人文历史": ["人文", "历史", "古迹", "文化", "寺", "博物馆"],
        "海边": ["海边", "看海", "海岛", "沙滩"],
        "拍照": ["拍照", "出片", "摄影"],
        "美食": ["美食", "小吃", "当地菜", "特色菜"],
        "购物": ["购物", "商场", "买东西"],
    }

    FOOD_PATTERNS: Dict[str, List[str]] = {
        "当地美食": ["当地美食", "特色美食", "本地美食", "当地菜", "特色菜"],
        "清淡": ["清淡", "少辣", "不吃辣"],
        "素食": ["素食", "吃素", "素菜"],
        "海鲜": ["海鲜", "吃海鲜"],
    }

    EXCLUDED_PATTERNS: Dict[str, List[str]] = {
        "不吃辣": ["不吃辣", "不能吃辣", "不要辣"],
        "不喝酒": ["不喝酒", "不要酒吧", "不去酒吧"],
        "排除购物": ["不购物", "不要购物", "不想购物"],
        "排除夜生活": ["不要夜生活", "不去夜店", "不去酒吧"],
    }

    BUDGET_PATTERNS: Dict[str, List[str]] = {
        "经济型": ["省钱", "经济", "预算有限", "便宜", "少花钱", "性价比"],
        "中档": ["中档", "适中", "别太贵", "舒适一点"],
        "舒适型": ["舒适", "舒服", "品质", "体验好"],
        "豪华型": ["豪华", "高端", "不差钱", "预算充足"],
    }

    def extract(self, text: str) -> UserPreference:
        """从文本中抽取用户偏好"""
        text = text or ""
        preference = UserPreference()

        preference.travel_styles = self._match_patterns(text, self.TRAVEL_STYLE_PATTERNS)
        preference.hotel_preferences = self._match_patterns(text, self.HOTEL_PATTERNS)
        preference.transport_preferences = self._match_patterns(text, self.TRANSPORT_PATTERNS)
        preference.attraction_preferences = self._match_patterns(text, self.ATTRACTION_PATTERNS)
        preference.food_preferences = self._match_patterns(text, self.FOOD_PATTERNS)
        preference.budget_preference = self._match_budget(text)
        preference.dietary_restrictions = self._match_patterns(text, self.EXCLUDED_PATTERNS)
        preference.excluded_preferences = list(preference.dietary_restrictions)
        preference.preference_evidence = self._build_evidence(text, {
            "travel_styles": self.TRAVEL_STYLE_PATTERNS,
            "hotel_preferences": self.HOTEL_PATTERNS,
            "transport_preferences": self.TRANSPORT_PATTERNS,
            "attraction_preferences": self.ATTRACTION_PATTERNS,
            "food_preferences": self.FOOD_PATTERNS,
            "dietary_restrictions": self.EXCLUDED_PATTERNS,
        })
        preference.raw_preferences = self._dedupe(
            preference.travel_styles
            + preference.hotel_preferences
            + preference.transport_preferences
            + preference.attraction_preferences
            + preference.food_preferences
            + preference.dietary_restrictions
            + ([preference.budget_preference] if preference.budget_preference else [])
        )

        if preference.has_preferences():
            preference.updated_at = datetime.now().isoformat()
        return preference

    def _match_patterns(self, text: str, patterns: Dict[str, List[str]]) -> List[str]:
        """按规则匹配偏好标签"""
        matched = []
        for label, keywords in patterns.items():
            if any(keyword in text for keyword in keywords):
                matched.append(label)
        return matched

    def _match_budget(self, text: str) -> Optional[str]:
        """匹配预算偏好，按规则顺序返回第一个命中的预算标签"""
        for label, keywords in self.BUDGET_PATTERNS.items():
            if any(keyword in text for keyword in keywords):
                return label
        return None

    def _build_evidence(self, text: str, pattern_groups: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
        """构建偏好证据，记录每个字段命中的标签。"""
        evidence: Dict[str, List[str]] = {}
        for field, patterns in pattern_groups.items():
            matched = self._match_patterns(text, patterns)
            if matched:
                evidence[field] = matched
        budget = self._match_budget(text)
        if budget:
            evidence["budget_preference"] = [budget]
        return evidence

    def _dedupe(self, values: List[str]) -> List[str]:
        """保持顺序去重"""
        result = []
        for value in values:
            if value and value not in result:
                result.append(value)
        return result
