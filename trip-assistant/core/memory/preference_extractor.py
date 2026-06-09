"""
用户偏好抽取器
基于规则从用户自然语言中抽取旅行长期偏好
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import re
from typing import Any, Dict, List, Optional

from core.llm import LLMClient, LLMMessage, LLMRequest
from core.llm.json_repair import parse_llm_json_object
from models.memory import UserPreference


@dataclass
class PreferenceExtractionReport:
    """Explainable decision report for rule-first preference extraction."""

    preference_intent_score: float
    rule_confidence: float
    category_count: int
    raw_hit_count: int
    abstract_marker_count: int
    ambiguity_score: float
    should_use_llm: bool
    decision_reason: str
    extraction_mode: str = "rule_only"
    rule_hit_fields: List[str] = field(default_factory=list)
    abstract_markers: List[str] = field(default_factory=list)
    ambiguity_markers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PreferenceExtractor:
    """规则型用户偏好抽取器"""

    LIST_FIELDS = {
        "travel_styles",
        "hotel_preferences",
        "transport_preferences",
        "attraction_preferences",
        "food_preferences",
        "dietary_restrictions",
        "excluded_preferences",
    }

    SCALAR_FIELDS = {"budget_preference"}

    PREFERENCE_INTENT_LLM_THRESHOLD = 0.55
    RULE_CONFIDENCE_LOW_THRESHOLD = 0.75
    RULE_CONFIDENCE_ACCEPT_THRESHOLD = 0.85
    ABSTRACT_RULE_CONFIDENCE_THRESHOLD = 0.90
    AMBIGUITY_LLM_THRESHOLD = 0.50

    PREFERENCE_INTENT_MARKERS = [
        "喜欢",
        "希望",
        "想要",
        "最好",
        "偏好",
        "倾向",
        "不要",
        "不想",
        "别太",
        "更想",
        "尽量",
        "适合",
    ]

    AMBIGUITY_MARKERS = [
        "一点",
        "一些",
        "别太",
        "不要太",
        "但",
        "但是",
        "尽量",
        "更想",
        "适合",
        "舒服",
        "舒适",
        "轻松",
        "少折腾",
        "别折腾",
        "游客化",
        "当地生活",
        "赶场",
    ]

    ABSTRACT_PREFERENCE_MARKERS = [
        "不想",
        "不要",
        "别太",
        "少折腾",
        "别折腾",
        "赶场",
        "舒服",
        "舒适",
        "松弛",
        "放松",
        "体验当地",
        "当地生活",
        "游客化",
        "带父母",
        "带老人",
        "带孩子",
        "不累",
        "轻松",
    ]

    CANONICAL_VALUES = {
        "travel_styles": {
            "慢节奏",
            "深度游",
            "亲子",
            "情侣",
            "少走路",
            "轻松休闲",
            "低强度",
            "当地生活体验",
        },
        "hotel_preferences": {
            "地铁附近",
            "交通方便",
            "安静",
            "经济型酒店",
            "高星级酒店",
            "舒适型酒店",
        },
        "transport_preferences": {
            "高铁优先",
            "飞机优先",
            "少换乘",
            "直达",
            "低折腾",
        },
        "attraction_preferences": {
            "自然风光",
            "人文历史",
            "海边",
            "拍照",
            "美食",
            "购物",
            "非游客化体验",
        },
        "food_preferences": {
            "当地美食",
            "清淡",
            "素食",
            "海鲜",
        },
        "dietary_restrictions": {
            "不吃辣",
            "不喝酒",
        },
        "excluded_preferences": {
            "不吃辣",
            "不喝酒",
            "排除购物",
            "排除夜生活",
            "排除赶场式行程",
            "排除过度游客化",
        },
    }

    VALUE_ALIASES = {
        "别太赶": "慢节奏",
        "不要太赶": "慢节奏",
        "不想赶场": "排除赶场式行程",
        "少折腾": "低折腾",
        "别折腾": "低折腾",
        "不要太累": "低强度",
        "不累": "低强度",
        "舒服一点": "舒适型酒店",
        "住得舒服": "舒适型酒店",
        "当地生活": "当地生活体验",
        "不要太游客化": "排除过度游客化",
        "不游客化": "排除过度游客化",
    }

    REAL_TRAVEL_STYLE_PATTERNS: Dict[str, List[str]] = {
        "慢节奏": ["慢节奏", "慢一点", "不要太赶", "别太赶", "不想太赶", "不赶", "轻松", "休闲", "不要太累", "别太累"],
        "深度游": ["深度游", "深度体验", "深度旅行"],
        "亲子": ["亲子", "带孩子", "带娃", "小朋友"],
        "情侣": ["情侣", "约会", "浪漫"],
        "少走路": ["少走路", "不想走太多", "别走太多", "老人", "腿脚不方便"],
    }

    REAL_HOTEL_PATTERNS: Dict[str, List[str]] = {
        "地铁附近": ["地铁附近", "靠近地铁", "离地铁近", "地铁口", "近地铁"],
        "交通方便": ["交通方便", "交通便利", "出行方便"],
        "安静": ["安静", "不要吵", "别太吵"],
        "经济型酒店": ["经济型酒店", "便宜酒店", "酒店便宜", "住宿便宜"],
        "高星级酒店": ["高星级", "五星", "五星级", "高端酒店", "豪华酒店"],
    }

    REAL_TRANSPORT_PATTERNS: Dict[str, List[str]] = {
        "高铁优先": ["高铁优先", "优先高铁", "坐高铁", "高铁出行"],
        "飞机优先": ["飞机优先", "优先飞机", "坐飞机", "航班优先"],
        "少换乘": ["少换乘", "不想换乘", "不要换乘", "少转车"],
        "直达": ["直达", "不要中转", "不中转"],
    }

    REAL_ATTRACTION_PATTERNS: Dict[str, List[str]] = {
        "自然风光": ["自然风光", "风景", "山水", "湖", "森林", "湿地"],
        "人文历史": ["人文", "历史", "古迹", "文化", "寺", "博物馆"],
        "海边": ["海边", "看海", "海岛", "沙滩"],
        "拍照": ["拍照", "出片", "摄影"],
        "美食": ["美食", "小吃", "当地菜", "特色菜"],
        "购物": ["购物", "商场", "买东西"],
    }

    REAL_FOOD_PATTERNS: Dict[str, List[str]] = {
        "当地美食": ["当地美食", "特色美食", "本地美食", "当地菜", "特色菜"],
        "清淡": ["清淡", "少辣", "不吃辣"],
        "素食": ["素食", "吃素", "素菜"],
        "海鲜": ["海鲜", "吃海鲜"],
    }

    REAL_EXCLUDED_PATTERNS: Dict[str, List[str]] = {
        "不吃辣": ["不吃辣", "不能吃辣", "不要辣"],
        "不喝酒": ["不喝酒", "不要酒吧", "不去酒吧"],
        "排除购物": ["不购物", "不要购物", "不想购物"],
        "排除夜生活": ["不要夜生活", "不去夜店", "不去酒吧"],
    }

    REAL_BUDGET_PATTERNS: Dict[str, List[str]] = {
        "经济型": ["省钱", "经济", "预算有限", "便宜", "少花钱", "性价比"],
        "中档": ["中档", "适中", "别太贵", "舒适一点"],
        "舒适型": ["舒适", "舒服", "品质", "体验好"],
        "豪华型": ["豪华", "高端", "不差钱", "预算充足"],
    }

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client or LLMClient()
        self.last_extraction_report: Optional[PreferenceExtractionReport] = None

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

        preference.travel_styles = self._dedupe(
            self._match_patterns(text, self.TRAVEL_STYLE_PATTERNS)
            + self._match_patterns(text, self.REAL_TRAVEL_STYLE_PATTERNS)
        )
        preference.hotel_preferences = self._dedupe(
            self._match_patterns(text, self.HOTEL_PATTERNS)
            + self._match_patterns(text, self.REAL_HOTEL_PATTERNS)
        )
        preference.transport_preferences = self._dedupe(
            self._match_patterns(text, self.TRANSPORT_PATTERNS)
            + self._match_patterns(text, self.REAL_TRANSPORT_PATTERNS)
        )
        preference.attraction_preferences = self._dedupe(
            self._match_patterns(text, self.ATTRACTION_PATTERNS)
            + self._match_patterns(text, self.REAL_ATTRACTION_PATTERNS)
        )
        preference.food_preferences = self._dedupe(
            self._match_patterns(text, self.FOOD_PATTERNS)
            + self._match_patterns(text, self.REAL_FOOD_PATTERNS)
        )
        preference.budget_preference = self._match_budget(text)
        preference.dietary_restrictions = self._dedupe(
            self._match_patterns(text, self.EXCLUDED_PATTERNS)
            + self._match_patterns(text, self.REAL_EXCLUDED_PATTERNS)
        )
        preference.excluded_preferences = list(preference.dietary_restrictions)
        preference.preference_evidence = self._build_evidence(text, {
            "travel_styles": self.TRAVEL_STYLE_PATTERNS,
            "hotel_preferences": self.HOTEL_PATTERNS,
            "transport_preferences": self.TRANSPORT_PATTERNS,
            "attraction_preferences": self.ATTRACTION_PATTERNS,
            "food_preferences": self.FOOD_PATTERNS,
            "dietary_restrictions": self.EXCLUDED_PATTERNS,
        })
        preference.preference_evidence = self._merge_evidence(
            preference.preference_evidence,
            self._build_evidence(text, {
                "travel_styles": self.REAL_TRAVEL_STYLE_PATTERNS,
                "hotel_preferences": self.REAL_HOTEL_PATTERNS,
                "transport_preferences": self.REAL_TRANSPORT_PATTERNS,
                "attraction_preferences": self.REAL_ATTRACTION_PATTERNS,
                "food_preferences": self.REAL_FOOD_PATTERNS,
                "dietary_restrictions": self.REAL_EXCLUDED_PATTERNS,
            }),
        )
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

    async def extract_hybrid(self, text: str) -> UserPreference:
        """Rule-first preference extraction with LLM fallback for abstract wording."""
        rule_preference = self.extract(text)
        report = self.evaluate_extraction(text, rule_preference)
        if not report.should_use_llm:
            report.extraction_mode = "rule_only"
            self.last_extraction_report = report
            return rule_preference

        llm_preference = await self._extract_with_llm(text, rule_preference)
        if not llm_preference.has_preferences():
            report.extraction_mode = "llm_failed_rule_fallback"
            self.last_extraction_report = report
            return rule_preference
        report.extraction_mode = "rule_llm_merged" if rule_preference.has_preferences() else "llm_only"
        self.last_extraction_report = report
        return self._merge_preferences(rule_preference, llm_preference)

    def _should_use_llm_fallback(self, text: str, rule_preference: UserPreference) -> bool:
        """Use LLM only when rules are weak and the text looks preference-bearing."""
        return self.evaluate_extraction(text, rule_preference).should_use_llm

    def evaluate_extraction(
        self,
        text: str,
        rule_preference: Optional[UserPreference] = None,
    ) -> PreferenceExtractionReport:
        """Score whether rule extraction is confident enough or should escalate to LLM."""
        text = text or ""
        rule_preference = rule_preference or self.extract(text)
        abstract_markers = self._matched_markers(text, self.ABSTRACT_PREFERENCE_MARKERS)
        ambiguity_markers = self._matched_markers(text, self.AMBIGUITY_MARKERS)
        intent_markers = self._matched_markers(text, self.PREFERENCE_INTENT_MARKERS)
        rule_hit_fields = self._rule_hit_fields(rule_preference)
        category_count = len(rule_hit_fields)
        raw_hit_count = len(rule_preference.raw_preferences or [])
        ambiguity_score = self._score_ambiguity(abstract_markers, ambiguity_markers)
        preference_intent_score = self._score_preference_intent(
            rule_preference=rule_preference,
            intent_markers=intent_markers,
            abstract_markers=abstract_markers,
            ambiguity_score=ambiguity_score,
        )
        rule_confidence = self._score_rule_confidence(
            rule_preference=rule_preference,
            category_count=category_count,
            raw_hit_count=raw_hit_count,
            abstract_marker_count=len(abstract_markers),
            ambiguity_score=ambiguity_score,
        )
        should_use_llm, reason = self._decide_llm_fallback(
            preference_intent_score=preference_intent_score,
            rule_confidence=rule_confidence,
            category_count=category_count,
            abstract_marker_count=len(abstract_markers),
            ambiguity_score=ambiguity_score,
            has_rule_preferences=rule_preference.has_preferences(),
        )
        return PreferenceExtractionReport(
            preference_intent_score=preference_intent_score,
            rule_confidence=rule_confidence,
            category_count=category_count,
            raw_hit_count=raw_hit_count,
            abstract_marker_count=len(abstract_markers),
            ambiguity_score=ambiguity_score,
            should_use_llm=should_use_llm,
            decision_reason=reason,
            rule_hit_fields=rule_hit_fields,
            abstract_markers=abstract_markers,
            ambiguity_markers=ambiguity_markers,
        )

    def _matched_markers(self, text: str, markers: List[str]) -> List[str]:
        return [marker for marker in markers if marker and marker in text]

    def _rule_hit_fields(self, preference: UserPreference) -> List[str]:
        fields = []
        for field in sorted(self.LIST_FIELDS):
            if getattr(preference, field, None):
                fields.append(field)
        if preference.budget_preference:
            fields.append("budget_preference")
        return fields

    def _score_ambiguity(self, abstract_markers: List[str], ambiguity_markers: List[str]) -> float:
        score = min(len(abstract_markers) * 0.18 + len(ambiguity_markers) * 0.16, 1.0)
        return round(score, 4)

    def _score_preference_intent(
        self,
        *,
        rule_preference: UserPreference,
        intent_markers: List[str],
        abstract_markers: List[str],
        ambiguity_score: float,
    ) -> float:
        score = 0.0
        if rule_preference.has_preferences():
            score += 0.45
        if intent_markers:
            score += 0.35
        if abstract_markers:
            score += 0.30
        if ambiguity_score >= 0.25:
            score += 0.15
        return round(min(score, 1.0), 4)

    def _score_rule_confidence(
        self,
        *,
        rule_preference: UserPreference,
        category_count: int,
        raw_hit_count: int,
        abstract_marker_count: int,
        ambiguity_score: float,
    ) -> float:
        if not rule_preference.has_preferences():
            return 0.0
        score = 0.35
        score += min(raw_hit_count * 0.15, 0.35)
        score += min(category_count * 0.10, 0.20)
        if rule_preference.preference_evidence:
            score += 0.10
        if abstract_marker_count:
            score -= 0.15
        if ambiguity_score >= self.AMBIGUITY_LLM_THRESHOLD:
            score -= 0.10
        if raw_hit_count >= 3 and category_count >= 3 and ambiguity_score < self.AMBIGUITY_LLM_THRESHOLD:
            score = max(score, 0.90)
        return round(min(max(score, 0.0), 1.0), 4)

    def _decide_llm_fallback(
        self,
        *,
        preference_intent_score: float,
        rule_confidence: float,
        category_count: int,
        abstract_marker_count: int,
        ambiguity_score: float,
        has_rule_preferences: bool,
    ) -> tuple[bool, str]:
        if preference_intent_score < self.PREFERENCE_INTENT_LLM_THRESHOLD:
            return False, "preference_intent_below_threshold"
        if not has_rule_preferences:
            return True, "preference_intent_without_rule_hit"
        if rule_confidence < self.RULE_CONFIDENCE_LOW_THRESHOLD:
            return True, "rule_confidence_below_threshold"
        if abstract_marker_count > 0 and rule_confidence < self.ABSTRACT_RULE_CONFIDENCE_THRESHOLD:
            return True, "abstract_preference_requires_semantic_mapping"
        if ambiguity_score >= self.AMBIGUITY_LLM_THRESHOLD:
            return True, "ambiguity_score_above_threshold"
        if rule_confidence >= self.RULE_CONFIDENCE_ACCEPT_THRESHOLD and category_count >= 2:
            return False, "high_confidence_rule_extraction"
        return False, "rule_extraction_accepted"

    def _looks_like_preference_statement(self, text: str) -> bool:
        if self._looks_abstract(text):
            return True
        return bool(self._matched_markers(text or "", self.PREFERENCE_INTENT_MARKERS))

    def _looks_abstract(self, text: str) -> bool:
        return any(marker in text for marker in self.ABSTRACT_PREFERENCE_MARKERS)

    async def _extract_with_llm(self, text: str, rule_preference: UserPreference) -> UserPreference:
        request = LLMRequest(
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "You extract long-term travel preferences for a travel planning agent. "
                        "Return strict JSON only. Do not invent route, hotel, flight, price, or POI data. "
                        "Map abstract wording into stable preference labels."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=(
                        "Extract travel preferences from the user message.\n"
                        "Allowed JSON fields: travel_styles, hotel_preferences, transport_preferences, "
                        "attraction_preferences, food_preferences, dietary_restrictions, excluded_preferences, "
                        "budget_preference, preference_evidence.\n"
                        "List fields must be arrays of short Chinese labels. budget_preference must be one of "
                        "经济型, 中档, 舒适型, 豪华型, or null.\n"
                        "Prefer these labels when applicable:\n"
                        f"{json.dumps(self._allowed_label_payload(), ensure_ascii=False)}\n"
                        "preference_evidence maps each field to short evidence phrases from the user message.\n"
                        f"Rule extraction result:\n{rule_preference.model_dump_json()}\n"
                        f"User message:\n{text}"
                    ),
                ),
            ],
            response_format="json_object",
            temperature=0,
            metadata={"fallback_for": "preference_extraction"},
        )
        response = await self.llm_client.chat(request)
        if not response.success:
            return UserPreference()

        parsed = parse_llm_json_object(response.content)
        if not parsed:
            return UserPreference()
        return self._sanitize_llm_preference(parsed)

    def _allowed_label_payload(self) -> Dict[str, List[str]]:
        return {field: sorted(values) for field, values in self.CANONICAL_VALUES.items()}

    def _sanitize_llm_preference(self, payload: Dict[str, Any]) -> UserPreference:
        preference = UserPreference()
        for field in self.LIST_FIELDS:
            values = payload.get(field, [])
            setattr(preference, field, self._sanitize_list_field(field, values))

        budget = payload.get("budget_preference")
        if isinstance(budget, str):
            budget = self._normalize_budget_label(budget)
            if budget:
                preference.budget_preference = budget

        evidence = payload.get("preference_evidence", {})
        if isinstance(evidence, dict):
            preference.preference_evidence = self._sanitize_evidence(evidence)

        preference.raw_preferences = self._dedupe(
            preference.travel_styles
            + preference.hotel_preferences
            + preference.transport_preferences
            + preference.attraction_preferences
            + preference.food_preferences
            + preference.dietary_restrictions
            + preference.excluded_preferences
            + ([preference.budget_preference] if preference.budget_preference else [])
        )
        if preference.has_preferences():
            preference.updated_at = datetime.now().isoformat()
        return preference

    def _sanitize_list_field(self, field: str, values: Any) -> List[str]:
        if not isinstance(values, list):
            values = [values] if isinstance(values, str) else []
        cleaned = []
        allowed = self.CANONICAL_VALUES.get(field, set())
        for value in values[:8]:
            if not isinstance(value, str):
                continue
            normalized = self._normalize_value(value)
            if not normalized or len(normalized) > 24:
                continue
            if allowed and normalized not in allowed and normalized not in self.VALUE_ALIASES.values():
                continue
            cleaned.append(normalized)
        return self._dedupe(cleaned)

    def _normalize_value(self, value: str) -> str:
        value = re.sub(r"\s+", "", value.strip())
        return self.VALUE_ALIASES.get(value, value)

    def _normalize_budget_label(self, value: str) -> Optional[str]:
        value = self._normalize_value(value)
        aliases = {
            "省钱": "经济型",
            "经济": "经济型",
            "预算有限": "经济型",
            "适中": "中档",
            "中等": "中档",
            "舒适": "舒适型",
            "品质": "舒适型",
            "高端": "豪华型",
            "豪华": "豪华型",
        }
        value = aliases.get(value, value)
        return value if value in {"经济型", "中档", "舒适型", "豪华型"} else None

    def _sanitize_evidence(self, evidence: Dict[str, Any]) -> Dict[str, List[str]]:
        cleaned: Dict[str, List[str]] = {}
        for field, values in evidence.items():
            if field not in self.LIST_FIELDS and field not in self.SCALAR_FIELDS:
                continue
            if not isinstance(values, list):
                values = [values] if isinstance(values, str) else []
            field_values = []
            for value in values[:5]:
                if isinstance(value, str):
                    text = value.strip()
                    if text and len(text) <= 40:
                        field_values.append(text)
            if field_values:
                cleaned[field] = self._dedupe(field_values)
        return cleaned

    def _merge_preferences(self, rule_preference: UserPreference, llm_preference: UserPreference) -> UserPreference:
        merged = UserPreference()
        for field in self.LIST_FIELDS:
            setattr(
                merged,
                field,
                self._dedupe(
                    list(getattr(rule_preference, field, []) or [])
                    + list(getattr(llm_preference, field, []) or [])
                ),
            )
        merged.budget_preference = rule_preference.budget_preference or llm_preference.budget_preference
        merged.preference_evidence = self._merge_evidence(
            rule_preference.preference_evidence,
            llm_preference.preference_evidence,
        )
        merged.raw_preferences = self._dedupe(
            rule_preference.raw_preferences + llm_preference.raw_preferences
        )
        if merged.has_preferences():
            merged.updated_at = datetime.now().isoformat()
        return merged

    def _merge_evidence(self, first: Dict[str, List[str]], second: Dict[str, List[str]]) -> Dict[str, List[str]]:
        merged = dict(first or {})
        for field, values in (second or {}).items():
            merged[field] = self._dedupe(list(merged.get(field, [])) + list(values or []))
        return merged

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
