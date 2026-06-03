"""
长期记忆
存储用户偏好和历史行为
"""
from typing import Any, Dict, List
import json
import os

from models.memory import UserPreference


class LongTermMemory:
    """长期记忆"""

    DEFAULT_PREFERENCES = {
        "travel_styles": [],
        "hotel_preferences": [],
        "transport_preferences": [],
        "attraction_preferences": [],
        "food_preferences": [],
        "budget_preference": None,
        "raw_preferences": [],
        "updated_at": None,
        # 兼容早期记忆字段
        "seat_preference": None,
        "cabin_class": None,
        "hotel_star": None,
        "budget_range": None,
        "travel_style": None,
        "dietary_restrictions": [],
    }

    LIST_FIELDS = {
        "travel_styles",
        "hotel_preferences",
        "transport_preferences",
        "attraction_preferences",
        "food_preferences",
        "raw_preferences",
        "dietary_restrictions",
    }

    def __init__(self, storage_path: str = "data/long_term_memory.json"):
        """
        初始化长期记忆

        Args:
            storage_path: 存储路径
        """
        self.storage_path = storage_path
        self.preferences: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        """加载记忆"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.preferences = json.load(f)
            except Exception:
                self.preferences = {}

    def _save(self):
        """保存记忆"""
        directory = os.path.dirname(self.storage_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.preferences, f, ensure_ascii=False, indent=2)

    def get_preferences(self, session_id: str = "default") -> Dict:
        """获取用户偏好"""
        stored = self.preferences.get(session_id, {})
        return self._with_defaults(stored)

    def update_preferences(self, new_preferences: Dict | UserPreference, session_id: str = "default") -> Dict:
        """
        更新用户偏好

        Args:
            new_preferences: 新的偏好
            session_id: 会话ID

        Returns:
            合并后的用户偏好
        """
        preference_dict = self._normalize_preferences(new_preferences)
        if not self._has_meaningful_preferences(preference_dict):
            return self.get_preferences(session_id)

        current = self._with_defaults(self.preferences.get(session_id, {}))
        merged = dict(current)

        for key, value in preference_dict.items():
            if value in (None, "", [], {}):
                continue
            if key in self.LIST_FIELDS:
                merged[key] = self._merge_list(merged.get(key, []), value if isinstance(value, list) else [value])
            else:
                merged[key] = value

        self.preferences[session_id] = merged
        self._save()
        return self.get_preferences(session_id)

    def add_behavior(self, behavior: Dict, session_id: str = "default"):
        """
        记录用户行为

        Args:
            behavior: 行为记录
            session_id: 会话ID
        """
        if session_id not in self.preferences:
            self.preferences[session_id] = {}

        if "behaviors" not in self.preferences[session_id]:
            self.preferences[session_id]["behaviors"] = []

        self.preferences[session_id]["behaviors"].append(behavior)
        self._save()

    def _normalize_preferences(self, preferences: Dict | UserPreference) -> Dict:
        """将偏好模型或字典规范化为dict"""
        if isinstance(preferences, UserPreference):
            return preferences.model_dump()
        return dict(preferences or {})

    def _with_defaults(self, preferences: Dict) -> Dict:
        """补充默认字段"""
        result = dict(self.DEFAULT_PREFERENCES)
        result.update(preferences or {})
        for field in self.LIST_FIELDS:
            value = result.get(field)
            result[field] = value if isinstance(value, list) else []
        return result

    def _has_meaningful_preferences(self, preferences: Dict) -> bool:
        """判断偏好更新是否包含有效内容"""
        ignored_fields = {"updated_at"}
        for key, value in preferences.items():
            if key in ignored_fields:
                continue
            if value not in (None, "", [], {}):
                return True
        return False

    def _merge_list(self, old_values: List[Any], new_values: List[Any]) -> List[Any]:
        """保持顺序合并列表并去重"""
        merged = list(old_values or [])
        for value in new_values or []:
            if value and value not in merged:
                merged.append(value)
        return merged
