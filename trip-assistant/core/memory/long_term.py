"""
长期记忆
存储用户偏好和历史行为
"""
from typing import Dict, List, Optional
import json
import os


class LongTermMemory:
    """长期记忆"""

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
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.preferences, f, ensure_ascii=False, indent=2)

    def get_preferences(self, session_id: str = "default") -> Dict:
        """获取用户偏好"""
        return self.preferences.get(session_id, {
            "seat_preference": None,  # 靠窗/过道
            "cabin_class": None,  # 经济舱/商务舱
            "hotel_star": None,  # 酒店星级
            "budget_range": None,  # 预算范围
            "travel_style": None,  # 旅行风格
            "dietary_restrictions": []  # 饮食限制
        })

    def update_preferences(self, new_preferences: Dict, session_id: str = "default"):
        """
        更新用户偏好

        Args:
            new_preferences: 新的偏好
            session_id: 会话ID
        """
        if session_id not in self.preferences:
            self.preferences[session_id] = {}

        self.preferences[session_id].update(new_preferences)
        self._save()

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
