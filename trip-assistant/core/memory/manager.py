"""
记忆管理器
管理短期记忆、长期记忆和情景记忆
"""
from typing import Dict, List, Optional

from core.memory.short_term import ShortTermMemory
from core.memory.long_term import LongTermMemory
from core.memory.episodic import EpisodicMemory
from core.memory.preference_extractor import PreferenceExtractor


class MemoryManager:
    """记忆管理器"""

    def __init__(
        self,
        long_term_storage_path: Optional[str] = None,
        episodic_storage_path: Optional[str] = None,
    ):
        """
        初始化记忆管理器

        Args:
            long_term_storage_path: 长期记忆存储路径，默认使用运行时data目录
            episodic_storage_path: 情景记忆存储路径，默认使用运行时data目录
        """
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(long_term_storage_path) if long_term_storage_path else LongTermMemory()
        self.episodic = EpisodicMemory(episodic_storage_path) if episodic_storage_path else EpisodicMemory()
        self.preference_extractor = PreferenceExtractor()

    def save(self, user_message: str, ai_message: str, session_id: str = "default"):
        """
        保存对话到记忆系统

        Args:
            user_message: 用户消息
            ai_message: AI回复
            session_id: 会话ID
        """
        # 保存到短期记忆
        self.short_term.add(user_message, ai_message, session_id)

        # 保存到情景记忆
        self.episodic.save_interaction(user_message, ai_message, session_id)

        # 抽取并更新长期用户偏好
        self.extract_and_save_preferences(user_message, session_id)

    def retrieve(self, query: str, session_id: str = "default") -> Dict:
        """
        检索相关记忆

        Args:
            query: 查询文本
            session_id: 会话ID

        Returns:
            相关记忆上下文
        """
        context = {}

        # 从短期记忆获取对话历史
        context["recent_history"] = self.short_term.get_history(session_id)

        # 从长期记忆获取用户偏好
        preferences = self.long_term.get_preferences(session_id)
        context["preferences"] = preferences
        context["user_preferences"] = preferences

        # 从情景记忆获取相关交互
        context["relevant_interactions"] = self.episodic.search(query, session_id)

        return context

    def extract_and_save_preferences(self, user_message: str, session_id: str = "default") -> Dict:
        """从用户消息中抽取偏好并保存到长期记忆"""
        preferences = self.preference_extractor.extract(user_message)
        if not preferences.has_preferences():
            return self.long_term.get_preferences(session_id)
        return self.long_term.update_preferences(preferences, session_id)

    def get_history(self, session_id: str) -> List[Dict]:
        """获取对话历史"""
        return self.short_term.get_history(session_id)

    def clear_history(self, session_id: str):
        """清除对话历史"""
        self.short_term.clear(session_id)

    def update_preferences(self, preferences: Dict, session_id: str = "default"):
        """更新用户偏好"""
        return self.long_term.update_preferences(preferences, session_id)
