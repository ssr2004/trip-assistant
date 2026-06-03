"""
短期记忆
管理当前会话的对话历史
"""
from typing import Dict, List
from collections import defaultdict


class ShortTermMemory:
    """短期记忆"""

    def __init__(self, max_turns: int = 20):
        """
        初始化短期记忆

        Args:
            max_turns: 最大保留轮数
        """
        self.max_turns = max_turns
        self.conversations: Dict[str, List[Dict]] = defaultdict(list)

    def add(self, user_message: str, ai_message: str, session_id: str = "default"):
        """
        添加对话

        Args:
            user_message: 用户消息
            ai_message: AI回复
            session_id: 会话ID
        """
        conversation = self.conversations[session_id]

        conversation.append({
            "role": "user",
            "content": user_message,
            "timestamp": self._get_timestamp()
        })

        conversation.append({
            "role": "assistant",
            "content": ai_message,
            "timestamp": self._get_timestamp()
        })

        # 超过最大轮数时删除最早的对话
        if len(conversation) > self.max_turns * 2:
            self.conversations[session_id] = conversation[-(self.max_turns * 2):]

    def get_history(self, session_id: str = "default") -> List[Dict]:
        """获取对话历史"""
        return self.conversations.get(session_id, [])

    def clear(self, session_id: str = "default"):
        """清除对话历史"""
        if session_id in self.conversations:
            del self.conversations[session_id]

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
