"""
情景记忆
存储和检索过去的交互经历
"""
from typing import Dict, List
from datetime import datetime
import json
import os


class EpisodicMemory:
    """情景记忆"""

    def __init__(self, storage_path: str = "data/episodic_memory.json"):
        """
        初始化情景记忆

        Args:
            storage_path: 存储路径
        """
        self.storage_path = storage_path
        self.episodes: Dict[str, List[Dict]] = {}
        self._load()

    def _load(self):
        """加载记忆"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.episodes = json.load(f)
            except Exception:
                self.episodes = {}

    def _save(self):
        """保存记忆"""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.episodes, f, ensure_ascii=False, indent=2)

    def save_interaction(self, user_message: str, ai_message: str, session_id: str = "default"):
        """
        保存交互记录

        Args:
            user_message: 用户消息
            ai_message: AI回复
            session_id: 会话ID
        """
        if session_id not in self.episodes:
            self.episodes[session_id] = []

        episode = {
            "user_message": user_message,
            "ai_message": ai_message,
            "timestamp": datetime.now().isoformat(),
            "keywords": self._extract_keywords(user_message)
        }

        self.episodes[session_id].append(episode)
        self._save()

    def search(self, query: str, session_id: str = "default", top_k: int = 3) -> List[Dict]:
        """
        搜索相关交互

        Args:
            query: 查询文本
            session_id: 会话ID
            top_k: 返回数量

        Returns:
            相关交互列表
        """
        episodes = self.episodes.get(session_id, [])

        if not episodes:
            return []

        # 简单的关键词匹配
        query_keywords = set(self._extract_keywords(query))

        scored_episodes = []
        for episode in episodes:
            episode_keywords = set(episode.get("keywords", []))
            score = len(query_keywords & episode_keywords)
            if score > 0:
                scored_episodes.append((score, episode))

        # 按分数排序
        scored_episodes.sort(key=lambda x: x[0], reverse=True)

        return [episode for _, episode in scored_episodes[:top_k]]

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（简化版）"""
        # 简单的分词
        words = text.split()
        # 过滤停用词
        stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
        keywords = [w for w in words if w not in stop_words and len(w) > 1]
        return keywords
