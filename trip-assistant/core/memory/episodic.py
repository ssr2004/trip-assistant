"""
情景记忆
存储和检索过去的交互经历，基于语义相似度检索。
"""
from typing import Dict, List, Optional
from datetime import datetime
import json
import math
import os

from rag.embeddings import EmbeddingManager


class EpisodicMemory:
    """情景记忆"""

    def __init__(
        self,
        storage_path: str = "data/episodic_memory.json",
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        """
        初始化情景记忆

        Args:
            storage_path: 存储路径
            embedding_manager: 嵌入管理器，用于语义检索；默认创建。
        """
        self.storage_path = storage_path
        self.embedding_manager = embedding_manager or EmbeddingManager()
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
            "keywords": self._extract_keywords(user_message),
        }

        self.episodes[session_id].append(episode)
        self._save()

    def search(self, query: str, session_id: str = "default", top_k: int = 3) -> List[Dict]:
        """
        语义检索相关交互。

        用 embedding 余弦相似度检索，替代原 ``split()`` 关键词匹配——
        原实现按空白分词，对中文整句视为一个 token，检索基本失效。
        无真实 Key 时 EmbeddingManager 走确定性降级，仍优于空白分词。

        Args:
            query: 查询文本
            session_id: 会话ID
            top_k: 返回数量

        Returns:
            相关交互列表（按相似度降序）
        """
        episodes = self.episodes.get(session_id, [])
        if not episodes:
            return []

        episode_texts = [
            f"{ep.get('user_message', '')} {ep.get('ai_message', '')}".strip()
            for ep in episodes
        ]
        embeddings = self.embedding_manager.embed_batch([query] + episode_texts)
        query_embedding = embeddings[0]

        scored = []
        for episode, embedding in zip(episodes, embeddings[1:]):
            score = self._cosine_similarity(query_embedding, embedding)
            if score > 0:
                scored.append((score, episode))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [episode for _, episode in scored[:top_k]]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度，兼容空向量和维度异常。"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if not norm1 or not norm2:
            return 0.0
        return dot / (norm1 * norm2)

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（保留供调试/元数据使用）。"""
        words = text.split()
        stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
        return [w for w in words if w not in stop_words and len(w) > 1]
