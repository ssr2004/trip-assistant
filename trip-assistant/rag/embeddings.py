"""
嵌入模型管理
提供文本向量化功能
"""
from typing import List
import os

from app.config import settings


class EmbeddingManager:
    """嵌入模型管理器"""

    def __init__(self):
        """初始化嵌入模型"""
        self.provider = settings.EMBEDDING_PROVIDER
        self.model = settings.EMBEDDING_MODEL
        self.api_key = settings.EMBEDDING_API_KEY
        self.base_url = settings.EMBEDDING_BASE_URL

        # 缓存
        self._cache = {}

    def embed(self, text: str) -> List[float]:
        """
        获取文本嵌入向量

        Args:
            text: 输入文本

        Returns:
            嵌入向量
        """
        # 检查缓存
        if text in self._cache:
            return self._cache[text]

        # 根据provider调用不同的API
        if self.provider == "openai":
            embedding = self._openai_embed(text)
        elif self.provider == "zhipu":
            embedding = self._zhipu_embed(text)
        else:
            embedding = self._local_embed(text)

        # 缓存结果
        self._cache[text] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        return [self.embed(text) for text in texts]

    def _openai_embed(self, text: str) -> List[float]:
        """OpenAI嵌入"""
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            response = client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"OpenAI嵌入失败: {e}")
            return self._fallback_embed(text)

    def _zhipu_embed(self, text: str) -> List[float]:
        """智谱AI嵌入"""
        try:
            from zhipuai import ZhipuAI
            client = ZhipuAI(api_key=self.api_key)
            response = client.embeddings.create(
                model="embedding-3",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"智谱AI嵌入失败: {e}")
            return self._fallback_embed(text)

    def _local_embed(self, text: str) -> List[float]:
        """本地嵌入（使用sentence-transformers）"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            return model.encode(text).tolist()
        except Exception as e:
            print(f"本地嵌入失败: {e}")
            return self._fallback_embed(text)

    def _fallback_embed(self, text: str) -> List[float]:
        """降级嵌入（随机向量）"""
        import random
        # 生成随机向量作为降级方案
        return [random.random() for _ in range(384)]
