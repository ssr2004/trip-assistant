"""
应用配置管理
使用环境变量管理敏感信息
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""

    # 应用基础配置
    APP_NAME: str = "旅行AI助手"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # LLM配置
    LLM_PROVIDER: str = "deepseek"  # deepseek, glm, qwen, openai
    LLM_MODEL: str = "deepseek-chat"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_TEMPERATURE: float = 0.7

    # Embedding配置
    EMBEDDING_PROVIDER: str = "openai"  # openai, zhipu
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://api.openai.com/v1"

    # RAG配置
    RAG_TOP_K: int = 5
    RAG_SCORE_THRESHOLD: float = 0.5

    # 数据库配置
    DATABASE_URL: str = "sqlite:///./travel.db"

    # API配置
    AMADEUS_API_KEY: str = ""
    AMADEUS_API_SECRET: str = ""
    AMAP_API_KEY: str = ""  # 高德地图API

    # 记忆系统配置
    MEMORY_MAX_TURNS: int = 20  # 短期记忆最大轮数
    LONG_TERM_MEMORY_ENABLED: bool = True

    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings
