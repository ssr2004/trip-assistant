"""
应用配置管理
使用环境变量管理敏感信息
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 应用基础配置
    APP_NAME: str = "TravelMind"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # LLM配置
    LLM_PROVIDER: str = "deepseek"  # deepseek, glm, qwen, openai
    LLM_MODEL: str = "deepseek-v4-flash"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.deepseek.com"
    LLM_TEMPERATURE: float = 0.7
    LLM_PLANNER_MODE: str = "auto"  # auto, off, always
    LLM_PLANNER_ENABLED: bool = False
    LLM_PLANNER_COMPLEXITY_THRESHOLD: int = 3
    ITINERARY_LLM_ENABLED: bool = True

    # Embedding配置
    EMBEDDING_PROVIDER: str = "openai"  # openai, zhipu
    EMBEDDING_MODEL: str = "text-embedding-v4"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # RAG配置
    RAG_TOP_K: int = 5
    RAG_SCORE_THRESHOLD: float = 0.5
    RAG_MIN_HIT_SCORE: float = 0.55
    RAG_CHUNK_DEDUPE_THRESHOLD: float = 0.88
    RAG_MIN_CHUNK_LENGTH: int = 80

    # Tavily搜索增强配置
    TAVILY_SEARCH_ENABLED: bool = True
    TAVILY_API_KEY: str = ""
    TAVILY_MAX_RESULTS: int = 5
    TAVILY_INCLUDE_DOMAINS: str = ""
    TAVILY_SEARCH_DEPTH: str = "basic"
    TAVILY_INCLUDE_RAW_CONTENT: bool = True
    TAVILY_GUIDE_TTL_DAYS: int = 30

    # 数据库配置
    DATABASE_URL: str = "sqlite:///./data/travelmind.db"

    # 外部API通用配置
    EXTERNAL_API_TIMEOUT: int = 10
    EXTERNAL_API_RETRY_TIMES: int = 2
    EXTERNAL_API_MOCK_ENABLED: bool = True
    EXTERNAL_API_CACHE_ENABLED: bool = True
    EXTERNAL_API_CACHE_TTL: int = 3600
    EXTERNAL_API_CACHE_BACKEND: str = "redis"
    REDIS_URL: str = "redis://localhost:6379/0"

    # API配置
    AMADEUS_API_KEY: str = ""
    AMADEUS_API_SECRET: str = ""
    AMAP_API_KEY: str = ""  # 高德地图API
    WEATHER_API_KEY: str = ""

    # MCP真实数据源配置
    MCP_ENABLED: bool = True
    MCP_TIMEOUT: int = 20
    MCP_12306_ENABLED: bool = True
    MCP_12306_URL: str = ""
    MCP_12306_COMMAND: str = "uvx"
    MCP_12306_ARGS: str = "mcp-server-12306"
    MCP_AMAP_ENABLED: bool = True
    MCP_AMAP_URL: str = ""
    MCP_AMAP_COMMAND: str = "npx"
    MCP_AMAP_ARGS: str = "-y @amap/amap-maps-mcp-server"
    MCP_FLIGHT_ENABLED: bool = False
    MCP_FLIGHT_URL: str = ""
    MCP_FLIGHT_COMMAND: str = ""
    MCP_FLIGHT_ARGS: str = ""

    # 记忆系统配置
    MEMORY_MAX_TURNS: int = 20  # 短期记忆最大轮数
    LONG_TERM_MEMORY_ENABLED: bool = True

    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000



# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings
