"""
测试全局配置
"""
import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def isolate_real_external_api_keys(monkeypatch):
    """自动化测试默认不读取本地真实外部API Key"""
    monkeypatch.setattr(settings, "AMADEUS_API_KEY", "")
    monkeypatch.setattr(settings, "AMADEUS_API_SECRET", "")
    monkeypatch.setattr(settings, "AMAP_API_KEY", "")
    monkeypatch.setattr(settings, "WEATHER_API_KEY", "")
    monkeypatch.setattr(settings, "EMBEDDING_API_KEY", "")
    monkeypatch.setattr(settings, "LLM_API_KEY", "")
    monkeypatch.setattr(settings, "DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setattr(settings, "MCP_ENABLED", False)
    monkeypatch.setattr(settings, "MCP_12306_ENABLED", False)
    monkeypatch.setattr(settings, "MCP_AMAP_ENABLED", False)
    monkeypatch.setattr(settings, "MCP_FLIGHT_ENABLED", False)
    monkeypatch.setattr(settings, "TAVILY_SEARCH_ENABLED", False)
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "")
