"""
LLM状态接口测试
"""
from fastapi.testclient import TestClient

from app import main as app_main
from app.config import settings


def make_client(monkeypatch, api_key="", base_url="https://api.deepseek.com"):
    """构建带可控LLM配置的测试客户端"""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(settings, "LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(settings, "LLM_API_KEY", api_key)
    monkeypatch.setattr(settings, "LLM_BASE_URL", base_url)
    return TestClient(app_main.app)


def test_llm_status_reports_rule_fallback_without_key(monkeypatch):
    """无LLM Key时状态接口展示规则降级模式"""
    client = make_client(monkeypatch, api_key="")

    response = client.get("/api/llm/status")

    assert response.status_code == 200
    data = response.json()
    assert data == {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key_configured": False,
        "key_source": None,
        "mode": "rule_fallback",
        "fallback_enabled": True,
        "openai_compatible": True,
    }


def test_llm_status_reports_real_llm_without_exposing_key(monkeypatch):
    """配置真实Key后展示真实LLM模式，但不泄露Key值"""
    client = make_client(monkeypatch, api_key="sk-deepseek-secret")

    response = client.get("/api/llm/status")

    assert response.status_code == 200
    data = response.json()
    assert data["api_key_configured"] is True
    assert data["key_source"] == "LLM_API_KEY"
    assert data["mode"] == "real_llm"
    assert data["provider"] == "deepseek"
    assert data["model"] == "deepseek-v4-flash"
    assert "sk-deepseek-secret" not in response.text
