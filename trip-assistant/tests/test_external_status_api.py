"""
外部API状态接口测试
"""
from fastapi.testclient import TestClient

from app import main as app_main
from app.config import settings


def make_client(monkeypatch, amap_key="", weather_key="", mock_enabled=True):
    """构建带可控配置的测试客户端"""
    monkeypatch.setattr(settings, "AMAP_API_KEY", amap_key)
    monkeypatch.setattr(settings, "WEATHER_API_KEY", weather_key)
    monkeypatch.setattr(settings, "EXTERNAL_API_MOCK_ENABLED", mock_enabled)
    return TestClient(app_main.app)


def service_by_name(data, name):
    """按服务名读取状态"""
    return {service["name"]: service for service in data["services"]}[name]


def test_external_status_uses_mock_fallback_without_keys(monkeypatch):
    """无Key且mock开启时展示mock fallback模式"""
    client = make_client(monkeypatch, amap_key="", weather_key="", mock_enabled=True)

    response = client.get("/api/external/status")

    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == {
        "total": 3,
        "real_api_count": 0,
        "mock_fallback_count": 3,
        "unavailable_count": 0,
        "all_operational": True,
    }
    for service in data["services"]:
        assert service["api_key_configured"] is False
        assert service["key_source"] is None
        assert service["mock_enabled"] is True
        assert service["mode"] == "mock_fallback"
        assert service["probe_type"] == "configuration"


def test_external_status_reports_real_api_with_amap_key(monkeypatch):
    """配置高德Key后POI、路线和天气均展示真实API模式"""
    client = make_client(monkeypatch, amap_key="configured-key", weather_key="", mock_enabled=True)

    response = client.get("/api/external/status")

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["real_api_count"] == 3
    assert data["summary"]["mock_fallback_count"] == 0
    assert data["summary"]["all_operational"] is True
    assert service_by_name(data, "amap_poi")["key_source"] == "AMAP_API_KEY"
    assert service_by_name(data, "amap_route")["key_source"] == "AMAP_API_KEY"
    assert service_by_name(data, "weather")["key_source"] == "AMAP_API_KEY"
    for service in data["services"]:
        assert service["api_key_configured"] is True
        assert service["mode"] == "real_api"
        assert "configured-key" not in str(service)


def test_external_status_weather_key_can_override_amap_key(monkeypatch):
    """天气Key单独配置时仅天气展示真实API模式"""
    client = make_client(monkeypatch, amap_key="", weather_key="weather-key", mock_enabled=True)

    response = client.get("/api/external/status")

    data = response.json()
    assert data["summary"]["real_api_count"] == 1
    assert data["summary"]["mock_fallback_count"] == 2
    assert service_by_name(data, "weather")["mode"] == "real_api"
    assert service_by_name(data, "weather")["key_source"] == "WEATHER_API_KEY"
    assert service_by_name(data, "amap_poi")["mode"] == "mock_fallback"
    assert "weather-key" not in response.text


def test_external_status_reports_unavailable_without_keys_and_mock_disabled(monkeypatch):
    """无Key且mock关闭时展示不可用状态"""
    client = make_client(monkeypatch, amap_key="", weather_key="", mock_enabled=False)

    response = client.get("/api/external/status")

    data = response.json()
    assert data["summary"] == {
        "total": 3,
        "real_api_count": 0,
        "mock_fallback_count": 0,
        "unavailable_count": 3,
        "all_operational": False,
    }
    for service in data["services"]:
        assert service["mock_enabled"] is False
        assert service["mode"] == "unavailable"
