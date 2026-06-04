"""
外部API配置治理测试
"""
from pathlib import Path

from app.config import Settings


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_external_api_settings_defaults():
    """外部API通用配置默认值合理"""
    settings = Settings(_env_file=None)

    assert settings.EXTERNAL_API_TIMEOUT == 10
    assert settings.EXTERNAL_API_RETRY_TIMES == 2
    assert settings.EXTERNAL_API_MOCK_ENABLED is True
    assert settings.AMADEUS_API_KEY == ""
    assert settings.AMADEUS_API_SECRET == ""
    assert settings.AMAP_API_KEY == ""
    assert settings.WEATHER_API_KEY == ""


def test_env_example_contains_external_api_placeholders():
    """环境变量示例文件包含外部API占位，不包含真实Key"""
    content = (PROJECT_DIR / ".env.example").read_text(encoding="utf-8")

    assert "EXTERNAL_API_TIMEOUT=10" in content
    assert "EXTERNAL_API_RETRY_TIMES=2" in content
    assert "EXTERNAL_API_MOCK_ENABLED=true" in content
    assert "AMADEUS_API_KEY=" in content
    assert "AMADEUS_API_SECRET=" in content
    assert "AMAP_API_KEY=" in content
    assert "WEATHER_API_KEY=" in content
    assert "不要提交真实 API Key" in content
