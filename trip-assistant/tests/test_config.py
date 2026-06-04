"""
应用配置模块测试
"""
from app.config import Settings


def test_settings_use_pydantic_v2_model_config():
    """配置模块使用 Pydantic V2 model_config 写法"""
    assert Settings.model_config["env_file"] == ".env"
    assert Settings.model_config["env_file_encoding"] == "utf-8"


def test_settings_defaults_are_project_aligned():
    """默认配置与当前项目命名和本地数据目录保持一致"""
    settings = Settings(_env_file=None)

    assert settings.APP_NAME == "TravelMind"
    assert settings.APP_ENV == "development"
    assert settings.DATABASE_URL == "sqlite:///./data/travelmind.db"
