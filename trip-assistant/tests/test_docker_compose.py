"""
Docker与本地工程化配置测试
"""
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_docker_compose_includes_redis_service():
    """Docker Compose包含Redis服务编排"""
    content = (PROJECT_DIR / "docker-compose.yml").read_text(encoding="utf-8")

    assert "redis:" in content
    assert "redis:7-alpine" in content
    assert "REDIS_URL=redis://redis:6379/0" in content
    assert "depends_on:" in content
    assert 'expose:\n      - "6379"' in content
    assert '"6379:6379"' not in content


def test_docker_compose_uses_env_file_without_mounting_secret_file():
    """Compose通过env_file注入配置，不把.env作为文件挂载进镜像。"""
    content = (PROJECT_DIR / "docker-compose.yml").read_text(encoding="utf-8")

    assert "env_file:" in content
    assert "- .env" in content
    assert "./.env:/app/.env" not in content
    assert "DATABASE_URL=sqlite:///./data/travelmind.db" in content
    assert "EXTERNAL_API_CACHE_BACKEND=redis" in content
    assert "./data:/app/data" in content


def test_dockerfile_uses_backend_runtime_healthcheck_and_non_root_user():
    """Dockerfile具备后端启动命令、健康检查和非root运行用户。"""
    content = (PROJECT_DIR / "Dockerfile").read_text(encoding="utf-8")

    assert "ARG PYTHON_IMAGE=python:3.11-slim" in content
    assert "PYTHONUNBUFFERED=1" in content
    assert "pip install --no-cache-dir -r requirements.txt" in content
    assert "useradd --create-home" in content
    assert "USER appuser" in content
    assert "HEALTHCHECK" in content
    assert "urllib.request.urlopen('http://127.0.0.1:8000/'" in content
    assert 'CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]' in content


def test_dockerignore_excludes_secrets_runtime_data_and_dev_artifacts():
    """Docker构建上下文排除密钥、运行数据和开发产物。"""
    content = (PROJECT_DIR / ".dockerignore").read_text(encoding="utf-8")

    required_patterns = [
        ".env",
        ".env.*",
        "!.env.example",
        ".venv/",
        "data/",
        "*.db",
        "local-docs/",
        "tests/",
        "frontend/",
        "node_modules/",
        "playwright-report/",
        "test-results/",
    ]
    for pattern in required_patterns:
        assert pattern in content


def test_env_example_covers_runtime_and_provider_configuration_without_real_keys():
    """环境变量样例覆盖运行配置和外部Provider配置，但不包含真实Key。"""
    content = (PROJECT_DIR / ".env.example").read_text(encoding="utf-8")

    required_keys = [
        "LLM_API_KEY=",
        "LLM_BASE_URL=",
        "LLM_PLANNER_ENABLED=",
        "LLM_PLANNER_MODE=",
        "EMBEDDING_API_KEY=",
        "EMBEDDING_BASE_URL=",
        "DATABASE_URL=",
        "EXTERNAL_API_CACHE_BACKEND=",
        "REDIS_URL=",
        "AMADEUS_API_KEY=",
        "AMADEUS_API_SECRET=",
        "AMAP_API_KEY=",
        "WEATHER_API_KEY=",
    ]
    for key in required_keys:
        assert key in content
    assert "sk-" not in content
    assert "真实 API Key" in content


def test_readme_mentions_quality_gate_and_safe_docker_startup():
    """README包含质量门禁和安全Docker启动说明。"""
    content = (PROJECT_DIR / "README.md").read_text(encoding="utf-8")

    assert "cp .env.example .env" in content
    assert "docker-compose up -d" in content
    assert "Redis 仅在 compose 内部网络暴露" in content
    assert "scripts\\run_quality_gate.py" in content
    assert "--json-compact" in content
