"""
Docker Compose配置测试
"""
from pathlib import Path


def test_docker_compose_includes_redis_service():
    """Docker Compose包含Redis服务编排"""
    compose_path = Path(__file__).resolve().parents[1] / "docker-compose.yml"

    content = compose_path.read_text(encoding="utf-8")

    assert "redis:" in content
    assert "redis:7-alpine" in content
    assert '"6379:6379"' in content
    assert "REDIS_URL=redis://redis:6379/0" in content
    assert "depends_on:" in content
