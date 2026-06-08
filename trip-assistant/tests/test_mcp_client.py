"""
MCP客户端配置与安全降级测试
"""
from app.config import settings
from core.mcp_client import MCPClient, MCPServerConfig, create_amap_mcp_client, split_mcp_args


def test_split_mcp_args_supports_command_line_string():
    """MCP参数支持从环境变量中按命令行规则拆分"""
    assert split_mcp_args("-y @amap/amap-maps-mcp-server") == ["-y", "@amap/amap-maps-mcp-server"]


def test_mcp_client_unavailable_without_global_enable(monkeypatch):
    """全局禁用MCP时客户端不会启动外部stdio进程"""
    monkeypatch.setattr(settings, "MCP_ENABLED", False)
    client = MCPClient(MCPServerConfig(name="test", command="uvx", args=["mcp-server-12306"]))

    assert client.available is False


def test_mcp_client_can_be_configured_with_remote_sse_url(monkeypatch):
    """远程SSE MCP只需要URL即可判定为可用配置"""
    monkeypatch.setattr(settings, "MCP_ENABLED", True)
    client = MCPClient(MCPServerConfig(name="remote", command="", url="https://example.com/sse"))

    assert client.available is True


def test_amap_mcp_client_requires_amap_key(monkeypatch):
    """高德MCP客户端必须有高德Key才可用，避免无效进程调用"""
    monkeypatch.setattr(settings, "MCP_ENABLED", True)
    monkeypatch.setattr(settings, "MCP_AMAP_ENABLED", True)
    monkeypatch.setattr(settings, "AMAP_API_KEY", "")

    client = create_amap_mcp_client()

    assert client.available is False
