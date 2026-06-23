"""Minimal MCP stdio client wrapper for real travel data providers."""
from __future__ import annotations

import asyncio
import json
import os
import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import settings


@dataclass(frozen=True)
class MCPServerConfig:
    """Configuration for one MCP server."""

    name: str
    command: str
    args: List[str] = field(default_factory=list)
    url: str = ""
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    timeout: int = 20


class MCPClient:
    """Small wrapper around the official MCP Python SDK.

    默认复用一个常驻 MCP 会话（避免每次调用都 fork 一次 npx/stdio 子进程，
    把多次工具调用的首字延迟从秒级降到毫秒级）。会话启动或调用失败时，
    自动降级为"单次调用"模式（原行为），保证常驻会话异常时也不中断。
    返回值始终是脱敏 dict，工具不依赖 SDK 模型类。
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        # 常驻会话状态（惰性启动，跨调用复用）
        self._session: Any = None
        self._session_task: Optional[asyncio.Task] = None
        self._session_ready: Optional[asyncio.Event] = None
        self._session_stop: Optional[asyncio.Event] = None
        self._session_lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return bool(settings.MCP_ENABLED and self.config.enabled and (self.config.url or self.config.command))

    async def list_tools(self) -> Dict[str, Any]:
        if not self.available:
            return self._error("MCP server未启用或命令未配置。", "mcp_unavailable")
        return await self._run("list_tools")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available:
            return self._error("MCP server未启用或命令未配置。", "mcp_unavailable")
        return await self._run("call_tool", tool_name=tool_name, arguments=arguments or {})

    async def _run(self, action: str, tool_name: str = "", arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        timeout = max(int(self.config.timeout or settings.MCP_TIMEOUT), 1)
        # 1) 优先复用常驻会话（快，避免每次 fork 子进程）
        try:
            session = await asyncio.wait_for(self._get_session(), timeout=timeout)
            if session is not None:
                return await asyncio.wait_for(
                    self._invoke(session, action, tool_name, arguments or {}),
                    timeout=timeout,
                )
        except asyncio.TimeoutError:
            return self._error("MCP调用超时。", "timeout")
        except Exception:
            # 常驻会话异常 → 重置后走单次调用兜底
            await self._reset_session()
        # 2) 兜底：每次调用起一个临时会话（原行为）
        try:
            return await asyncio.wait_for(
                self._run_sdk(action, tool_name=tool_name, arguments=arguments or {}),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return self._error("MCP调用超时。", "timeout")
        except Exception as exc:
            return self._error(self._sanitize_error(exc), self._classify_error(exc))

    async def _run_sdk(self, action: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.url:
            return await self._run_sse_sdk(action, tool_name, arguments)
        return await self._run_stdio_sdk(action, tool_name, arguments)

    async def _invoke(self, session: Any, action: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """在已初始化的 ClientSession 上执行 list_tools / call_tool。"""
        if action == "list_tools":
            tools = await session.list_tools()
            return self._success(
                {
                    "tools": [
                        {
                            "name": tool.name,
                            "description": getattr(tool, "description", None),
                            "input_schema": getattr(tool, "inputSchema", None),
                        }
                        for tool in tools.tools
                    ]
                },
                tool_name="list_tools",
            )
        result = await session.call_tool(tool_name, arguments)
        return self._success(
            self._normalize_call_result(result),
            tool_name=tool_name,
            is_error=bool(getattr(result, "isError", False)),
        )

    async def _get_session(self) -> Optional[Any]:
        """惰性启动并返回常驻 MCP 会话；不可用或启动失败时返回 None（由调用方走兜底）。"""
        if not self.available:
            return None
        if self._session is not None:
            return self._session
        async with self._session_lock:
            if self._session is not None:
                return self._session
            self._session_ready = asyncio.Event()
            self._session_stop = asyncio.Event()
            self._session = None
            self._session_task = asyncio.create_task(self._session_runner())
            await self._session_ready.wait()
        return self._session

    async def _session_runner(self) -> None:
        """后台任务：持有 transport + ClientSession 打开，直到 _session_stop 被设置。"""
        try:
            async with self._open_transport() as (read, write):
                from mcp import ClientSession

                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._session_ready.set()
                    await self._session_stop.wait()
        except Exception:
            self._session = None
            if self._session_ready is not None:
                self._session_ready.set()

    def _open_transport(self):
        """返回 stdio / sse transport 的 async context manager。"""
        if self.config.url:
            from mcp.client.sse import sse_client

            timeout = max(int(self.config.timeout or settings.MCP_TIMEOUT), 1)
            return sse_client(self.config.url, timeout=timeout, sse_read_timeout=timeout)
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self._resolve_command(self.config.command),
            args=list(self.config.args),
            env=self._build_env(),
        )
        return stdio_client(params)

    async def _reset_session(self) -> None:
        """关闭并清理常驻会话，下次调用会重新启动。"""
        stop = self._session_stop
        task = self._session_task
        self._session = None
        self._session_stop = None
        self._session_ready = None
        self._session_task = None
        if stop is not None:
            stop.set()
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=3)
            except Exception:
                task.cancel()

    async def _run_stdio_sdk(self, action: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self._resolve_command(self.config.command),
            args=self.config.args,
            env=self._build_env(),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await self._invoke(session, action, tool_name, arguments)

    async def _run_sse_sdk(self, action: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        async with sse_client(
            self.config.url,
            timeout=max(int(self.config.timeout or settings.MCP_TIMEOUT), 1),
            sse_read_timeout=max(int(self.config.timeout or settings.MCP_TIMEOUT), 1),
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await self._invoke(session, action, tool_name, arguments)

    def _normalize_call_result(self, result: Any) -> Dict[str, Any]:
        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            return structured

        content = getattr(result, "content", None) or []
        text_values = []
        for item in content:
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text")
            if text:
                text_values.append(str(text))

        if len(text_values) == 1:
            parsed = self._parse_json_text(text_values[0])
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"items": parsed}
        return {"content": text_values}

    def _parse_json_text(self, text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return None

    def _build_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env.update({key: value for key, value in self.config.env.items() if value is not None})
        return env

    def _resolve_command(self, command: str) -> str:
        if os.name == "nt" and command == "npx":
            return "npx.cmd"
        return command

    def _success(self, data: Dict[str, Any], tool_name: str, is_error: bool = False) -> Dict[str, Any]:
        return {
            "success": not is_error,
            "data": data or {},
            "error": "MCP工具返回错误。" if is_error else None,
            "metadata": {
                "provider": self.config.name,
                "source": "mcp",
                "tool_name": tool_name,
                "api_status": "failed" if is_error else "success",
                "execution_mode": "real_mcp_failed" if is_error else "real_mcp",
                "fallback_used": False,
                "mock": False,
            },
        }

    def _error(self, error: str, error_type: str) -> Dict[str, Any]:
        return {
            "success": False,
            "data": {},
            "error": error,
            "metadata": {
                "provider": self.config.name,
                "source": "mcp",
                "api_status": "unavailable" if error_type == "mcp_unavailable" else "failed",
                "execution_mode": "unavailable" if error_type == "mcp_unavailable" else "real_mcp_failed",
                "error_type": error_type,
                "fallback_used": False,
                "mock": False,
            },
        }

    def _sanitize_error(self, exc: Exception) -> str:
        message = str(exc) or exc.__class__.__name__
        for secret in [settings.AMAP_API_KEY, settings.LLM_API_KEY, settings.EMBEDDING_API_KEY]:
            if secret:
                message = message.replace(secret, "***")
        return message

    def _classify_error(self, exc: Exception) -> str:
        name = exc.__class__.__name__.lower()
        message = str(exc).lower()
        if "timeout" in name or "timeout" in message:
            return "timeout"
        if "not found" in message or "no such file" in message:
            return "command_not_found"
        if "permission" in message or "access" in message:
            return "permission"
        return "mcp_error"


def split_mcp_args(value: str) -> List[str]:
    """Split env-configured MCP args using shell-like rules."""
    if not value:
        return []
    return shlex.split(value, posix=False if os.name == "nt" else True)


def create_12306_mcp_client() -> MCPClient:
    return MCPClient(
        MCPServerConfig(
            name="mcp_12306",
            command=settings.MCP_12306_COMMAND,
            args=split_mcp_args(settings.MCP_12306_ARGS),
            url=settings.MCP_12306_URL,
            enabled=bool(settings.MCP_12306_ENABLED),
            timeout=settings.MCP_TIMEOUT,
        )
    )


def create_amap_mcp_client() -> MCPClient:
    return MCPClient(
        MCPServerConfig(
            name="mcp_amap",
            command=settings.MCP_AMAP_COMMAND,
            args=split_mcp_args(settings.MCP_AMAP_ARGS),
            url=settings.MCP_AMAP_URL,
            env={"AMAP_MAPS_API_KEY": settings.AMAP_API_KEY},
            enabled=bool(settings.MCP_AMAP_ENABLED and (settings.AMAP_API_KEY or settings.MCP_AMAP_URL)),
            timeout=settings.MCP_TIMEOUT,
        )
    )
