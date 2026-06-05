"""
旅行AI助手 - 主应用入口
FastAPI应用，提供REST API和WebSocket接口
"""
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import json
import uuid

from app.config import settings
from core.agent import TravelAgent
from core.artifacts import ChatArtifacts
from core.trace import ExecutionTrace


# 全局Agent实例
agent = None


class ChatRequest(BaseModel):
    """聊天请求协议"""
    message: str = Field(..., description="用户消息")
    session_id: str | None = Field(default=None, description="会话ID")


class ChatResponse(BaseModel):
    """聊天响应协议"""
    session_id: str
    response: str
    artifacts: ChatArtifacts = Field(default_factory=ChatArtifacts)
    execution_trace: ExecutionTrace = Field(default_factory=ExecutionTrace)


class APIErrorPayload(BaseModel):
    """统一错误对象，保留detail字段兼容FastAPI默认错误。"""
    code: str
    message: str
    recoverable: bool = True


class APIErrorResponse(BaseModel):
    """统一错误响应协议。"""
    detail: Any
    error: APIErrorPayload


class HistoryMessage(BaseModel):
    """历史消息协议。"""
    role: str
    content: str
    timestamp: str | None = None
    run_id: str | None = None
    source: str | None = None


class HistoryResponse(BaseModel):
    """历史消息响应协议。"""
    session_id: str
    history: list[HistoryMessage] = Field(default_factory=list)


class TaskSummaryItem(BaseModel):
    """持久化任务摘要协议。"""
    task_id: str | None = None
    task_type: str | None = None
    tool: str | None = None
    name: str | None = None
    success: bool
    error: str | None = None
    duration_ms: int | None = None
    execution_mode: str | None = None
    degraded: bool | None = None
    fallback_used: bool | None = None
    result_summary: str | None = None


class SessionRunRecord(BaseModel):
    """会话运行记录协议。"""
    run_id: str
    session_id: str
    user_message: str
    ai_message: str
    intent_type: str | None = None
    task_count: int = 0
    failed_count: int = 0
    artifact_keys: list[str] = Field(default_factory=list)
    trace_summary: dict[str, Any] = Field(default_factory=dict)
    artifacts: ChatArtifacts = Field(default_factory=ChatArtifacts)
    execution_trace: ExecutionTrace = Field(default_factory=ExecutionTrace)
    task_summary: list[TaskSummaryItem] = Field(default_factory=list)
    created_at: str


class SessionRunsResponse(BaseModel):
    """会话运行历史响应协议。"""
    session_id: str
    runs: list[SessionRunRecord] = Field(default_factory=list)
    count: int


class ClearHistoryResponse(BaseModel):
    """清理历史响应协议。"""
    message: str


class ExternalServiceStatus(BaseModel):
    """外部服务状态"""
    name: str
    provider: str
    capability: str
    api_key_configured: bool
    key_source: str | None = None
    mock_enabled: bool
    mode: str
    probe_type: str = "configuration"


class ExternalStatusSummary(BaseModel):
    """外部服务状态汇总。"""
    total: int
    real_api_count: int
    mock_fallback_count: int
    unavailable_count: int
    all_operational: bool


class ExternalStatusResponse(BaseModel):
    """外部服务状态响应"""
    services: list[ExternalServiceStatus]
    summary: ExternalStatusSummary


class LLMStatusResponse(BaseModel):
    """LLM运行状态响应，不暴露真实密钥"""
    provider: str
    model: str
    base_url: str
    api_key_configured: bool
    key_source: str | None = None
    mode: str
    fallback_enabled: bool = True
    openai_compatible: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global agent
    # 启动时初始化Agent
    agent = TravelAgent()
    print(f"旅行AI助手 v{settings.APP_VERSION} 启动成功")
    yield
    # 关闭时清理资源
    print("旅行AI助手关闭")


# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """返回兼容FastAPI detail字段的统一HTTP错误结构。"""
    message = str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error": {
                "code": "http_error",
                "message": message,
                "recoverable": exc.status_code < 500,
            },
        },
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """返回稳定的参数校验错误结构，同时保留detail列表。"""
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "error": {
                "code": "validation_error",
                "message": "请求参数校验失败",
                "recoverable": True,
            },
        },
    )


@app.get("/")
async def root():
    """健康检查"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
    responses={400: {"model": APIErrorResponse}, 422: {"model": APIErrorResponse}},
)
async def chat(request: ChatRequest):
    """普通对话接口"""
    message = request.message.strip()
    session_id = (request.session_id or "").strip() or str(uuid.uuid4())

    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    # 调用Agent处理
    if hasattr(agent, "arun_with_artifacts"):
        result = await agent.arun_with_artifacts(message, session_id)
        response = result.get("response", "处理完成")
        artifacts = result.get("artifacts", {})
        execution_trace = result.get("execution_trace", {})
    else:
        response = await agent.arun(message, session_id)
        artifacts = {}
        execution_trace = {}

    return ChatResponse(session_id=session_id, response=response, artifacts=artifacts, execution_trace=execution_trace)


@app.get("/api/external/status", response_model=ExternalStatusResponse)
async def external_status():
    """查看外部API配置状态，不暴露真实密钥"""
    services = [
        _build_external_service_status(
            name="amap_poi",
            provider="amap",
            capability="poi_search",
            api_key_configured=bool(settings.AMAP_API_KEY),
            key_source="AMAP_API_KEY" if settings.AMAP_API_KEY else None,
        ),
        _build_external_service_status(
            name="amap_route",
            provider="amap",
            capability="route_distance",
            api_key_configured=bool(settings.AMAP_API_KEY),
            key_source="AMAP_API_KEY" if settings.AMAP_API_KEY else None,
        ),
        _build_external_service_status(
            name="weather",
            provider="amap",
            capability="weather_forecast",
            api_key_configured=bool(settings.WEATHER_API_KEY or settings.AMAP_API_KEY),
            key_source=_weather_key_source(),
        ),
    ]
    unavailable_count = sum(1 for service in services if service.mode == "unavailable")
    mock_count = sum(1 for service in services if service.mode == "mock_fallback")
    real_api_count = sum(1 for service in services if service.mode == "real_api")
    return ExternalStatusResponse(
        services=services,
        summary={
            "total": len(services),
            "real_api_count": real_api_count,
            "mock_fallback_count": mock_count,
            "unavailable_count": unavailable_count,
            "all_operational": unavailable_count == 0,
        },
    )


@app.get("/api/llm/status", response_model=LLMStatusResponse)
async def llm_status():
    """查看LLM配置状态，不暴露真实密钥"""
    configured = bool(settings.LLM_API_KEY and settings.LLM_BASE_URL)
    return LLMStatusResponse(
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
        base_url=settings.LLM_BASE_URL,
        api_key_configured=configured,
        key_source="LLM_API_KEY" if settings.LLM_API_KEY else None,
        mode="real_llm" if configured else "rule_fallback",
    )


def _build_external_service_status(
    name: str,
    provider: str,
    capability: str,
    api_key_configured: bool,
    key_source: str | None,
) -> ExternalServiceStatus:
    """构建外部服务配置状态"""
    return ExternalServiceStatus(
        name=name,
        provider=provider,
        capability=capability,
        api_key_configured=api_key_configured,
        key_source=key_source,
        mock_enabled=settings.EXTERNAL_API_MOCK_ENABLED,
        mode=_resolve_external_mode(api_key_configured),
    )


def _resolve_external_mode(api_key_configured: bool) -> str:
    """根据Key和mock配置判断外部服务模式"""
    if api_key_configured:
        return "real_api"
    if settings.EXTERNAL_API_MOCK_ENABLED:
        return "mock_fallback"
    return "unavailable"


def _weather_key_source() -> str | None:
    """天气服务Key来源，不返回真实Key"""
    if settings.WEATHER_API_KEY:
        return "WEATHER_API_KEY"
    if settings.AMAP_API_KEY:
        return "AMAP_API_KEY"
    return None


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket实时对话接口"""
    await websocket.accept()
    session_id = str(uuid.uuid4())

    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message = message_data.get("message", "")

            if not message:
                await websocket.send_json({"error": "消息不能为空"})
                continue

            # 流式处理
            async for chunk in agent.astream(message, session_id):
                await websocket.send_json({
                    "type": "chunk",
                    "content": chunk
                })

            # 发送完成信号
            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        print(f"客户端断开连接: {session_id}")
    except Exception as e:
        await websocket.send_json({"error": str(e)})


@app.get("/api/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """获取对话历史"""
    history = agent.get_history(session_id)
    return HistoryResponse(session_id=session_id, history=history)


@app.get("/api/history/{session_id}/runs", response_model=SessionRunsResponse)
async def get_session_runs(session_id: str, limit: int = 20):
    """获取会话运行历史，包括artifact和execution trace"""
    runs = agent.get_session_runs(session_id, limit=limit)
    return SessionRunsResponse(session_id=session_id, runs=runs, count=len(runs))


@app.delete("/api/history/{session_id}", response_model=ClearHistoryResponse)
async def clear_history(session_id: str):
    """清除对话历史"""
    agent.clear_history(session_id)
    return ClearHistoryResponse(message="历史记录已清除")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
