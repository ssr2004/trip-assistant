"""
旅行AI助手 - 主应用入口
FastAPI应用，提供REST API和WebSocket接口
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
import uuid

from app.config import settings
from core.agent import TravelAgent


# 全局Agent实例
agent = None


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


@app.get("/")
async def root():
    """健康检查"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.post("/api/chat")
async def chat(request: dict):
    """普通对话接口"""
    message = request.get("message", "")
    session_id = request.get("session_id", str(uuid.uuid4()))

    if not message:
        return {"error": "消息不能为空"}

    # 调用Agent处理
    response = await agent.arun(message, session_id)

    return {
        "session_id": session_id,
        "response": response
    }


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


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """获取对话历史"""
    history = agent.get_history(session_id)
    return {"session_id": session_id, "history": history}


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    """清除对话历史"""
    agent.clear_history(session_id)
    return {"message": "历史记录已清除"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
