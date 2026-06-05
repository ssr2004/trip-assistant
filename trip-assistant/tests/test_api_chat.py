"""
API聊天接口测试
"""
from uuid import UUID

from fastapi.testclient import TestClient

from app import main as app_main


class FakeAgent:
    """用于API测试的轻量Agent替身"""

    def __init__(self):
        self.calls = []
        self.cleared_sessions = []

    async def arun(self, message: str, session_id: str) -> str:
        self.calls.append({"message": message, "session_id": session_id})
        return f"已处理：{message}"

    def get_history(self, session_id: str):
        return [
            {"role": "user", "content": "你好", "timestamp": "2026-06-05T12:00:00"},
            {"role": "assistant", "content": "你好，我是旅行助手", "timestamp": "2026-06-05T12:00:01"},
        ]

    def get_session_runs(self, session_id: str, limit: int = 20):
        return [
            {
                "run_id": "run-1",
                "session_id": session_id,
                "user_message": "你好",
                "ai_message": "你好，我是旅行助手",
                "intent_type": "general_chat",
                "task_count": 0,
                "failed_count": 0,
                "artifact_keys": [],
                "trace_summary": {"intent": "general_chat"},
                "artifacts": {},
                "execution_trace": {"steps": [], "summary": {"intent": "general_chat"}},
                "task_summary": [],
                "created_at": "2026-06-05T12:00:01",
            }
        ][:limit]

    def clear_history(self, session_id: str):
        self.cleared_sessions.append(session_id)


class FakeArtifactAgent(FakeAgent):
    """支持结构化结果的测试Agent替身"""

    async def arun_with_artifacts(self, message: str, session_id: str):
        self.calls.append({"message": message, "session_id": session_id})
        return {
            "response": f"已处理：{message}",
            "artifacts": {
                "itinerary": {
                    "destination": "杭州",
                    "days": [{"day": 1, "title": "西湖初体验", "activities": ["西湖"]}],
                }
            },
        }


def make_client(monkeypatch):
    fake_agent = FakeAgent()
    monkeypatch.setattr(app_main, "agent", fake_agent)
    return TestClient(app_main.app), fake_agent


def test_chat_generates_session_id_when_missing(monkeypatch):
    """无session_id时自动生成并返回"""
    client, fake_agent = make_client(monkeypatch)

    response = client.post("/api/chat", json={"message": "杭州有什么好玩的"})

    assert response.status_code == 200
    data = response.json()
    UUID(data["session_id"])
    assert data["artifacts"] == {}
    assert data["execution_trace"] == {"steps": [], "summary": {}}
    assert data["response"] == "已处理：杭州有什么好玩的"
    assert fake_agent.calls == [
        {"message": "杭州有什么好玩的", "session_id": data["session_id"]}
    ]


def test_chat_reuses_provided_session_id(monkeypatch):
    """有session_id时保持复用"""
    client, fake_agent = make_client(monkeypatch)

    response = client.post(
        "/api/chat",
        json={"message": "如果下雨怎么办？", "session_id": "session-module26"},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "session-module26"
    assert fake_agent.calls == [
        {"message": "如果下雨怎么办？", "session_id": "session-module26"}
    ]


def test_chat_reuses_session_across_requests(monkeypatch):
    """连续请求可以复用同一会话上下文"""
    client, fake_agent = make_client(monkeypatch)

    first = client.post("/api/chat", json={"message": "我要从郑州去杭州玩三天"})
    session_id = first.json()["session_id"]
    second = client.post(
        "/api/chat",
        json={"message": "如果下雨怎么办？", "session_id": session_id},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id
    assert [call["session_id"] for call in fake_agent.calls] == [session_id, session_id]


def test_chat_rejects_empty_message(monkeypatch):
    """空消息返回明确错误且不调用Agent"""
    client, fake_agent = make_client(monkeypatch)

    response = client.post("/api/chat", json={"message": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "消息不能为空"
    assert response.json()["error"] == {
        "code": "http_error",
        "message": "消息不能为空",
        "recoverable": True,
    }
    assert fake_agent.calls == []


def test_chat_returns_artifacts_when_agent_provides_them(monkeypatch):
    """聊天接口透传Agent结构化展示数据"""
    fake_agent = FakeArtifactAgent()
    monkeypatch.setattr(app_main, "agent", fake_agent)
    client = TestClient(app_main.app)

    response = client.post("/api/chat", json={"message": "生成行程", "session_id": "artifact-session"})

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "已处理：生成行程"
    assert data["artifacts"]["itinerary"]["destination"] == "杭州"
    assert data["artifacts"]["itinerary"]["days"][0]["title"] == "西湖初体验"


def test_chat_openapi_uses_structured_artifact_schema(monkeypatch):
    """OpenAPI exposes artifacts as a typed contract instead of a free-form dict."""
    client, _ = make_client(monkeypatch)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    schemas = response.json()["components"]["schemas"]
    chat_response = schemas["ChatResponse"]
    artifacts_schema = chat_response["properties"]["artifacts"]
    assert artifacts_schema["$ref"].endswith("/ChatArtifacts")
    assert chat_response["properties"]["execution_trace"]["$ref"].endswith("/ExecutionTrace")
    assert "ItineraryArtifact" in schemas
    assert "RouteArtifact" in schemas
    assert "TraceStep" in schemas
    assert "HistoryResponse" in schemas
    assert "SessionRunsResponse" in schemas
    assert "APIErrorResponse" in schemas
    paths = response.json()["paths"]
    assert paths["/api/history/{session_id}"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/HistoryResponse")
    assert paths["/api/history/{session_id}/runs"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/SessionRunsResponse")


def test_history_api_uses_typed_message_contract(monkeypatch):
    """历史消息接口返回稳定结构。"""
    client, _ = make_client(monkeypatch)

    response = client.get("/api/history/session-contract")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "session-contract"
    assert data["history"][0]["role"] == "user"
    assert data["history"][0]["content"] == "你好"
    assert data["history"][1]["role"] == "assistant"


def test_session_runs_api_uses_typed_runtime_contract(monkeypatch):
    """运行历史接口返回artifact、trace和task summary边界。"""
    client, _ = make_client(monkeypatch)

    response = client.get("/api/history/session-contract/runs?limit=1")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "session-contract"
    assert data["count"] == 1
    run = data["runs"][0]
    assert run["run_id"] == "run-1"
    assert run["execution_trace"]["summary"]["intent"] == "general_chat"
    assert run["task_summary"] == []


def test_clear_history_api_uses_typed_response(monkeypatch):
    """清理历史接口返回稳定响应，并调用Agent清理。"""
    client, fake_agent = make_client(monkeypatch)

    response = client.delete("/api/history/session-contract")

    assert response.status_code == 200
    assert response.json() == {"message": "历史记录已清除"}
    assert fake_agent.cleared_sessions == ["session-contract"]
