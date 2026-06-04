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

    async def arun(self, message: str, session_id: str) -> str:
        self.calls.append({"message": message, "session_id": session_id})
        return f"已处理：{message}"


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
