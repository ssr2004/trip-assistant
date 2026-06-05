"""Tests for persistent session history governance."""

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from core.agent import TravelAgent
from core.session_history import SessionHistoryStore


def test_session_history_store_persists_runs_and_reconstructs_messages(tmp_path):
    """会话运行记录可以持久化，并恢复为对话消息历史。"""
    store = SessionHistoryStore(database_url=f"sqlite:///{tmp_path / 'history.db'}")

    record = store.save_run(
        session_id="session-a",
        user_message="帮我规划杭州三天",
        ai_message="已生成杭州三天行程",
        artifacts={"itinerary": {"destination": "杭州"}},
        execution_trace={"summary": {"intent": "travel_plan"}, "steps": []},
        intent={"intent": "travel_plan"},
        task_results=[
            {
                "task": {"task_id": "search_attractions_1", "tool": "search_attractions"},
                "success": True,
                "error": None,
                "meta": {"duration_ms": 12, "execution_mode": "mock_fallback"},
            }
        ],
    )

    runs = store.list_runs("session-a")
    messages = store.get_recent_messages("session-a")

    assert record["run_id"]
    assert len(runs) == 1
    assert runs[0]["intent_type"] == "travel_plan"
    assert runs[0]["artifact_keys"] == ["itinerary"]
    assert runs[0]["trace_summary"]["intent"] == "travel_plan"
    assert runs[0]["task_summary"][0]["tool"] == "search_attractions"
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "帮我规划杭州三天"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["source"] == "persistent_session_history"


def test_session_history_store_clear_session_removes_persisted_runs(tmp_path):
    """清理会话时同步移除持久化运行记录。"""
    store = SessionHistoryStore(database_url=f"sqlite:///{tmp_path / 'history.db'}")
    store.save_run(
        session_id="session-clear",
        user_message="你好",
        ai_message="你好",
        artifacts={},
        execution_trace={},
        intent={"intent": "general_chat"},
        task_results=[],
    )

    removed = store.clear_session("session-clear")

    assert removed == 1
    assert store.list_runs("session-clear") == []
    assert store.get_recent_messages("session-clear") == []


@pytest.mark.asyncio
async def test_agent_persists_run_history_and_can_restore_messages(monkeypatch, tmp_path):
    """Agent运行结束后写入持久化历史，新的Agent实例可恢复消息。"""
    db_url = f"sqlite:///{tmp_path / 'agent-history.db'}"
    from app.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", db_url)
    first_agent = TravelAgent()

    result = await first_agent.arun_with_artifacts("杭州明天天气怎么样？", "persisted-session")
    second_agent = TravelAgent()

    runs = second_agent.get_session_runs("persisted-session")
    history = second_agent.get_history("persisted-session")

    assert result["history_persistence"]["stored"] is True
    assert runs
    assert runs[0]["execution_trace"]["summary"]["intent"] == "weather_query"
    assert runs[0]["ai_message"] == result["response"]
    assert history[-2]["role"] == "user"
    assert history[-2]["content"] == "杭州明天天气怎么样？"
    assert history[-1]["role"] == "assistant"


def test_session_runs_api_returns_persisted_artifacts_and_trace(monkeypatch, tmp_path):
    """运行历史API返回持久化artifact和execution trace。"""
    db_url = f"sqlite:///{tmp_path / 'api-history.db'}"
    from app.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", db_url)
    test_agent = TravelAgent()
    test_agent.session_history_store.save_run(
        session_id="api-session",
        user_message="规划杭州",
        ai_message="已规划",
        artifacts={"itinerary": {"destination": "杭州"}},
        execution_trace={"summary": {"intent": "travel_plan"}, "steps": []},
        intent={"intent": "travel_plan"},
        task_results=[],
    )
    monkeypatch.setattr(app_main, "agent", test_agent)
    client = TestClient(app_main.app)

    response = client.get("/api/history/api-session/runs")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "api-session"
    assert data["count"] == 1
    assert data["runs"][0]["artifacts"]["itinerary"]["destination"] == "杭州"
    assert data["runs"][0]["execution_trace"]["summary"]["intent"] == "travel_plan"
