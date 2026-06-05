"""Persistent session run history for agent observability and recovery."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import settings


class SessionHistoryStore:
    """SQLite-backed store for complete agent turn records."""

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or settings.DATABASE_URL
        self.db_path = self._resolve_sqlite_path(self.database_url)
        self._memory_connection: sqlite3.Connection | None = None
        self._ensure_schema()

    def save_run(
        self,
        session_id: str,
        user_message: str,
        ai_message: str,
        artifacts: dict[str, Any] | None = None,
        execution_trace: dict[str, Any] | None = None,
        intent: dict[str, Any] | None = None,
        task_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Persist one user turn plus sanitized runtime outputs."""
        run_id = uuid4().hex
        created_at = self._now()
        task_summary = self._task_summary(task_results or [])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_runs (
                    run_id,
                    session_id,
                    user_message,
                    ai_message,
                    intent_type,
                    task_count,
                    failed_count,
                    artifact_keys,
                    trace_summary,
                    artifacts_json,
                    execution_trace_json,
                    task_summary_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    session_id or "default",
                    user_message,
                    ai_message,
                    self._intent_type(intent),
                    len(task_summary),
                    sum(1 for task in task_summary if not task.get("success")),
                    json.dumps(sorted((artifacts or {}).keys()), ensure_ascii=False),
                    json.dumps((execution_trace or {}).get("summary", {}), ensure_ascii=False),
                    self._json_dumps(artifacts or {}),
                    self._json_dumps(execution_trace or {}),
                    self._json_dumps(task_summary),
                    created_at,
                ),
            )
        return {
            "run_id": run_id,
            "session_id": session_id or "default",
            "created_at": created_at,
            "task_count": len(task_summary),
            "failed_count": sum(1 for task in task_summary if not task.get("success")),
            "artifact_keys": sorted((artifacts or {}).keys()),
        }

    def list_runs(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return persisted agent runs for a session, newest first."""
        normalized_limit = max(1, min(int(limit or 20), 100))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    run_id,
                    session_id,
                    user_message,
                    ai_message,
                    intent_type,
                    task_count,
                    failed_count,
                    artifact_keys,
                    trace_summary,
                    artifacts_json,
                    execution_trace_json,
                    task_summary_json,
                    created_at
                FROM agent_runs
                WHERE session_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (session_id or "default", normalized_limit),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def get_recent_messages(self, session_id: str, limit: int = 40) -> list[dict[str, Any]]:
        """Reconstruct chat messages from persisted runs in chronological order."""
        run_limit = max(1, min((int(limit or 40) + 1) // 2, 100))
        runs = list(reversed(self.list_runs(session_id, limit=run_limit)))
        messages: list[dict[str, Any]] = []
        for run in runs:
            created_at = run.get("created_at")
            messages.append({
                "role": "user",
                "content": run.get("user_message", ""),
                "timestamp": created_at,
                "run_id": run.get("run_id"),
                "source": "persistent_session_history",
            })
            messages.append({
                "role": "assistant",
                "content": run.get("ai_message", ""),
                "timestamp": created_at,
                "run_id": run.get("run_id"),
                "source": "persistent_session_history",
            })
        return messages[-max(1, int(limit or 40)):]

    def clear_session(self, session_id: str) -> int:
        """Remove persisted runs for a session and return affected row count."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM agent_runs WHERE session_id = ?",
                (session_id or "default",),
            )
        return int(cursor.rowcount or 0)

    def _connect(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._memory_connection is None:
                self._memory_connection = sqlite3.connect(":memory:")
                self._memory_connection.row_factory = sqlite3.Row
            return self._memory_connection
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    ai_message TEXT NOT NULL,
                    intent_type TEXT,
                    task_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    artifact_keys TEXT NOT NULL DEFAULT '[]',
                    trace_summary TEXT NOT NULL DEFAULT '{}',
                    artifacts_json TEXT NOT NULL DEFAULT '{}',
                    execution_trace_json TEXT NOT NULL DEFAULT '{}',
                    task_summary_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_runs_session_created ON agent_runs(session_id, created_at)"
            )

    def _resolve_sqlite_path(self, database_url: str) -> str:
        if database_url == "sqlite:///:memory:":
            return ":memory:"
        if not database_url.startswith("sqlite:///"):
            raise ValueError("SessionHistoryStore currently supports sqlite:/// DATABASE_URL only")
        raw_path = database_url.removeprefix("sqlite:///")
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        return str(path)

    def _row_to_run(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "session_id": row["session_id"],
            "user_message": row["user_message"],
            "ai_message": row["ai_message"],
            "intent_type": row["intent_type"],
            "task_count": row["task_count"],
            "failed_count": row["failed_count"],
            "artifact_keys": self._json_loads(row["artifact_keys"], []),
            "trace_summary": self._json_loads(row["trace_summary"], {}),
            "artifacts": self._json_loads(row["artifacts_json"], {}),
            "execution_trace": self._json_loads(row["execution_trace_json"], {}),
            "task_summary": self._json_loads(row["task_summary_json"], []),
            "created_at": row["created_at"],
        }

    def _task_summary(self, task_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summary = []
        for task_result in task_results:
            task = task_result.get("task", {}) or {}
            meta = task_result.get("meta", {}) or {}
            summary.append({
                "task_id": task.get("task_id"),
                "task_type": task.get("task_type"),
                "tool": task.get("tool"),
                "name": task.get("name"),
                "success": bool(task_result.get("success")),
                "error": task_result.get("error"),
                "duration_ms": meta.get("duration_ms"),
                "execution_mode": meta.get("execution_mode"),
                "degraded": meta.get("degraded"),
                "fallback_used": meta.get("fallback_used"),
                "result_summary": meta.get("result_summary"),
            })
        return summary

    def _intent_type(self, intent: dict[str, Any] | None) -> str:
        if not isinstance(intent, dict):
            return "general_chat"
        return str(intent.get("intent") or "general_chat")

    def _json_dumps(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    def _json_loads(self, value: str, fallback: Any) -> Any:
        try:
            return json.loads(value)
        except Exception:
            return fallback

    def _now(self) -> str:
        return datetime.now().isoformat()
