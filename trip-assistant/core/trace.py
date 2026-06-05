"""Typed execution trace contracts for agent observability."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    stage: str
    label: str
    status: str = "success"
    detail: str | None = None
    task_type: str | None = None
    tool: str | None = None
    duration_ms: int | None = None
    execution_mode: str | None = None
    error_type: str | None = None
    result_summary: str | None = None
    source_count: int | None = None
    dependency_ids: list[str] | None = None
    resolved_dependencies: list[str] | None = None
    missing_dependencies: list[str] | None = None
    failed_dependencies: list[str] | None = None
    dependency_context_keys: list[str] | None = None
    dependency_error_count: int | None = None
    failure_category: str | None = None
    recoverable: bool | None = None
    degraded: bool | None = None
    fallback_used: bool | None = None
    recovery_strategy: str | None = None
    degradation_reason: str | None = None
    provider: str | None = None
    api_status: str | None = None
    cache_hit: bool | None = None
    cache_backend: str | None = None
    cache_write: bool | None = None
    memory_preference_source: str | None = None
    memory_used_preferences: list[str] | None = None
    memory_preference_count: int | None = None


class ExecutionTrace(BaseModel):
    steps: list[TraceStep] = Field(default_factory=list)
    summary: dict[str, int | str | bool | list[str] | dict[str, int]] = Field(default_factory=dict)


def normalize_execution_trace(value: Any) -> dict[str, Any]:
    """Validate execution trace payloads and return the frontend JSON shape."""

    if isinstance(value, ExecutionTrace):
        return value.model_dump(exclude_none=True)
    return ExecutionTrace.model_validate(value or {}).model_dump(exclude_none=True)
