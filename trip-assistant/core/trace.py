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


class ExecutionTrace(BaseModel):
    steps: list[TraceStep] = Field(default_factory=list)
    summary: dict[str, int | str | bool | list[str]] = Field(default_factory=dict)


def normalize_execution_trace(value: Any) -> dict[str, Any]:
    """Validate execution trace payloads and return the frontend JSON shape."""

    if isinstance(value, ExecutionTrace):
        return value.model_dump(exclude_none=True)
    return ExecutionTrace.model_validate(value or {}).model_dump(exclude_none=True)
