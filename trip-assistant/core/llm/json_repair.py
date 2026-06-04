"""Utilities for robust LLM JSON parsing and one-shot repair."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Dict, Optional

from core.llm.schemas import LLMMessage, LLMRequest


JSON_PARSE_FAILED = "json_parse_failed"
SCHEMA_VALIDATION_FAILED = "schema_validation_failed"


@dataclass
class StructuredJSONResult:
    """Result for parsing and validating structured LLM JSON."""

    data: Optional[Dict[str, Any]] = None
    validated: Any = None
    error_type: Optional[str] = None
    repair_attempted: bool = False
    repair_success: bool = False
    repair_call_success: bool = False
    repair_error_type: Optional[str] = None
    repair_duration_ms: int = 0
    repair_prompt_tokens: int = 0
    repair_completion_tokens: int = 0
    repair_total_tokens: int = 0

    @property
    def success(self) -> bool:
        return self.error_type is None and (self.validated is not None or self.data is not None)


def parse_llm_json_object(content: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from raw LLM output.

    Supports plain JSON, Markdown fenced JSON, and extracting the first balanced
    JSON object from surrounding explanatory text.
    """
    if not content:
        return None

    text = _strip_markdown_fence(content.strip())
    for candidate in [text, _extract_first_json_object(text)]:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


async def parse_or_repair_json_object(
    content: str,
    *,
    llm_client: Any,
    validator: Callable[[Dict[str, Any]], Any],
    repair_instructions: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> StructuredJSONResult:
    """Parse and validate JSON, then ask the LLM to repair it once if needed."""
    parsed = parse_llm_json_object(content)
    if parsed is not None:
        try:
            return StructuredJSONResult(data=parsed, validated=validator(parsed))
        except Exception:
            failure_type = SCHEMA_VALIDATION_FAILED
    else:
        failure_type = JSON_PARSE_FAILED

    result = StructuredJSONResult(error_type=failure_type, repair_attempted=True)
    repair_response = await llm_client.chat(
        LLMRequest(
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "You repair malformed or schema-invalid JSON from another LLM call. "
                        "Return only one valid JSON object. Do not include Markdown fences."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=(
                        f"Failure type: {failure_type}\n"
                        f"Schema and output requirements:\n{repair_instructions}\n"
                        f"Original output:\n{content}"
                    ),
                ),
            ],
            response_format="json_object",
            metadata={
                **(metadata or {}),
                "repair_for": "structured_json",
                "repair_failure_type": failure_type,
            },
        )
    )
    if not repair_response.success:
        result.repair_error_type = repair_response.metadata.get("error_type")
        result.repair_duration_ms = int(repair_response.metadata.get("duration_ms") or 0)
        return result

    result.repair_call_success = True
    result.repair_duration_ms = int(repair_response.metadata.get("duration_ms") or 0)
    result.repair_prompt_tokens = int(repair_response.metadata.get("prompt_tokens") or 0)
    result.repair_completion_tokens = int(repair_response.metadata.get("completion_tokens") or 0)
    result.repair_total_tokens = int(repair_response.metadata.get("total_tokens") or 0)

    repaired = parse_llm_json_object(repair_response.content)
    if repaired is None:
        result.error_type = JSON_PARSE_FAILED
        return result

    try:
        result.validated = validator(repaired)
    except Exception:
        result.error_type = SCHEMA_VALIDATION_FAILED
        result.data = repaired
        return result

    result.data = repaired
    result.error_type = None
    result.repair_success = True
    return result


def _strip_markdown_fence(text: str) -> str:
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
