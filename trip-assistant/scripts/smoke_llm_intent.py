"""Manual smoke test for real LLM intent fallback.

This script is intentionally not part of automated tests. It uses local
configuration and prints only sanitized runtime metadata.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.agent import TravelAgent  # noqa: E402
from core.intent import IntentParser  # noqa: E402


DEFAULT_QUERY = "想找一个安静不赶路的周末旅行地，预算两千左右"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a sanitized manual smoke test for real LLM intent fallback.",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Low-confidence travel query to test.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show rule intent and whether LLM fallback would be attempted.",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Also run TravelAgent.arun_with_artifacts and print sanitized trace summary.",
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Exit non-zero if the final intent source is not llm.",
    )
    return parser


async def main() -> int:
    args = build_parser().parse_args()
    parser = IntentParser()
    rule_result = parser.parse(args.query)
    would_use_llm = parser._should_use_llm_fallback(rule_result)

    preflight = {
        "query": args.query,
        "llm_available": parser.llm_client.available,
        "rule_intent": rule_result.get("intent"),
        "rule_confidence": rule_result.get("confidence"),
        "would_use_llm_fallback": would_use_llm,
        "llm_model": parser.llm_client.settings.LLM_MODEL,
        "llm_provider": parser.llm_client.settings.LLM_PROVIDER,
    }
    if args.dry_run:
        print(json.dumps({"preflight": preflight}, ensure_ascii=False, indent=2))
        return 0

    parsed = await parser.parse_async(args.query)
    result: Dict[str, Any] = {
        "preflight": preflight,
        "intent_result": {
            "intent": parsed.get("intent"),
            "confidence": parsed.get("confidence"),
            "missing_slots": parsed.get("missing_slots", []),
            "metadata": _sanitize_intent_metadata(parsed.get("metadata", {})),
        },
    }

    if args.agent:
        agent = TravelAgent()
        agent_result = await agent.arun_with_artifacts(args.query, "manual-smoke-llm-intent")
        trace = agent_result.get("execution_trace", {}) or {}
        result["agent_trace_summary"] = _sanitize_trace_summary(trace.get("summary", {}) or {})

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.require_llm and parsed.get("metadata", {}).get("source") != "llm":
        return 2
    return 0


def _sanitize_intent_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    allowed_keys = {
        "source",
        "llm_error_type",
        "llm_model",
        "provider",
        "prompt_id",
        "prompt_version",
        "llm_output_quality_passed",
        "llm_output_quality_issue_count",
        "llm_output_quality_issues",
        "json_repair_attempted",
        "json_repair_success",
        "json_repair_error_type",
        "llm_call_count",
        "llm_success_count",
        "llm_failure_count",
        "llm_duration_ms",
        "llm_prompt_tokens",
        "llm_completion_tokens",
        "llm_total_tokens",
    }
    return {key: value for key, value in metadata.items() if key in allowed_keys}


def _sanitize_trace_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    allowed_prefixes = ("llm_",)
    allowed_keys = {
        "intent",
        "intent_source",
        "llm_mode",
        "llm_model",
        "task_count",
        "tool_count",
        "failed_count",
        "source_count",
        "total_duration_ms",
        "tool_total_duration_ms",
        "real_api_count",
        "mock_fallback_count",
        "template_task_count",
        "dynamic_rag_count",
        "internal_task_count",
        "json_repair_attempted",
        "json_repair_success",
    }
    return {
        key: value
        for key, value in summary.items()
        if key in allowed_keys or any(key.startswith(prefix) for prefix in allowed_prefixes)
    }


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
