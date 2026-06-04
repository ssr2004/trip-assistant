"""Manual smoke test for auto-routed LLM task planning.

Automated tests use --dry-run only. Real LLM planning uses --mode auto by default.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.planner import TaskPlanner  # noqa: E402


DEFAULT_QUERY = "我下个月从郑州去杭州三天，预算3000，不要太累，同时兼顾美食、地铁附近住宿和雨天备选路线"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a sanitized manual smoke test for LLM planner gating.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Complex planning query to test.")
    parser.add_argument("--mode", choices=["auto", "off", "always"], default="auto", help="Internal LLM planner routing mode.")
    parser.add_argument("--enable", action="store_true", help="Deprecated alias for --mode auto.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call LLM; show template plan and gating state.")
    parser.add_argument("--require-llm-plan", action="store_true", help="Exit non-zero if LLM planner is not adopted.")
    return parser


async def main() -> int:
    args = build_parser().parse_args()
    intent = _complex_intent()
    context = {"query": args.query}
    planner_mode = "auto" if args.enable else args.mode
    planner = TaskPlanner(llm_planner_mode=planner_mode)

    template_tasks = planner.plan(intent, context)
    template_metadata = planner.last_plan_metadata
    skip_reason = planner._llm_plan_skip_reason(intent, context, planner._template_plan(intent, context))
    result: Dict[str, Any] = {
        "preflight": {
            "llm_planner_enabled": planner.llm_planner_enabled,
            "planner_mode_config": planner.llm_planner_mode,
            "llm_available": planner.llm_client.available,
            "would_try_llm_planner": skip_reason is None,
            "skip_reason": skip_reason,
            "complexity_score": planner.last_plan_metadata.get("llm_planner_complexity_score", 0),
            "complexity_signals": planner.last_plan_metadata.get("llm_planner_complexity_signals", []),
            "template_task_count": len(template_tasks),
        },
        "template_plan": _task_summary(template_tasks),
        "template_metadata": _sanitize_metadata(template_metadata),
    }

    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    tasks = await planner.plan_async(intent, context)
    result["planner_result"] = {
        "tasks": _task_summary(tasks),
        "metadata": _sanitize_metadata(planner.last_plan_metadata),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.require_llm_plan and not planner.last_plan_metadata.get("llm_planner_adopted"):
        return 2
    return 0


def _complex_intent() -> Dict[str, Any]:
    return {
        "intent": "travel_plan",
        "entities": {
            "origin": "郑州",
            "destination": "杭州",
            "departure_date": "2026-07-10",
            "return_date": None,
            "duration": 3,
            "budget": 3000,
            "travelers": 2,
            "preferences": ["慢节奏", "地铁附近", "美食", "雨天备选"],
        },
        "missing_slots": [],
        "confidence": 0.8,
    }


def _task_summary(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "task_id": task.get("task_id"),
            "task_type": task.get("task_type"),
            "tool": task.get("tool"),
            "depends_on": task.get("depends_on", []),
        }
        for task in tasks
    ]


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    allowed_keys = {
        "planner_mode",
        "llm_planner_enabled",
        "llm_planner_available",
        "llm_planner_attempted",
        "llm_planner_adopted",
        "llm_planner_model",
        "llm_planner_error_type",
        "llm_planner_duration_ms",
        "llm_planner_prompt_tokens",
        "llm_planner_completion_tokens",
        "llm_planner_total_tokens",
        "planner_mode_config",
        "llm_planner_auto_route",
        "llm_planner_complexity_score",
        "llm_planner_complexity_threshold",
        "llm_planner_complexity_signals",
        "llm_planner_route_decision",
        "fallback_reason",
        "skip_reason",
        "template_task_count",
        "llm_task_count",
    }
    return {key: value for key, value in metadata.items() if key in allowed_keys}


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
