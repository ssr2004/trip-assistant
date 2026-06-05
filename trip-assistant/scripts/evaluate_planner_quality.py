"""Run the planner quality benchmark suite.

Default mode is template-only and does not call LLM. Use --attempt-llm when you
explicitly want to evaluate the async planner path with the configured provider.
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
from core.planner_evaluation import (  # noqa: E402
    PlannerEvaluationResult,
    evaluate_benchmark_suite,
    evaluate_planner_tasks,
    get_default_planner_benchmarks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate planner quality on curated benchmark cases.")
    parser.add_argument("--mode", choices=["auto", "off", "always"], default="auto", help="Internal planner routing mode.")
    parser.add_argument("--attempt-llm", action="store_true", help="Use plan_async and allow LLM calls when routing permits.")
    parser.add_argument("--min-average-score", type=float, default=0.75, help="Minimum suite average normalized score.")
    parser.add_argument("--json-compact", action="store_true", help="Print compact JSON.")
    return parser


async def main() -> int:
    args = build_parser().parse_args()
    planner = TaskPlanner(llm_planner_mode=args.mode)
    results: List[PlannerEvaluationResult] = []

    for benchmark in get_default_planner_benchmarks():
        context = {**benchmark.context, "query": benchmark.query}
        if args.attempt_llm:
            tasks = await planner.plan_async(benchmark.intent, context)
        else:
            tasks = planner.plan(benchmark.intent, context)
        results.append(evaluate_planner_tasks(tasks, benchmark))

    suite = evaluate_benchmark_suite(results)
    payload: Dict[str, Any] = {
        "suite": {
            **suite,
            "mode": args.mode,
            "attempt_llm": args.attempt_llm,
            "llm_available": planner.llm_client.available,
        },
        "results": [_sanitize_result(result) for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=None if args.json_compact else 2))

    if not suite["passed"] or suite["average_normalized_score"] < args.min_average_score:
        return 2
    return 0


def _sanitize_result(result: PlannerEvaluationResult) -> Dict[str, Any]:
    return {
        "case_id": result.case_id,
        "name": result.name,
        "score": result.score,
        "max_score": result.max_score,
        "normalized_score": result.normalized_score,
        "passed": result.passed,
        "metrics": result.metrics,
        "covered_constraints": result.covered_constraints,
        "missing_constraints": result.missing_constraints,
        "issues": [issue.model_dump() for issue in result.issues],
    }


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
