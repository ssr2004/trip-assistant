"""Run end-to-end Agent benchmark scenarios.

Default mode disables configured real API keys and uses mock/fallback paths so
the benchmark is deterministic. Use --use-configured-services only for manual
local smoke runs when you explicitly want real providers involved.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from core.agent import TravelAgent  # noqa: E402
from core.agent_evaluation import (  # noqa: E402
    AgentScenarioEvaluationResult,
    evaluate_agent_scenario,
    get_default_agent_e2e_benchmarks,
    summarize_agent_evaluation,
)
from core.memory.manager import MemoryManager  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TravelMind Agent E2E benchmark scenarios.")
    parser.add_argument("--json-compact", action="store_true", help="Print compact JSON.")
    parser.add_argument("--use-configured-services", action="store_true", help="Allow local real API/LLM settings.")
    parser.add_argument("--scenario", help="Run only one scenario_id.")
    return parser


async def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    if not args.use_configured_services:
        _force_deterministic_settings()

    with tempfile.TemporaryDirectory(prefix="travelmind-agent-e2e-") as temp_dir:
        with redirect_stdout(sys.stderr):
            agent = TravelAgent()
            agent.memory_manager = MemoryManager(
                long_term_storage_path=str(Path(temp_dir) / "long_term_memory.json"),
                episodic_storage_path=str(Path(temp_dir) / "episodic_memory.json"),
            )
            benchmarks = get_default_agent_e2e_benchmarks()
            if args.scenario:
                benchmarks = [case for case in benchmarks if case.scenario_id == args.scenario]
            if not benchmarks:
                print(json.dumps({"suite": {"passed": False, "error": "scenario_not_found"}}, ensure_ascii=False))
                return 2

            results: List[AgentScenarioEvaluationResult] = []
            for scenario in benchmarks:
                results.append(
                    await evaluate_agent_scenario(
                        agent,
                        scenario,
                        session_id=f"agent-e2e-{scenario.scenario_id}",
                        sensitive_markers=[
                            "AMAP_API_KEY",
                            "WEATHER_API_KEY",
                            "LLM_API_KEY",
                            "AMADEUS_API_KEY",
                            "AMADEUS_API_SECRET",
                            settings.AMAP_API_KEY,
                            settings.WEATHER_API_KEY,
                            settings.LLM_API_KEY,
                            "sk-",
                        ],
                    )
                )

    suite = summarize_agent_evaluation(results)
    payload: Dict[str, Any] = {
        "suite": {
            **suite,
            "deterministic_mode": not args.use_configured_services,
        },
        "results": [_sanitize_result(result) for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=None if args.json_compact else 2))
    return 0 if suite["passed"] else 2


def _force_deterministic_settings() -> None:
    settings.LLM_API_KEY = ""
    settings.LLM_PLANNER_MODE = "off"
    settings.AMAP_API_KEY = ""
    settings.WEATHER_API_KEY = ""
    settings.AMADEUS_API_KEY = ""
    settings.AMADEUS_API_SECRET = ""
    settings.EXTERNAL_API_MOCK_ENABLED = True


def _sanitize_result(result: AgentScenarioEvaluationResult) -> Dict[str, Any]:
    return {
        "scenario_id": result.scenario_id,
        "name": result.name,
        "passed": result.passed,
        "metrics": result.metrics,
        "issues": result.issues,
        "turns": [
            {
                "turn_index": turn.turn_index,
                "passed": turn.passed,
                "issues": turn.issues,
                "metrics": turn.metrics,
            }
            for turn in result.turns
        ],
    }


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
