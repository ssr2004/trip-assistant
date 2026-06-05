"""End-to-end Agent benchmark definitions and evaluation utilities.

The evaluator runs the real TravelAgent graph and checks product-engineering
contracts: intent, artifacts, trace stages, tool coverage, execution modes, and
key leakage boundaries. It is separate from runtime code so it can be used by
tests, smoke scripts, and interview demos without changing the serving path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence
import json

from pydantic import BaseModel, Field

from core.agent import TravelAgent


DEFAULT_SENSITIVE_MARKERS = [
    "AMAP_API_KEY",
    "WEATHER_API_KEY",
    "LLM_API_KEY",
    "AMADEUS_API_KEY",
    "AMADEUS_API_SECRET",
    "sk-",
]


class AgentBenchmarkTurn(BaseModel):
    """One user turn in an Agent E2E benchmark scenario."""

    message: str
    expected_intent: str
    required_artifacts: List[str] = Field(default_factory=list)
    required_tools: List[str] = Field(default_factory=list)
    required_execution_modes: List[str] = Field(default_factory=list)
    required_trace_stages: List[str] = Field(default_factory=lambda: ["intent", "context", "planning"])
    required_response_keywords: List[str] = Field(default_factory=list)
    min_task_count: int = 1


class AgentBenchmarkScenario(BaseModel):
    """A multi-turn Agent E2E benchmark scenario."""

    scenario_id: str
    name: str
    turns: List[AgentBenchmarkTurn]
    required_memory_keywords: List[str] = Field(default_factory=list)


class AgentTurnEvaluationResult(BaseModel):
    """Evaluation result for one turn."""

    turn_index: int
    message: str
    passed: bool
    issues: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class AgentScenarioEvaluationResult(BaseModel):
    """Evaluation result for one scenario."""

    scenario_id: str
    name: str
    passed: bool
    turns: List[AgentTurnEvaluationResult]
    metrics: Dict[str, Any] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)


def get_default_agent_e2e_benchmarks() -> List[AgentBenchmarkScenario]:
    """Return curated Agent E2E benchmark scenarios."""

    return [
        AgentBenchmarkScenario(
            scenario_id="complete_plan_artifact_trace",
            name="完整旅行规划Artifact与Trace",
            turns=[
                AgentBenchmarkTurn(
                    message="我要从郑州去杭州玩三天，预算3000，6月10日出发",
                    expected_intent="travel_plan",
                    required_artifacts=["itinerary", "attractions"],
                    required_tools=[
                        "search_attractions",
                        "generate_itinerary",
                    ],
                    required_response_keywords=["杭州"],
                    min_task_count=5,
                )
            ],
        ),
        AgentBenchmarkScenario(
            scenario_id="demo_multi_turn_chain",
            name="规划-雨天调整-路线优化-动态RAG追问",
            turns=[
                AgentBenchmarkTurn(
                    message="我要从郑州去杭州玩三天，预算3000，6月10日出发",
                    expected_intent="travel_plan",
                    required_artifacts=["itinerary", "attractions"],
                    required_tools=["search_attractions", "generate_itinerary"],
                    required_response_keywords=["杭州"],
                    min_task_count=4,
                ),
                AgentBenchmarkTurn(
                    message="如果下雨怎么办？",
                    expected_intent="itinerary_revision",
                    required_artifacts=["itinerary", "weather_adjustment"],
                    required_execution_modes=["internal_revision"],
                    required_response_keywords=["调整"],
                ),
                AgentBenchmarkTurn(
                    message="帮我按距离优化一下第二天行程",
                    expected_intent="itinerary_revision",
                    required_artifacts=["itinerary", "route"],
                    required_execution_modes=["internal_revision"],
                    required_response_keywords=["路线"],
                ),
                AgentBenchmarkTurn(
                    message="西湖在哪里？",
                    expected_intent="dynamic_knowledge_query",
                    required_artifacts=["attractions"],
                    required_execution_modes=["dynamic_rag"],
                    required_response_keywords=["西湖"],
                ),
            ],
        ),
        AgentBenchmarkScenario(
            scenario_id="memory_preference_to_later_plan",
            name="记忆偏好影响后续规划",
            turns=[
                AgentBenchmarkTurn(
                    message="我喜欢慢节奏，也喜欢当地美食和自然风光",
                    expected_intent="general_chat",
                    min_task_count=0,
                    required_response_keywords=["理解"],
                ),
                AgentBenchmarkTurn(
                    message="我要从郑州去杭州玩三天，预算3000，6月10日出发",
                    expected_intent="travel_plan",
                    required_artifacts=["itinerary", "attractions"],
                    required_tools=["search_hotels", "generate_itinerary"],
                    required_response_keywords=["杭州"],
                    min_task_count=4,
                ),
            ],
            required_memory_keywords=["慢节奏", "当地美食", "自然风光"],
        ),
    ]


async def evaluate_agent_scenario(
    agent: TravelAgent,
    scenario: AgentBenchmarkScenario,
    *,
    session_id: str,
    sensitive_markers: Sequence[str] | None = None,
) -> AgentScenarioEvaluationResult:
    """Run and evaluate one Agent benchmark scenario."""

    sensitive_markers = list(sensitive_markers or DEFAULT_SENSITIVE_MARKERS)
    turn_results: List[AgentTurnEvaluationResult] = []
    scenario_issues: List[str] = []

    for index, turn in enumerate(scenario.turns, start=1):
        result = await agent.arun_with_artifacts(turn.message, session_id)
        turn_results.append(_evaluate_turn(index, turn, result, sensitive_markers))

    if scenario.required_memory_keywords:
        memory_context = agent.memory_manager.retrieve("", session_id)
        memory_blob = json.dumps(memory_context, ensure_ascii=False)
        for keyword in scenario.required_memory_keywords:
            if keyword not in memory_blob:
                scenario_issues.append(f"memory_missing:{keyword}")

    return AgentScenarioEvaluationResult(
        scenario_id=scenario.scenario_id,
        name=scenario.name,
        passed=all(turn.passed for turn in turn_results) and not scenario_issues,
        turns=turn_results,
        metrics={
            "turn_count": len(turn_results),
            "passed_turn_count": len([turn for turn in turn_results if turn.passed]),
            "failed_turn_count": len([turn for turn in turn_results if not turn.passed]),
        },
        issues=scenario_issues,
    )


def summarize_agent_evaluation(results: Sequence[AgentScenarioEvaluationResult]) -> Dict[str, Any]:
    """Aggregate scenario-level Agent E2E evaluation results."""

    scenario_count = len(results)
    passed_count = len([result for result in results if result.passed])
    turn_count = sum(result.metrics.get("turn_count", 0) for result in results)
    passed_turn_count = sum(result.metrics.get("passed_turn_count", 0) for result in results)
    return {
        "scenario_count": scenario_count,
        "passed_scenario_count": passed_count,
        "failed_scenario_count": scenario_count - passed_count,
        "turn_count": turn_count,
        "passed_turn_count": passed_turn_count,
        "failed_turn_count": turn_count - passed_turn_count,
        "pass_rate": round(passed_count / scenario_count, 4) if scenario_count else 0.0,
        "passed": passed_count == scenario_count,
    }


def _evaluate_turn(
    turn_index: int,
    turn: AgentBenchmarkTurn,
    result: Dict[str, Any],
    sensitive_markers: Sequence[str],
) -> AgentTurnEvaluationResult:
    issues: List[str] = []
    artifacts = result.get("artifacts", {}) or {}
    trace = result.get("execution_trace", {}) or {}
    summary = trace.get("summary", {}) or {}
    steps = trace.get("steps", []) or []
    response = result.get("response", "") or ""

    if not response:
        issues.append("empty_response")

    if summary.get("intent") != turn.expected_intent:
        issues.append(f"intent_mismatch:{summary.get('intent')}!={turn.expected_intent}")

    if int(summary.get("task_count") or 0) < turn.min_task_count:
        issues.append(f"task_count_below_min:{summary.get('task_count')}<{turn.min_task_count}")

    actual_artifacts = set(artifacts.keys())
    for artifact in turn.required_artifacts:
        if artifact not in actual_artifacts or not artifacts.get(artifact):
            issues.append(f"missing_artifact:{artifact}")

    actual_tools = {step.get("tool") for step in steps if step.get("tool")}
    for tool in turn.required_tools:
        if tool not in actual_tools:
            issues.append(f"missing_tool:{tool}")

    actual_modes = {step.get("execution_mode") for step in steps if step.get("execution_mode")}
    for mode in turn.required_execution_modes:
        if mode not in actual_modes:
            issues.append(f"missing_execution_mode:{mode}")

    actual_stages = {step.get("stage") for step in steps}
    for stage in turn.required_trace_stages:
        if stage not in actual_stages:
            issues.append(f"missing_trace_stage:{stage}")

    for keyword in turn.required_response_keywords:
        if keyword and keyword not in response:
            issues.append(f"missing_response_keyword:{keyword}")

    serialized_result = json.dumps(result, ensure_ascii=False, default=str)
    for marker in sensitive_markers:
        if marker and marker in serialized_result:
            issues.append(f"sensitive_marker_leaked:{marker}")

    return AgentTurnEvaluationResult(
        turn_index=turn_index,
        message=turn.message,
        passed=not issues,
        issues=issues,
        metrics={
            "intent": summary.get("intent"),
            "task_count": summary.get("task_count", 0),
            "tool_count": summary.get("tool_count", 0),
            "source_count": summary.get("source_count", 0),
            "artifact_keys": sorted(actual_artifacts),
            "tools": sorted(tool for tool in actual_tools if tool),
            "execution_modes": sorted(mode for mode in actual_modes if mode),
        },
    )
