"""Planner quality benchmarks and scoring utilities.

This module is intentionally separate from the runtime planner path. It is used
by tests and smoke scripts to prove that task planning is measurable, safe, and
regression-tested instead of being judged by ad-hoc manual inspection.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
import json

from pydantic import BaseModel, Field

from models.task import PlanningTask, TaskPlan


DEFAULT_ALLOWED_TASK_TYPES = {
    "ask_user",
    "tool_call",
    "recommend_destination",
    "generate_itinerary",
    "dynamic_rag_query",
    "revise_itinerary",
}
DEFAULT_ALLOWED_TOOLS = {
    "search_flights",
    "search_hotels",
    "search_attractions",
    "retrieve_policy",
    "retrieve_guide",
    "generate_itinerary",
    "get_weather_forecast",
}


class ConstraintExpectation(BaseModel):
    """A user constraint that should be represented somewhere in the task plan."""

    signal: str
    keywords: List[str] = Field(default_factory=list)
    weight: int = 1


class PlannerBenchmarkCase(BaseModel):
    """A planner quality benchmark case."""

    case_id: str
    name: str
    query: str
    intent: Dict[str, Any]
    context: Dict[str, Any] = Field(default_factory=dict)
    expected_tools: List[str] = Field(default_factory=list)
    expected_task_types: List[str] = Field(default_factory=list)
    expected_constraints: List[ConstraintExpectation] = Field(default_factory=list)
    min_normalized_score: float = 0.75


class PlannerEvaluationIssue(BaseModel):
    """A concrete quality issue found in a task plan."""

    code: str
    severity: str
    detail: str


class PlannerEvaluationResult(BaseModel):
    """Serializable planner quality evaluation result."""

    case_id: str
    name: str
    score: int
    max_score: int
    normalized_score: float
    passed: bool
    metrics: Dict[str, Any] = Field(default_factory=dict)
    covered_constraints: List[str] = Field(default_factory=list)
    missing_constraints: List[str] = Field(default_factory=list)
    issues: List[PlannerEvaluationIssue] = Field(default_factory=list)


def get_default_planner_benchmarks() -> List[PlannerBenchmarkCase]:
    """Return the curated planner benchmark suite used by tests and smoke scripts."""

    return [
        PlannerBenchmarkCase(
            case_id="complex_hangzhou_multi_constraint",
            name="杭州三天多约束完整规划",
            query="我下个月从郑州去杭州三天，预算3000，不要太累，同时兼顾美食、地铁附近住宿和雨天备选路线",
            intent={
                "intent": "travel_plan",
                "entities": {
                    "origin": "郑州",
                    "destination": "杭州",
                    "departure_date": "2026-07-10",
                    "return_date": None,
                    "duration": 3,
                    "budget": 3000,
                    "travelers": 2,
                    "preferences": ["慢节奏", "美食", "地铁附近", "雨天备选"],
                },
                "missing_slots": [],
                "confidence": 0.86,
            },
            context={
                "query": "我下个月从郑州去杭州三天，预算3000，不要太累，同时兼顾美食、地铁附近住宿和雨天备选路线",
            },
            expected_tools=[
                "search_flights",
                "search_hotels",
                "search_attractions",
                "retrieve_guide",
                "get_weather_forecast",
                "generate_itinerary",
            ],
            expected_task_types=["tool_call", "generate_itinerary"],
            expected_constraints=[
                ConstraintExpectation(signal="budget", keywords=["预算", "3000", "budget"]),
                ConstraintExpectation(signal="pace", keywords=["不要太累", "慢节奏", "轻松"]),
                ConstraintExpectation(signal="food", keywords=["美食"]),
                ConstraintExpectation(signal="metro_hotel", keywords=["地铁附近", "地铁", "住宿", "酒店"]),
                ConstraintExpectation(signal="rain_backup", keywords=["雨天", "备选", "天气"]),
            ],
            min_normalized_score=0.78,
        ),
        PlannerBenchmarkCase(
            case_id="family_beijing_budget_route",
            name="北京亲子预算与路线约束",
            query="一家三口从上海去北京玩4天，预算8000，希望亲子友好，少走路，住宿靠近地铁，路线尽量顺",
            intent={
                "intent": "travel_plan",
                "entities": {
                    "origin": "上海",
                    "destination": "北京",
                    "departure_date": "2026-08-03",
                    "return_date": None,
                    "duration": 4,
                    "budget": 8000,
                    "travelers": 3,
                    "preferences": ["亲子友好", "少走路", "地铁附近", "路线顺"],
                },
                "missing_slots": [],
                "confidence": 0.84,
            },
            context={
                "query": "一家三口从上海去北京玩4天，预算8000，希望亲子友好，少走路，住宿靠近地铁，路线尽量顺",
            },
            expected_tools=[
                "search_flights",
                "search_hotels",
                "search_attractions",
                "retrieve_guide",
                "generate_itinerary",
            ],
            expected_task_types=["tool_call", "generate_itinerary"],
            expected_constraints=[
                ConstraintExpectation(signal="family", keywords=["亲子", "一家三口", "孩子"]),
                ConstraintExpectation(signal="budget", keywords=["预算", "8000", "budget"]),
                ConstraintExpectation(signal="route", keywords=["少走路", "路线", "顺路"]),
                ConstraintExpectation(signal="metro_hotel", keywords=["地铁", "住宿", "酒店"]),
            ],
            min_normalized_score=0.76,
        ),
        PlannerBenchmarkCase(
            case_id="destination_recommendation_no_city",
            name="目的地不明确的候选推荐",
            query="我想从广州出发玩三天，预算2500，想看海、吃海鲜、节奏轻松，但还没想好去哪",
            intent={
                "intent": "travel_plan",
                "entities": {
                    "origin": "广州",
                    "destination": None,
                    "duration": 3,
                    "budget": 2500,
                    "travelers": 2,
                    "preferences": ["看海", "海鲜", "节奏轻松"],
                },
                "missing_slots": [],
                "confidence": 0.81,
            },
            context={
                "query": "我想从广州出发玩三天，预算2500，想看海、吃海鲜、节奏轻松，但还没想好去哪",
            },
            expected_task_types=["recommend_destination"],
            expected_constraints=[
                ConstraintExpectation(signal="origin", keywords=["广州"]),
                ConstraintExpectation(signal="budget", keywords=["2500", "预算", "budget"]),
                ConstraintExpectation(signal="sea", keywords=["看海", "海"]),
                ConstraintExpectation(signal="food", keywords=["海鲜"]),
                ConstraintExpectation(signal="pace", keywords=["轻松", "节奏"]),
            ],
            min_normalized_score=0.74,
        ),
    ]


def evaluate_planner_tasks(
    tasks: Sequence[Dict[str, Any]],
    benchmark: PlannerBenchmarkCase,
    *,
    allowed_task_types: Optional[Set[str]] = None,
    allowed_tools: Optional[Set[str]] = None,
) -> PlannerEvaluationResult:
    """Evaluate a task list against one benchmark case."""

    allowed_task_types = allowed_task_types or DEFAULT_ALLOWED_TASK_TYPES
    allowed_tools = allowed_tools or DEFAULT_ALLOWED_TOOLS
    issues: List[PlannerEvaluationIssue] = []
    validated_tasks = _validate_tasks(tasks, issues)

    score = 0
    max_score = 100

    if validated_tasks:
        score += 15
    else:
        _add_issue(issues, "schema_invalid", "critical", "Task list cannot be validated as PlanningTask objects.")

    safety_score, safety_issues = _score_safety(validated_tasks, allowed_task_types, allowed_tools)
    score += safety_score
    issues.extend(safety_issues)

    tool_score, missing_tools = _score_required_tools(validated_tasks, benchmark.expected_tools)
    score += tool_score
    for tool in missing_tools:
        _add_issue(issues, "missing_tool", "major", f"Required tool is missing: {tool}")

    task_type_score, missing_task_types = _score_required_task_types(validated_tasks, benchmark.expected_task_types)
    score += task_type_score
    for task_type in missing_task_types:
        _add_issue(issues, "missing_task_type", "major", f"Required task type is missing: {task_type}")

    dependency_score, dependency_issues = _score_dependency_order(validated_tasks)
    score += dependency_score
    issues.extend(dependency_issues)

    entity_score, missing_entities = _score_entity_coverage(validated_tasks, benchmark.intent)
    score += entity_score
    for entity in missing_entities:
        _add_issue(issues, "missing_entity", "minor", f"Entity is not reflected in task params: {entity}")

    constraint_score, covered_constraints, missing_constraints = _score_constraint_coverage(validated_tasks, benchmark)
    score += constraint_score
    for signal in missing_constraints:
        _add_issue(issues, "missing_constraint", "major", f"User constraint is not reflected in plan: {signal}")

    normalized_score = round(score / max_score, 4)
    return PlannerEvaluationResult(
        case_id=benchmark.case_id,
        name=benchmark.name,
        score=score,
        max_score=max_score,
        normalized_score=normalized_score,
        passed=normalized_score >= benchmark.min_normalized_score and not _has_critical_issue(issues),
        metrics={
            "task_count": len(validated_tasks),
            "tool_count": len([task for task in validated_tasks if task.tool]),
            "required_tool_coverage": _coverage_ratio(
                {task.tool for task in validated_tasks if task.tool},
                set(benchmark.expected_tools),
            ),
            "required_task_type_coverage": _coverage_ratio(
                {task.task_type for task in validated_tasks},
                set(benchmark.expected_task_types),
            ),
            "min_normalized_score": benchmark.min_normalized_score,
        },
        covered_constraints=covered_constraints,
        missing_constraints=missing_constraints,
        issues=issues,
    )


def evaluate_benchmark_suite(
    benchmark_results: Iterable[PlannerEvaluationResult],
) -> Dict[str, Any]:
    """Aggregate per-case planner evaluation results into a compact suite summary."""

    results = list(benchmark_results)
    average_score = 0.0
    if results:
        average_score = round(sum(result.normalized_score for result in results) / len(results), 4)
    return {
        "case_count": len(results),
        "passed_count": len([result for result in results if result.passed]),
        "failed_count": len([result for result in results if not result.passed]),
        "average_normalized_score": average_score,
        "passed": all(result.passed for result in results),
    }


def _validate_tasks(tasks: Sequence[Dict[str, Any]], issues: List[PlannerEvaluationIssue]) -> List[PlanningTask]:
    validated_tasks: List[PlanningTask] = []
    for index, task in enumerate(tasks):
        try:
            validated_tasks.append(PlanningTask.model_validate(task))
        except Exception as exc:
            _add_issue(issues, "task_schema_invalid", "critical", f"Task at index {index} is invalid: {exc}")
    return validated_tasks


def _score_safety(
    tasks: Sequence[PlanningTask],
    allowed_task_types: Set[str],
    allowed_tools: Set[str],
) -> tuple[int, List[PlannerEvaluationIssue]]:
    issues: List[PlannerEvaluationIssue] = []
    if not tasks:
        return 0, issues

    score = 15
    task_ids: Set[str] = set()
    for task in tasks:
        if not task.task_id or task.task_id in task_ids:
            score -= 5
            _add_issue(issues, "duplicate_task_id", "critical", f"Duplicate or empty task id: {task.task_id}")
        task_ids.add(task.task_id)
        if task.task_type not in allowed_task_types:
            score -= 5
            _add_issue(issues, "unsafe_task_type", "critical", f"Unexpected task type: {task.task_type}")
        if task.tool and task.tool not in allowed_tools:
            score -= 5
            _add_issue(issues, "unsafe_tool", "critical", f"Unexpected tool: {task.tool}")
        if task.task_type in {"ask_user", "recommend_destination"} and task.tool:
            score -= 5
            _add_issue(issues, "tool_not_allowed", "critical", f"{task.task_type} should not bind tool {task.tool}")
    return max(score, 0), issues


def _score_required_tools(tasks: Sequence[PlanningTask], expected_tools: Sequence[str]) -> tuple[int, List[str]]:
    if not expected_tools:
        return 20, []
    actual_tools = {task.tool for task in tasks if task.tool}
    expected = set(expected_tools)
    missing = sorted(expected - actual_tools)
    return round(20 * _coverage_ratio(actual_tools, expected)), missing


def _score_required_task_types(tasks: Sequence[PlanningTask], expected_task_types: Sequence[str]) -> tuple[int, List[str]]:
    if not expected_task_types:
        return 10, []
    actual_task_types = {task.task_type for task in tasks}
    expected = set(expected_task_types)
    missing = sorted(expected - actual_task_types)
    return round(10 * _coverage_ratio(actual_task_types, expected)), missing


def _score_dependency_order(tasks: Sequence[PlanningTask]) -> tuple[int, List[PlannerEvaluationIssue]]:
    issues: List[PlannerEvaluationIssue] = []
    if not tasks:
        return 0, issues
    seen: Set[str] = set()
    score = 15
    for task in tasks:
        for dependency in task.depends_on:
            if dependency not in seen:
                score -= 5
                _add_issue(
                    issues,
                    "invalid_dependency_order",
                    "critical",
                    f"Task {task.task_id} depends on unknown or later task {dependency}",
                )
        seen.add(task.task_id)
    return max(score, 0), issues


def _score_entity_coverage(tasks: Sequence[PlanningTask], intent: Dict[str, Any]) -> tuple[int, List[str]]:
    entities = intent.get("entities", {}) or {}
    required_entity_keys = ["origin", "destination", "duration", "budget", "travelers"]
    expected_entities = {
        key: value
        for key, value in entities.items()
        if key in required_entity_keys and value not in (None, "", [])
    }
    if not expected_entities:
        return 10, []

    serialized = _serialize_tasks(tasks)
    missing = []
    for key, value in expected_entities.items():
        if str(value) not in serialized and key not in serialized:
            missing.append(key)
    return round(10 * (1 - len(missing) / len(expected_entities))), missing


def _score_constraint_coverage(
    tasks: Sequence[PlanningTask],
    benchmark: PlannerBenchmarkCase,
) -> tuple[int, List[str], List[str]]:
    constraints = benchmark.expected_constraints
    if not constraints:
        return 15, [], []

    serialized = _serialize_tasks(tasks)
    covered: List[str] = []
    missing: List[str] = []
    for constraint in constraints:
        if any(keyword and keyword in serialized for keyword in constraint.keywords):
            covered.append(constraint.signal)
        else:
            missing.append(constraint.signal)
    return round(15 * len(covered) / len(constraints)), covered, missing


def _coverage_ratio(actual: Set[str], expected: Set[str]) -> float:
    if not expected:
        return 1.0
    return len(actual & expected) / len(expected)


def _serialize_tasks(tasks: Sequence[PlanningTask]) -> str:
    payload = [task.model_dump() for task in tasks]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _add_issue(issues: List[PlannerEvaluationIssue], code: str, severity: str, detail: str) -> None:
    issues.append(PlannerEvaluationIssue(code=code, severity=severity, detail=detail))


def _has_critical_issue(issues: Sequence[PlannerEvaluationIssue]) -> bool:
    return any(issue.severity == "critical" for issue in issues)
