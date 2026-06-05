"""Tests for planner quality benchmark scoring."""
import json
import os
import subprocess
import sys
from pathlib import Path

from core.planner import TaskPlanner
from core.planner_evaluation import (
    evaluate_benchmark_suite,
    evaluate_planner_tasks,
    get_default_planner_benchmarks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_planner_quality.py"


def test_default_planner_benchmarks_have_complex_constraints():
    benchmarks = get_default_planner_benchmarks()

    assert len(benchmarks) >= 3
    assert all(case.case_id for case in benchmarks)
    assert any("rain_backup" in [constraint.signal for constraint in case.expected_constraints] for case in benchmarks)
    assert any(case.expected_tools for case in benchmarks)


def test_template_planner_passes_default_quality_benchmarks():
    planner = TaskPlanner(llm_planner_mode="off")
    results = []

    for benchmark in get_default_planner_benchmarks():
        tasks = planner.plan(benchmark.intent, {**benchmark.context, "query": benchmark.query})
        result = evaluate_planner_tasks(tasks, benchmark)
        results.append(result)

        assert result.passed is True
        assert result.normalized_score >= benchmark.min_normalized_score
        assert not result.missing_constraints
        assert not [issue for issue in result.issues if issue.severity == "critical"]
        assert not [issue for issue in result.issues if issue.code == "missing_tool"]

    suite = evaluate_benchmark_suite(results)
    assert suite["passed"] is True
    assert suite["average_normalized_score"] == 1.0


def test_planner_evaluation_flags_unsafe_or_incomplete_plan():
    benchmark = get_default_planner_benchmarks()[0]
    unsafe_tasks = [
        {
            "task_id": "unsafe_1",
            "task_type": "tool_call",
            "name": "Unknown tool",
            "priority": 1,
            "tool": "book_private_supplier",
            "params": {"destination": "杭州"},
            "reason": "unsafe external action",
            "depends_on": [],
        },
        {
            "task_id": "generate_itinerary_1",
            "task_type": "generate_itinerary",
            "name": "Generate itinerary",
            "priority": 2,
            "tool": "generate_itinerary",
            "params": {"destination": "杭州"},
            "reason": "final plan",
            "depends_on": ["missing_task"],
        },
    ]

    result = evaluate_planner_tasks(unsafe_tasks, benchmark)

    assert result.passed is False
    assert result.normalized_score < benchmark.min_normalized_score
    assert "budget" in result.missing_constraints
    assert any(issue.code == "unsafe_tool" for issue in result.issues)
    assert any(issue.code == "invalid_dependency_order" for issue in result.issues)


def test_planner_quality_script_runs_without_llm_or_key_leakage():
    env = {
        **os.environ,
        "LLM_API_KEY": "test-key-that-must-not-leak",
        "LLM_BASE_URL": "https://api.deepseek.com",
    }

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--json-compact"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        timeout=30,
    )

    payload = json.loads(result.stdout)
    assert payload["suite"]["passed"] is True
    assert payload["suite"]["attempt_llm"] is False
    assert payload["suite"]["case_count"] >= 3
    assert payload["suite"]["average_normalized_score"] == 1.0
    assert "LLM_API_KEY" not in result.stdout
    assert "test-key-that-must-not-leak" not in result.stdout
