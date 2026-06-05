"""Tests for the unified quality gate runner."""

from pathlib import Path
import subprocess

from scripts.run_quality_gate import QualityStep, build_quality_steps, run_quality_gate


def test_quality_gate_builds_default_command_plan():
    """默认质量门禁包含后端测试、三组评测和前端验证。"""
    steps = build_quality_steps()

    names = [step.name for step in steps]
    assert names == [
        "backend_tests",
        "planner_quality",
        "agent_e2e",
        "rag_quality",
        "frontend_build",
        "frontend_e2e",
    ]
    assert any("pytest" in part for part in steps[0].command)
    assert steps[1].command[-1] == "--json-compact"
    assert steps[1].json_output is True
    assert steps[-1].cwd.name == "frontend"


def test_quality_gate_can_skip_expensive_steps():
    """门禁脚本支持跳过后端全量测试和前端E2E。"""
    steps = build_quality_steps(include_backend_tests=False, include_frontend_e2e=False)

    assert [step.name for step in steps] == [
        "planner_quality",
        "agent_e2e",
        "rag_quality",
        "frontend_build",
    ]


def test_quality_gate_aggregates_success_and_parses_json_suite():
    """成功执行时聚合每个阶段结果，并解析评测脚本suite。"""
    steps = [
        QualityStep(name="planner_quality", label="Planner", command=["planner"], json_output=True),
        QualityStep(name="frontend_build", label="Build", command=["build"], cwd=Path("frontend")),
    ]
    calls = []

    def fake_runner(step, env):
        calls.append((step.name, env.get("PYTHONIOENCODING")))
        if step.json_output:
            return subprocess.CompletedProcess(
                step.command,
                0,
                stdout='{"suite":{"passed":true,"case_count":3}}',
                stderr="",
            )
        return subprocess.CompletedProcess(step.command, 0, stdout="built", stderr="")

    payload = run_quality_gate(steps, runner=fake_runner)

    assert payload["suite"]["passed"] is True
    assert payload["suite"]["passed_count"] == 2
    assert payload["suite"]["failed_count"] == 0
    assert payload["steps"][0]["suite"] == {"passed": True, "case_count": 3}
    assert calls == [("planner_quality", "utf-8"), ("frontend_build", "utf-8")]


def test_quality_gate_reports_failure_and_stop_on_failure():
    """失败时返回非通过suite，并可按配置短路后续阶段。"""
    steps = [
        QualityStep(name="backend_tests", label="Pytest", command=["pytest"]),
        QualityStep(name="planner_quality", label="Planner", command=["planner"]),
    ]

    def fake_runner(step, env):
        return subprocess.CompletedProcess(step.command, 1, stdout="", stderr=f"{step.name} failed")

    payload = run_quality_gate(steps, runner=fake_runner, stop_on_failure=True)

    assert payload["suite"]["passed"] is False
    assert payload["suite"]["executed_count"] == 1
    assert payload["suite"]["failed_count"] == 1
    assert payload["suite"]["skipped_count"] == 1
    assert payload["steps"][0]["passed"] is False
    assert "backend_tests failed" in payload["steps"][0]["stderr_tail"]
