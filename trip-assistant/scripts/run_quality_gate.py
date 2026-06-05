"""Run the unified local quality gate for TravelMind.

The gate intentionally uses deterministic defaults: benchmark scripts avoid
real provider calls unless their own flags are changed manually.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"


@dataclass(frozen=True)
class QualityStep:
    """One command in the quality gate."""

    name: str
    label: str
    command: list[str]
    cwd: Path = ROOT
    timeout_seconds: int = 240
    json_output: bool = False


CommandRunner = Callable[[QualityStep, dict[str, str]], subprocess.CompletedProcess[str]]


def build_quality_steps(
    *,
    include_frontend_e2e: bool = True,
    include_backend_tests: bool = True,
) -> list[QualityStep]:
    """Build the default deterministic quality gate command plan."""
    python = sys.executable
    npm = _npm_command()
    steps = []
    if include_backend_tests:
        steps.append(
            QualityStep(
                name="backend_tests",
                label="Backend pytest",
                command=[python, "-m", "pytest", "-q", "tests", "--tb=short"],
                timeout_seconds=300,
            )
        )
    steps.extend(
        [
            QualityStep(
                name="planner_quality",
                label="Planner quality benchmark",
                command=[python, "scripts/evaluate_planner_quality.py", "--json-compact"],
                timeout_seconds=120,
                json_output=True,
            ),
            QualityStep(
                name="agent_e2e",
                label="Agent E2E benchmark",
                command=[python, "scripts/evaluate_agent_e2e.py", "--json-compact"],
                timeout_seconds=180,
                json_output=True,
            ),
            QualityStep(
                name="rag_quality",
                label="RAG quality benchmark",
                command=[python, "scripts/evaluate_rag_quality.py", "--json-compact"],
                timeout_seconds=120,
                json_output=True,
            ),
            QualityStep(
                name="frontend_build",
                label="Frontend TypeScript/Vite build",
                command=[npm, "run", "build"],
                cwd=FRONTEND_DIR,
                timeout_seconds=180,
            ),
        ]
    )
    if include_frontend_e2e:
        steps.append(
            QualityStep(
                name="frontend_e2e",
                label="Frontend Playwright E2E",
                command=[npm, "run", "test:e2e"],
                cwd=FRONTEND_DIR,
                timeout_seconds=180,
            )
        )
    return steps


def run_quality_gate(
    steps: list[QualityStep],
    *,
    runner: CommandRunner | None = None,
    stop_on_failure: bool = False,
) -> dict[str, Any]:
    """Run all quality steps and aggregate a machine-readable result."""
    started_at = time.perf_counter()
    runner = runner or run_step
    env = build_command_env()
    results = []
    for step in steps:
        result = run_single_step(step, runner=runner, env=env)
        results.append(result)
        if stop_on_failure and not result["passed"]:
            break

    passed_count = sum(1 for result in results if result["passed"])
    failed_count = len(results) - passed_count
    skipped_count = len(steps) - len(results)
    return {
        "suite": {
            "passed": failed_count == 0 and skipped_count == 0,
            "step_count": len(steps),
            "executed_count": len(results),
            "passed_count": passed_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "duration_seconds": round(time.perf_counter() - started_at, 3),
        },
        "steps": results,
    }


def run_single_step(
    step: QualityStep,
    *,
    runner: CommandRunner,
    env: dict[str, str],
) -> dict[str, Any]:
    """Run one quality step and convert the process result to the gate schema."""
    started_at = time.perf_counter()
    try:
        completed = runner(step, env)
        duration = round(time.perf_counter() - started_at, 3)
        parsed_json = _parse_json_output(completed.stdout) if step.json_output else None
        return {
            "name": step.name,
            "label": step.label,
            "command": step.command,
            "cwd": str(step.cwd),
            "duration_seconds": duration,
            "exit_code": completed.returncode,
            "passed": completed.returncode == 0,
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
            "suite": parsed_json.get("suite") if isinstance(parsed_json, dict) else None,
        }
    except subprocess.TimeoutExpired as exc:
        duration = round(time.perf_counter() - started_at, 3)
        return {
            "name": step.name,
            "label": step.label,
            "command": step.command,
            "cwd": str(step.cwd),
            "duration_seconds": duration,
            "exit_code": None,
            "passed": False,
            "stdout_tail": _tail(_decode_timeout_output(exc.stdout)),
            "stderr_tail": _tail(_decode_timeout_output(exc.stderr) or "command timed out"),
            "suite": None,
            "error_type": "timeout",
        }


def run_step(step: QualityStep, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Default subprocess runner."""
    return subprocess.run(
        step.command,
        cwd=step.cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=step.timeout_seconds,
        check=False,
    )


def build_command_env() -> dict[str, str]:
    """Build environment for stable local command output."""
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def print_human_summary(payload: dict[str, Any]) -> None:
    """Print compact human-readable quality gate output."""
    suite = payload["suite"]
    status = "PASSED" if suite["passed"] else "FAILED"
    print(
        f"Quality Gate {status}: "
        f"{suite['passed_count']}/{suite['step_count']} passed, "
        f"{suite['failed_count']} failed, {suite['skipped_count']} skipped, "
        f"{suite['duration_seconds']}s"
    )
    for step in payload["steps"]:
        marker = "PASS" if step["passed"] else "FAIL"
        print(f"- [{marker}] {step['name']} ({step['duration_seconds']}s)")
        if not step["passed"] and step.get("stderr_tail"):
            print(f"  stderr: {step['stderr_tail']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TravelMind unified quality gate.")
    parser.add_argument("--json", action="store_true", help="Print formatted JSON result.")
    parser.add_argument("--json-compact", action="store_true", help="Print compact JSON result.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failed step.")
    parser.add_argument("--skip-frontend-e2e", action="store_true", help="Skip Playwright E2E for quicker local checks.")
    parser.add_argument("--skip-backend-tests", action="store_true", help="Skip full pytest suite.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    steps = build_quality_steps(
        include_frontend_e2e=not args.skip_frontend_e2e,
        include_backend_tests=not args.skip_backend_tests,
    )
    payload = run_quality_gate(steps, stop_on_failure=args.stop_on_failure)
    if args.json or args.json_compact:
        print(json.dumps(payload, ensure_ascii=False, indent=None if args.json_compact else 2))
    else:
        print_human_summary(payload)
    return 0 if payload["suite"]["passed"] else 2


def _npm_command() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def _parse_json_output(stdout: str) -> dict[str, Any] | None:
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _tail(value: str | None, max_lines: int = 12, max_chars: int = 2000) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    lines = text.splitlines()[-max_lines:]
    compact = "\n".join(lines)
    return compact[-max_chars:]


def _decode_timeout_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
