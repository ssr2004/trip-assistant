"""Tests for the manual LLM planner smoke script."""
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "smoke_llm_planner.py"


def test_llm_planner_smoke_script_supports_dry_run_without_network():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--dry-run"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )

    payload = json.loads(result.stdout)
    preflight = payload["preflight"]
    assert preflight["llm_planner_enabled"] is False
    assert preflight["would_try_llm_planner"] is False
    assert preflight["skip_reason"] == "disabled"
    assert preflight["template_task_count"] >= 1
    assert payload["template_plan"]
    assert "LLM_API_KEY" not in result.stdout
    assert "sk-" not in result.stdout


def test_llm_planner_smoke_script_documents_manual_options():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "--dry-run" in script
    assert "--enable" in script
    assert "--require-llm-plan" in script
    assert "_sanitize_metadata" in script
    assert "_task_summary" in script
