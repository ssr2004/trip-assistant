"""Tests for the manual LLM planner smoke script."""
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "smoke_llm_planner.py"


def test_llm_planner_smoke_script_supports_dry_run_without_network():
    env = {
        **os.environ,
        "LLM_API_KEY": "test-key-for-dry-run",
        "LLM_BASE_URL": "https://api.deepseek.com",
        "LLM_PLANNER_MODE": "auto",
    }
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )

    payload = json.loads(result.stdout)
    preflight = payload["preflight"]
    assert preflight["llm_planner_enabled"] is True
    assert preflight["planner_mode_config"] == "auto"
    assert preflight["would_try_llm_planner"] is True
    assert preflight["skip_reason"] is None
    assert preflight["complexity_score"] >= 3
    assert preflight["complexity_signals"]
    assert preflight["template_task_count"] >= 1
    assert payload["template_plan"]
    assert "LLM_API_KEY" not in result.stdout
    assert "sk-" not in result.stdout


def test_llm_planner_smoke_script_documents_manual_options():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "--dry-run" in script
    assert "--enable" in script
    assert "--mode" in script
    assert "--require-llm-plan" in script
    assert "_sanitize_metadata" in script
    assert "_task_summary" in script
