"""Tests for the manual LLM intent smoke script."""
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "smoke_llm_intent.py"


def test_llm_intent_smoke_script_supports_dry_run_without_network():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--query",
            "想找一个安静不赶路的周末旅行地，预算两千左右",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        timeout=20,
    )

    payload = json.loads(result.stdout)
    preflight = payload["preflight"]
    assert preflight["rule_intent"] in {"general_chat", "travel_plan"}
    assert preflight["rule_confidence"] < 0.55
    assert preflight["would_use_llm_fallback"] is True
    assert "llm_available" in preflight
    assert "deepseek" in preflight["llm_provider"]
    assert "LLM_API_KEY" not in result.stdout
    assert "sk-" not in result.stdout


def test_llm_intent_smoke_script_documents_manual_options():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "--dry-run" in script
    assert "--require-llm" in script
    assert "--agent" in script
    assert "_sanitize_intent_metadata" in script
    assert "_sanitize_trace_summary" in script
