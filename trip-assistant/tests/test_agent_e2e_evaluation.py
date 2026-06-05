"""Tests for end-to-end Agent benchmark evaluation."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import settings
from core.agent import TravelAgent
from core.agent_evaluation import (
    evaluate_agent_scenario,
    get_default_agent_e2e_benchmarks,
    summarize_agent_evaluation,
)
from core.memory.manager import MemoryManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_agent_e2e.py"


@pytest.mark.asyncio
async def test_agent_e2e_benchmarks_pass_with_deterministic_fallbacks(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LLM_API_KEY", "")
    monkeypatch.setattr(settings, "LLM_PLANNER_MODE", "off")
    monkeypatch.setattr(settings, "EXTERNAL_API_MOCK_ENABLED", True)

    agent = TravelAgent()
    agent.memory_manager = MemoryManager(
        long_term_storage_path=str(tmp_path / "long_term_memory.json"),
        episodic_storage_path=str(tmp_path / "episodic_memory.json"),
    )

    results = []
    for scenario in get_default_agent_e2e_benchmarks():
        results.append(
            await evaluate_agent_scenario(
                agent,
                scenario,
                session_id=f"test-agent-e2e-{scenario.scenario_id}",
                sensitive_markers=["AMAP_API_KEY", "WEATHER_API_KEY", "LLM_API_KEY", "test-secret-key"],
            )
        )

    suite = summarize_agent_evaluation(results)
    assert suite["passed"] is True
    assert suite["scenario_count"] >= 3
    assert suite["turn_count"] >= 7
    assert suite["pass_rate"] == 1.0
    assert all(not result.issues for result in results)


def test_agent_e2e_script_outputs_parseable_sanitized_json():
    env = {
        **os.environ,
        "LLM_API_KEY": "test-secret-key-that-must-not-leak",
        "AMAP_API_KEY": "test-amap-key-that-must-not-leak",
        "WEATHER_API_KEY": "test-weather-key-that-must-not-leak",
    }

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--json-compact"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
        timeout=60,
    )

    payload = json.loads(result.stdout)
    assert payload["suite"]["passed"] is True
    assert payload["suite"]["deterministic_mode"] is True
    assert payload["suite"]["passed_turn_count"] == payload["suite"]["turn_count"]
    assert "test-secret-key-that-must-not-leak" not in result.stdout
    assert "test-amap-key-that-must-not-leak" not in result.stdout
    assert "test-weather-key-that-must-not-leak" not in result.stdout
    assert "LLM_API_KEY" not in result.stdout
    assert "AMAP_API_KEY" not in result.stdout
