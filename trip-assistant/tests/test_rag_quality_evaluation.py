"""RAG质量评测脚本测试。"""

import json
import os
import subprocess
import sys


def test_rag_quality_script_passes_with_local_fallback():
    """RAG质量脚本默认不调用真实embedding服务，并通过基准集。"""
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_rag_quality.py", "--json-compact"],
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["suite"]["passed"] is True
    assert payload["suite"]["top1_accuracy"] >= 0.75
    assert payload["suite"]["recall_at_k"] == 1.0
    assert payload["suite"]["use_configured_embedding"] is False
    assert "sk-" not in completed.stdout
