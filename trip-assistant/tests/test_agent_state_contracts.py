"""Agent runtime state and task result contract tests."""

import time

from langchain_core.messages import AIMessage

from core.agent import TravelAgent
from core.state import TASK_RESULT_META_FIELDS, TASK_RESULT_REQUIRED_FIELDS


def test_initial_state_has_stable_runtime_boundaries():
    agent = TravelAgent()

    state = agent._build_initial_state("杭州怎么玩？", "state-contract-session")

    assert state["messages"][0].content == "杭州怎么玩？"
    assert state["intent"] is None
    assert state["tasks"] == []
    assert state["task_results"] == []
    assert state["rag_context"] == []
    assert state["memory_context"] == {}
    assert state["dynamic_rag_context"] == {}
    assert state["session_id"] == "state-contract-session"


def test_finalized_task_result_has_required_executor_contract():
    agent = TravelAgent()
    task = {"task_id": "ask_user_1", "task_type": "ask_user", "name": "补充信息"}

    task_result = agent._finalize_task_result(
        agent._build_internal_result(task, {"question": "请补充出发日期"}),
        time.perf_counter(),
    )

    assert TASK_RESULT_REQUIRED_FIELDS.issubset(task_result.keys())
    assert set(task_result["meta"]).issubset(TASK_RESULT_META_FIELDS)
    assert task_result["task"] == task
    assert task_result["success"] is True
    assert task_result["error"] is None
    assert isinstance(task_result["meta"]["duration_ms"], int)
    assert task_result["meta"]["execution_mode"] == "internal_rule"
    assert task_result["meta"]["recovery_strategy"] == "none"


def test_agent_output_boundary_normalizes_artifacts_and_trace():
    agent = TravelAgent()
    task = {"task_id": "generate_itinerary_1", "task_type": "generate_itinerary", "tool": "generate_itinerary"}
    task_result = agent._finalize_task_result(
        agent._build_internal_result(
            task,
            {
                "origin": "郑州",
                "destination": "杭州",
                "duration": 1,
                "itinerary": [{"day": 1, "title": "西湖", "activities": ["西湖"]}],
                "budget_summary": {},
                "summary": "一日行程",
            },
        ),
        time.perf_counter(),
    )

    output = agent._build_agent_output(
        {
            "messages": [AIMessage(content="已生成")],
            "intent": {"intent": "travel_plan", "metadata": {"source": "rule"}},
            "tasks": [task],
            "task_results": [task_result],
            "rag_context": [],
            "dynamic_rag_context": {},
        }
    )

    assert output["response"] == "已生成"
    assert output["artifacts"]["itinerary"]["destination"] == "杭州"
    assert output["execution_trace"]["summary"]["intent"] == "travel_plan"
    assert "AMAP_API_KEY" not in str(output)
