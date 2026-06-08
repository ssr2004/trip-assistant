"""
任务依赖结果注入测试
"""
import pytest

from core.agent import TravelAgent
from core.planner import TaskPlanner


@pytest.fixture
def agent():
    """创建Agent实例"""
    return TravelAgent()


@pytest.mark.asyncio
async def test_agent_injects_dependency_context_into_itinerary(agent):
    """Agent执行器会把前置工具结果注入给行程生成任务"""
    tasks = [
        {
            "task_id": "search_trains_1",
            "task_type": "tool_call",
            "name": "搜索火车",
            "priority": 1,
            "tool": "search_trains",
            "params": {"origin": "郑州", "destination": "杭州", "date": "2026-06-10"},
            "depends_on": [],
        },
        {
            "task_id": "search_flights_1",
            "task_type": "tool_call",
            "name": "搜索航班",
            "priority": 2,
            "tool": "search_flights",
            "params": {"origin": "郑州", "destination": "杭州", "date": "2026-06-10"},
            "depends_on": [],
        },
        {
            "task_id": "search_hotels_1",
            "task_type": "tool_call",
            "name": "搜索酒店",
            "priority": 3,
            "tool": "search_hotels",
            "params": {"location": "杭州"},
            "depends_on": [],
        },
        {
            "task_id": "search_attractions_1",
            "task_type": "tool_call",
            "name": "搜索景点",
            "priority": 4,
            "tool": "search_attractions",
            "params": {"location": "杭州"},
            "depends_on": [],
        },
        {
            "task_id": "retrieve_guide_1",
            "task_type": "tool_call",
            "name": "检索旅行攻略",
            "priority": 5,
            "tool": "retrieve_guide",
            "params": {"query": "杭州三天旅行攻略", "destination": "杭州"},
            "depends_on": [],
        },
        {
            "task_id": "get_weather_forecast_1",
            "task_type": "tool_call",
            "name": "查询天气",
            "priority": 6,
            "tool": "get_weather_forecast",
            "params": {"city": "杭州", "days": 3},
            "depends_on": [],
        },
        {
            "task_id": "generate_itinerary_1",
            "task_type": "generate_itinerary",
            "name": "生成旅行行程",
            "priority": 7,
            "tool": "generate_itinerary",
            "params": {"origin": "郑州", "destination": "杭州", "duration": 3, "budget": 3000},
            "depends_on": [
                "search_trains_1",
                "search_flights_1",
                "search_hotels_1",
                "search_attractions_1",
                "retrieve_guide_1",
                "get_weather_forecast_1",
            ],
        },
    ]

    result = await agent._execute_tasks({"tasks": tasks})

    itinerary_result = result["task_results"][-1]["result"]
    context_summary = itinerary_result["data"]["context_summary"]
    assert context_summary["train_count"] == 0
    assert context_summary["flight_count"] == 0
    assert context_summary["hotel_count"] == 0
    assert context_summary["attraction_count"] == 4
    assert context_summary["has_guide"] is True
    assert context_summary["has_weather"] is True
    assert context_summary["weather_forecast_count"] >= 1
    assert context_summary["dependency_error_count"] == 3

    itinerary_meta = result["task_results"][-1]["meta"]
    assert itinerary_meta["dependency_ids"] == [
        "search_trains_1",
        "search_flights_1",
        "search_hotels_1",
        "search_attractions_1",
        "retrieve_guide_1",
        "get_weather_forecast_1",
    ]
    assert itinerary_meta["resolved_dependencies"] == [
        "search_attractions_1",
        "retrieve_guide_1",
        "get_weather_forecast_1",
    ]
    assert itinerary_meta["missing_dependencies"] == []
    assert itinerary_meta["failed_dependencies"] == ["search_trains_1", "search_flights_1", "search_hotels_1"]
    assert set(itinerary_meta["dependency_context_keys"]) >= {"attractions", "guide", "weather", "errors"}
    assert "trains" not in itinerary_meta["dependency_context_keys"]
    assert "flights" not in itinerary_meta["dependency_context_keys"]
    assert "hotels" not in itinerary_meta["dependency_context_keys"]


@pytest.mark.asyncio
async def test_agent_keeps_running_when_dependency_fails(agent):
    """依赖任务失败时后续任务仍可执行，并记录依赖错误"""
    tasks = [
        {
            "task_id": "search_flights_1",
            "task_type": "tool_call",
            "name": "搜索航班",
            "priority": 1,
            "tool": "search_flights",
            "params": {"origin": "郑州"},
            "depends_on": [],
        },
        {
            "task_id": "generate_itinerary_1",
            "task_type": "generate_itinerary",
            "name": "生成旅行行程",
            "priority": 2,
            "tool": "generate_itinerary",
            "params": {"origin": "郑州", "destination": "杭州", "duration": 2},
            "depends_on": ["search_flights_1"],
        },
    ]

    result = await agent._execute_tasks({"tasks": tasks})

    flight_result = result["task_results"][0]
    itinerary_result = result["task_results"][1]
    context_summary = itinerary_result["result"]["data"]["context_summary"]
    assert flight_result["success"] is False
    assert itinerary_result["success"] is True
    assert context_summary["flight_count"] == 0
    assert context_summary["dependency_error_count"] == 1
    assert itinerary_result["meta"]["failed_dependencies"] == ["search_flights_1"]
    assert itinerary_result["meta"]["dependency_error_count"] == 1
    assert flight_result["meta"]["failure_category"] == "missing_required_params"
    assert flight_result["meta"]["recoverable"] is True
    assert flight_result["meta"]["degraded"] is True
    assert flight_result["meta"]["fallback_used"] is False
    assert flight_result["meta"]["recovery_strategy"] == "continue_with_error_context"
    assert itinerary_result["meta"]["failure_category"] == "dependency_failed"
    assert itinerary_result["meta"]["recoverable"] is True
    assert itinerary_result["meta"]["degraded"] is True
    assert itinerary_result["meta"]["recovery_strategy"] == "partial_dependency_context"


@pytest.mark.asyncio
async def test_agent_recovers_when_tool_raises_exception(agent):
    """工具抛异常时 Executor 记录可恢复失败，并继续执行依赖下游任务。"""
    original_registry = agent.tool_registry

    class FailingAttractionRegistry:
        async def execute(self, tool_name, params):
            if tool_name == "search_attractions":
                raise RuntimeError("provider timeout")
            return await original_registry.execute(tool_name, params)

    agent.tool_registry = FailingAttractionRegistry()
    tasks = [
        {
            "task_id": "search_attractions_1",
            "task_type": "tool_call",
            "name": "搜索景点",
            "priority": 1,
            "tool": "search_attractions",
            "params": {"location": "杭州"},
            "depends_on": [],
        },
        {
            "task_id": "generate_itinerary_1",
            "task_type": "generate_itinerary",
            "name": "生成旅行行程",
            "priority": 2,
            "tool": "generate_itinerary",
            "params": {"origin": "郑州", "destination": "杭州", "duration": 2},
            "depends_on": ["search_attractions_1"],
        },
    ]

    result = await agent._execute_tasks({"tasks": tasks})

    failed_result = result["task_results"][0]
    itinerary_result = result["task_results"][1]
    assert failed_result["success"] is False
    assert failed_result["meta"]["failure_category"] == "exception"
    assert failed_result["meta"]["recoverable"] is True
    assert failed_result["meta"]["recovery_strategy"] == "continue_with_error_context"
    assert itinerary_result["success"] is True
    assert itinerary_result["meta"]["failed_dependencies"] == ["search_attractions_1"]
    assert itinerary_result["meta"]["recovery_strategy"] == "partial_dependency_context"


def test_dependency_context_does_not_mutate_original_params(agent):
    """依赖注入不会修改原始任务参数"""
    task = {
        "task_id": "generate_itinerary_1",
        "params": {"destination": "杭州"},
        "depends_on": ["missing_task"],
    }

    injected = agent._inject_dependency_context(task, task["params"], {})

    assert "context" not in task["params"]
    assert "missing_task" in injected["context"]["errors"]
    assert injected["context"]["missing_dependencies"] == ["missing_task"]


def test_dependency_metadata_indexes_missing_and_failed_dependencies(agent):
    """Dependency metadata gives trace-safe indexes for downstream tasks."""
    failed_result = {
        "task": {"task_id": "search_flights_1", "task_type": "tool_call", "tool": "search_flights"},
        "success": False,
        "result": {"success": False, "data": {"flights": []}, "error": "missing destination"},
        "error": "missing destination",
    }

    metadata = agent._resolve_dependency_metadata(
        ["search_flights_1", "missing_task"],
        {"search_flights_1": failed_result},
    )

    assert metadata["dependency_ids"] == ["search_flights_1", "missing_task"]
    assert metadata["resolved_dependencies"] == []
    assert metadata["failed_dependencies"] == ["search_flights_1"]
    assert metadata["missing_dependencies"] == ["missing_task"]
    assert metadata["dependency_error_count"] == 2
    assert metadata["dependency_context_keys"] == ["errors"]


def test_template_planner_declares_itinerary_dependencies():
    """模板规划会声明行程生成依赖的前置任务"""
    planner = TaskPlanner()
    tasks = planner.plan(
        {
            "intent": "travel_plan",
            "entities": {
                "origin": "郑州",
                "destination": "杭州",
                "departure_date": "2026-06-10",
                "duration": 3,
                "budget": 3000,
                "preferences": [],
            },
            "missing_slots": [],
            "confidence": 0.9,
        },
        {"query": "我要从郑州去杭州玩三天，预算3000，6月10日出发"},
    )

    itinerary_task = next(task for task in tasks if task["task_id"] == "generate_itinerary_1")
    assert itinerary_task["depends_on"] == [
        "search_trains_1",
        "search_flights_1",
        "search_hotels_1",
        "search_attractions_1",
        "retrieve_guide_1",
    ]
