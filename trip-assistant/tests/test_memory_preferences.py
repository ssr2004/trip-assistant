"""
Memory用户偏好抽取与长期记忆测试
"""
import pytest

from core.agent import TravelAgent
from core.memory.long_term import LongTermMemory
from core.memory.manager import MemoryManager
from core.memory.preference_extractor import PreferenceExtractor
from core.planner import TaskPlanner


@pytest.fixture
def memory_paths(tmp_path):
    """测试记忆文件路径"""
    return {
        "long_term": str(tmp_path / "long_term_memory.json"),
        "episodic": str(tmp_path / "episodic_memory.json"),
    }


def test_preference_extractor_extracts_travel_preferences():
    """偏好抽取器可以从自然语言中抽取旅行偏好"""
    extractor = PreferenceExtractor()

    preference = extractor.extract("我喜欢慢节奏，不想太赶，酒店最好靠近地铁，喜欢自然风光和当地美食，不吃辣")

    assert "慢节奏" in preference.travel_styles
    assert "地铁附近" in preference.hotel_preferences
    assert "自然风光" in preference.attraction_preferences
    assert "当地美食" in preference.food_preferences
    assert "不吃辣" in preference.dietary_restrictions
    assert preference.preference_evidence["travel_styles"] == ["慢节奏"]
    assert preference.preference_evidence["hotel_preferences"] == ["地铁附近"]
    assert "慢节奏" in preference.raw_preferences
    assert preference.updated_at is not None


def test_preference_extractor_returns_empty_for_no_preference():
    """没有偏好关键词时不产生有效偏好"""
    extractor = PreferenceExtractor()

    preference = extractor.extract("你好")

    assert preference.has_preferences() is False
    assert preference.raw_preferences == []
    assert preference.updated_at is None


def test_long_term_memory_merges_and_deduplicates_preferences(memory_paths):
    """长期记忆可以合并偏好并自动去重"""
    memory = LongTermMemory(memory_paths["long_term"])

    memory.update_preferences({"travel_styles": ["慢节奏"], "hotel_preferences": ["地铁附近"]}, "user-1")
    memory.update_preferences({"travel_styles": ["慢节奏", "亲子"], "budget_preference": "经济型"}, "user-1")

    preferences = memory.get_preferences("user-1")
    assert preferences["travel_styles"] == ["慢节奏", "亲子"]
    assert preferences["hotel_preferences"] == ["地铁附近"]
    assert preferences["budget_preference"] == "经济型"
    assert preferences["planning_profile"]["used_preference_count"] >= 4
    assert "search_hotels" in preferences["planning_profile"]["tool_preferences"]


def test_long_term_memory_ignores_empty_preferences(memory_paths):
    """空偏好不会污染长期记忆"""
    memory = LongTermMemory(memory_paths["long_term"])

    preferences = memory.update_preferences({}, "user-1")

    assert preferences["travel_styles"] == []
    assert memory.preferences == {}


def test_memory_manager_extracts_saves_and_retrieves_preferences(memory_paths):
    """MemoryManager可以抽取、保存并检索用户偏好"""
    manager = MemoryManager(
        long_term_storage_path=memory_paths["long_term"],
        episodic_storage_path=memory_paths["episodic"],
    )

    manager.extract_and_save_preferences("我喜欢慢节奏，酒店靠近地铁，预算有限", "session-a")
    context = manager.retrieve("帮我规划杭州旅行", "session-a")

    assert "慢节奏" in context["preferences"]["travel_styles"]
    assert "地铁附近" in context["preferences"]["hotel_preferences"]
    assert context["preferences"]["budget_preference"] == "经济型"
    assert context["user_preferences"] == context["preferences"]
    assert context["planning_profile"]["budget_preference"] == "经济型"
    assert "search_hotels" in context["planning_profile"]["tool_preferences"]


def test_memory_profile_detects_conflicts_and_tool_scopes(memory_paths):
    """长期记忆画像能识别冲突并按工具分流偏好"""
    memory = LongTermMemory(memory_paths["long_term"])

    preferences = memory.update_preferences(
        {
            "travel_styles": ["慢节奏", "少走路"],
            "hotel_preferences": ["地铁附近", "高星级酒店"],
            "transport_preferences": ["高铁优先"],
            "attraction_preferences": ["自然风光"],
            "food_preferences": ["当地美食"],
            "budget_preference": "经济型",
            "excluded_preferences": ["不吃辣"],
        },
        "user-profile",
    )

    profile = preferences["planning_profile"]
    assert profile["budget_preference"] == "经济型"
    assert "经济型预算与高星级酒店偏好可能冲突" in profile["conflicts"]
    assert "地铁附近" in profile["tool_preferences"]["search_hotels"]
    assert "高铁优先" in profile["tool_preferences"]["search_flights"]
    assert "自然风光" in profile["tool_preferences"]["search_attractions"]
    assert "不吃辣" in profile["excluded_preferences"]


def test_planner_merges_memory_preferences_into_task_params():
    """任务规划器会把长期偏好合并到工具参数中"""
    planner = TaskPlanner()
    intent = {
        "intent": "travel_plan",
        "entities": {
            "origin": "郑州",
            "destination": "杭州",
            "departure_date": "2026-06-10",
            "duration": 3,
            "budget": 3000,
            "preferences": ["自然风光"],
        },
        "missing_slots": [],
        "confidence": 0.9,
    }
    context = {
        "query": "我要从郑州去杭州玩三天，预算3000，6月10日出发",
        "memory": {
            "preferences": {
                "travel_styles": ["慢节奏"],
                "hotel_preferences": ["地铁附近"],
                "attraction_preferences": [],
                "transport_preferences": [],
                "food_preferences": [],
                "raw_preferences": ["慢节奏", "地铁附近"],
            }
        },
    }

    tasks = planner.plan(intent, context)

    itinerary_task = next(task for task in tasks if task["task_id"] == "generate_itinerary_1")
    hotel_task = next(task for task in tasks if task["task_id"] == "search_hotels_1")
    assert itinerary_task["params"]["preferences"] == ["自然风光", "慢节奏", "地铁附近"]
    assert hotel_task["params"]["preferences"] == ["自然风光", "慢节奏", "地铁附近"]
    assert itinerary_task["params"]["memory_preference_source"] == "long_term_memory"
    assert itinerary_task["params"]["memory_profile"]["used_preference_count"] >= 2
    assert planner.last_plan_metadata["memory_personalization_applied"] is True


def test_planner_scopes_memory_preferences_by_tool():
    """Planner会把长期偏好按工具语义分流，而不是所有工具共享同一份偏好"""
    planner = TaskPlanner()
    intent = {
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
    }
    context = {
        "query": "我要从郑州去杭州玩三天",
        "memory": {
            "preferences": {
                "travel_styles": ["慢节奏"],
                "hotel_preferences": ["地铁附近"],
                "transport_preferences": ["高铁优先"],
                "attraction_preferences": ["自然风光"],
                "food_preferences": ["当地美食"],
                "budget_preference": "经济型",
            }
        },
    }

    tasks = planner.plan(intent, context)

    flight_task = next(task for task in tasks if task["task_id"] == "search_flights_1")
    hotel_task = next(task for task in tasks if task["task_id"] == "search_hotels_1")
    attraction_task = next(task for task in tasks if task["task_id"] == "search_attractions_1")
    itinerary_task = next(task for task in tasks if task["task_id"] == "generate_itinerary_1")
    assert "高铁优先" in flight_task["params"]["preferences"]
    assert "地铁附近" not in flight_task["params"]["preferences"]
    assert "地铁附近" in hotel_task["params"]["preferences"]
    assert "高铁优先" not in hotel_task["params"]["preferences"]
    assert "自然风光" in attraction_task["params"]["keywords"]
    assert "当地美食" in itinerary_task["params"]["preferences"]


@pytest.mark.asyncio
async def test_agent_uses_saved_preferences_in_later_planning(memory_paths):
    """Agent后续规划可以读取此前保存的长期偏好"""
    agent = TravelAgent()
    agent.memory_manager = MemoryManager(
        long_term_storage_path=memory_paths["long_term"],
        episodic_storage_path=memory_paths["episodic"],
    )

    await agent.arun("我喜欢慢节奏，酒店最好靠近地铁", "memory-session")
    retrieve_result = await agent._retrieve_context({
        "messages": [type("Message", (), {"content": "我要从郑州去杭州玩三天，预算3000，6月10日出发"})()],
        "session_id": "memory-session",
    })
    plan_result = await agent._plan_tasks({
        "messages": [type("Message", (), {"content": "我要从郑州去杭州玩三天，预算3000，6月10日出发"})()],
        "intent": {
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
        "rag_context": [],
        "memory_context": retrieve_result["memory_context"],
    })

    itinerary_task = next(task for task in plan_result["tasks"] if task["task_id"] == "generate_itinerary_1")
    assert "慢节奏" in itinerary_task["params"]["preferences"]
    assert "地铁附近" in itinerary_task["params"]["preferences"]
