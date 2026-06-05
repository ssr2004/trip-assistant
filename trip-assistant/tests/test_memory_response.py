"""
Memory上下文驱动的个性化响应测试
"""
import pytest

from core.agent import TravelAgent
from core.memory.manager import MemoryManager
from core.response_builder import ResponseBuilder


def _minimal_travel_plan_results():
    """构造最小完整旅行规划结果"""
    return [
        {
            "task": {"task_type": "tool_call", "tool": "generate_itinerary", "name": "生成旅行行程"},
            "success": True,
            "result": {
                "success": True,
                "data": {
                    "origin": "郑州",
                    "destination": "杭州",
                    "duration": 3,
                    "budget": 3000,
                    "preferences": ["慢节奏", "地铁附近"],
                    "itinerary": [
                        {
                            "day": 1,
                            "title": "抵达杭州与西湖初体验",
                            "activities": ["抵达杭州", "游览西湖"],
                            "notes": "第一天以适应节奏为主。",
                        }
                    ],
                    "budget_summary": {
                        "input_budget": 3000,
                        "estimated_hotel": 1200,
                        "estimated_food": 540,
                        "estimated_local_transport": 240,
                        "estimated_attractions": 360,
                        "estimated_total_without_flight": 2340,
                        "budget_note": "预算基本可以覆盖当地开销。",
                    },
                },
                "error": None,
                "metadata": {"tool": "generate_itinerary"},
            },
            "error": None,
        }
    ]


def test_travel_plan_response_shows_memory_preference_note():
    """完整旅行规划回复会自然体现长期偏好"""
    builder = ResponseBuilder()

    response = builder.build(
        intent={"intent": "travel_plan", "entities": {"origin": "郑州", "destination": "杭州", "duration": 3}},
        task_results=_minimal_travel_plan_results(),
        memory_context={
            "preferences": {
                "travel_styles": ["慢节奏"],
                "hotel_preferences": ["地铁附近"],
                "attraction_preferences": ["自然风光"],
                "raw_preferences": ["慢节奏", "地铁附近"],
            }
        },
    )

    assert "个性化参考" in response
    assert "慢节奏" in response
    assert "地铁附近住宿" in response
    assert "自然风光体验" in response
    assert "一、出行概览" in response


def test_travel_plan_response_hides_note_without_memory_preferences():
    """没有长期偏好时不展示个性化说明"""
    builder = ResponseBuilder()

    response = builder.build(
        intent={"intent": "travel_plan", "entities": {"origin": "郑州", "destination": "杭州", "duration": 3}},
        task_results=_minimal_travel_plan_results(),
        memory_context={"preferences": {"travel_styles": [], "hotel_preferences": []}},
    )

    assert "个性化参考" not in response


def test_followup_response_hides_memory_preference_note():
    """追问场景不展示长期偏好说明"""
    builder = ResponseBuilder()

    response = builder.build(
        intent={"intent": "travel_plan"},
        task_results=[
            {
                "task": {"task_type": "ask_user", "name": "补充缺失信息"},
                "success": True,
                "result": {"missing_slots": ["departure_date"], "question": "请问您什么时候出发？"},
                "error": None,
            }
        ],
        memory_context={"preferences": {"travel_styles": ["慢节奏"]}},
    )

    assert "请问您什么时候出发" in response
    assert "个性化参考" not in response


def test_policy_response_hides_memory_preference_note():
    """政策查询不展示长期偏好说明"""
    builder = ResponseBuilder()

    response = builder.build(
        intent={"intent": "policy_query"},
        task_results=[
            {
                "task": {"task_type": "tool_call", "tool": "retrieve_policy", "name": "检索政策文档"},
                "success": True,
                "result": {
                    "success": True,
                    "data": {"query": "机票能退吗", "answer": "机票可按规则退改。", "sources": []},
                    "error": None,
                    "metadata": {"tool": "retrieve_policy"},
                },
                "error": None,
            }
        ],
        memory_context={"preferences": {"travel_styles": ["慢节奏"]}},
    )

    assert "关于您的政策问题" in response
    assert "个性化参考" not in response


@pytest.mark.asyncio
async def test_agent_memory_preferences_appear_in_later_response(tmp_path):
    """Agent后续完整规划回复会体现此前保存的长期偏好"""
    agent = TravelAgent()
    agent.memory_manager = MemoryManager(
        long_term_storage_path=str(tmp_path / "long_term_memory.json"),
        episodic_storage_path=str(tmp_path / "episodic_memory.json"),
    )

    await agent.arun("我喜欢慢节奏，酒店最好靠近地铁", "memory-response-session")
    response = await agent.arun("我要从郑州去杭州玩三天，预算3000，6月10日出发", "memory-response-session")

    assert "个性化参考" in response
    assert "慢节奏" in response
    assert "地铁附近住宿" in response
    assert "不生成航班号、票价、舱位、余票、酒店房态、房型或可订价格" in response
    assert "东方航空" not in response
    assert "杭州西湖国宾馆" not in response
    assert "每日行程" in response
    assert "工具不存在" not in response
