"""
智能任务规划模块测试
"""
from core.planner import TaskPlanner


def test_plan_followup_when_required_slot_missing():
    """关键信息缺失时生成追问任务"""
    planner = TaskPlanner()
    intent = {
        "intent": "travel_plan",
        "entities": {
            "origin": "郑州",
            "destination": "杭州",
            "departure_date": None,
            "duration": 3,
            "budget": 3000,
            "preferences": [],
        },
        "missing_slots": ["departure_date"],
        "followup_question": "请问您计划什么时候从郑州出发？",
    }

    tasks = planner.plan(intent, {"query": "我要从郑州去杭州玩三天"})

    assert len(tasks) == 1
    assert tasks[0]["task_type"] == "ask_user"
    assert tasks[0]["params"]["question"] == "请问您计划什么时候从郑州出发？"


def test_plan_destination_recommendation_when_destination_missing():
    """目的地不明确时生成目的地推荐任务"""
    planner = TaskPlanner()
    intent = {
        "intent": "travel_plan",
        "entities": {
            "origin": "郑州",
            "destination": None,
            "duration": 3,
            "budget": 3000,
            "preferences": ["海边", "放松"],
        },
        "missing_slots": ["destination"],
        "followup_question": "请问您计划去哪个城市旅行？",
    }

    tasks = planner.plan(intent, {"query": "预算3000，想去海边放松三天"})

    assert len(tasks) == 1
    assert tasks[0]["task_type"] == "recommend_destination"
    assert tasks[0]["params"]["budget"] == 3000
    assert "海边" in tasks[0]["params"]["preferences"]


def test_plan_full_travel_tasks():
    """目的地明确且信息完整时生成完整旅行规划任务"""
    planner = TaskPlanner()
    intent = {
        "intent": "travel_plan",
        "entities": {
            "origin": "郑州",
            "destination": "杭州",
            "departure_date": "2026-06-10",
            "return_date": None,
            "duration": 3,
            "budget": 3000,
            "travelers": 2,
            "preferences": ["干净", "地铁附近"],
        },
        "missing_slots": [],
    }

    tasks = planner.plan(intent, {"query": "我要从郑州去杭州玩三天"})
    task_types = [task["task_type"] for task in tasks]
    tools = [task.get("tool") for task in tasks]

    assert len(tasks) == 5
    assert task_types[-1] == "generate_itinerary"
    assert "search_flights" in tools
    assert "search_hotels" in tools
    assert "search_attractions" in tools
    assert "retrieve_guide" in tools
    assert "generate_itinerary" in tools


def test_plan_policy_query():
    """政策查询生成政策检索任务"""
    planner = TaskPlanner()
    intent = {
        "intent": "policy_query",
        "entities": {},
        "missing_slots": [],
    }

    tasks = planner.plan(intent, {"query": "机票能退吗"})

    assert len(tasks) == 1
    assert tasks[0]["task_type"] == "tool_call"
    assert tasks[0]["tool"] == "retrieve_policy"
    assert tasks[0]["params"]["query"] == "机票能退吗"


def test_plan_weather_query():
    """天气查询生成天气工具任务"""
    planner = TaskPlanner()
    intent = {
        "intent": "weather_query",
        "entities": {"destination": "杭州", "duration": None},
        "missing_slots": [],
    }

    tasks = planner.plan(intent, {"query": "杭州明天天气怎么样？"})

    assert len(tasks) == 1
    assert tasks[0]["task_type"] == "tool_call"
    assert tasks[0]["tool"] == "get_weather_forecast"
    assert tasks[0]["params"]["city"] == "杭州"
    assert tasks[0]["params"]["days"] == 3



def test_plan_itinerary_revision():
    """行程调整生成内部修订任务"""
    planner = TaskPlanner()
    intent = {
        "intent": "itinerary_revision",
        "entities": {},
        "missing_slots": [],
    }

    tasks = planner.plan(intent, {"query": "把西湖安排到第一天"})

    assert len(tasks) == 1
    assert tasks[0]["task_type"] == "revise_itinerary"
    assert tasks[0]["tool"] is None
    assert tasks[0]["params"]["query"] == "把西湖安排到第一天"



def test_plan_dynamic_knowledge_query():
    """动态知识追问生成内部动态RAG检索任务"""
    planner = TaskPlanner()
    intent = {
        "intent": "dynamic_knowledge_query",
        "entities": {},
        "missing_slots": [],
    }

    tasks = planner.plan(intent, {"query": "西湖在哪里？"})

    assert len(tasks) == 1
    assert tasks[0]["task_type"] == "dynamic_rag_query"
    assert tasks[0]["tool"] is None
    assert tasks[0]["params"]["query"] == "西湖在哪里？"



def test_plan_hotel_search():
    """酒店查询生成酒店搜索任务"""
    planner = TaskPlanner()
    intent = {
        "intent": "hotel_search",
        "entities": {"destination": "杭州", "budget": 1000, "preferences": ["地铁附近"]},
        "missing_slots": [],
    }

    tasks = planner.plan(intent, {"query": "帮我找杭州酒店"})

    assert len(tasks) == 1
    assert tasks[0]["tool"] == "search_hotels"
    assert tasks[0]["params"]["location"] == "杭州"
    assert tasks[0]["params"]["budget"] == 1000
