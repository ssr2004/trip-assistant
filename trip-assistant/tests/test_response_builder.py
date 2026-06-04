"""
Agent响应生成模块测试
"""
from core.response_builder import ResponseBuilder


def test_build_followup_response():
    """缺少槽位时输出追问而不是内部任务结构"""
    builder = ResponseBuilder()
    response = builder.build(
        intent={"intent": "travel_plan"},
        task_results=[
            {
                "task": {"task_type": "ask_user", "name": "补充缺失信息"},
                "success": True,
                "result": {
                    "missing_slots": ["departure_date"],
                    "question": "请问您计划什么时候从郑州出发？",
                },
                "error": None,
            }
        ],
    )

    assert "请问您计划什么时候从郑州出发" in response
    assert "出发日期" in response
    assert "task_type" not in response


def test_build_destination_recommendation_response():
    """目的地不明确时输出候选目的地推荐"""
    builder = ResponseBuilder()
    response = builder.build(
        intent={"intent": "travel_plan"},
        task_results=[
            {
                "task": {"task_type": "recommend_destination", "name": "推荐旅行目的地"},
                "success": True,
                "result": {
                    "budget": 3000,
                    "preferences": ["自然风光"],
                    "candidates": [
                        {"city": "三亚", "reason": "适合看海和放松。"},
                        {"city": "厦门", "reason": "适合海边休闲。"},
                    ],
                    "message": "根据您的预算、时间和偏好，我先为您推荐以下候选目的地：",
                },
                "error": None,
            }
        ],
    )

    assert "三亚" in response
    assert "厦门" in response
    assert "预算约3000元" in response
    assert "继续为您规划交通、酒店和每日行程" in response


def test_build_full_travel_plan_response():
    """完整旅行规划输出结构化中文方案"""
    builder = ResponseBuilder()
    response = builder.build(
        intent={
            "intent": "travel_plan",
            "entities": {
                "origin": "郑州",
                "destination": "杭州",
                "departure_date": "2026-06-10",
                "duration": 3,
                "budget": 3000,
                "travelers": 2,
                "preferences": ["慢节奏"],
            },
        },
        task_results=[
            {
                "task": {"task_type": "tool_call", "tool": "search_flights", "name": "搜索航班"},
                "success": True,
                "result": [
                    {
                        "flight_no": "MU1234",
                        "airline": "东方航空",
                        "departure_airport": "郑州新郑国际机场",
                        "arrival_airport": "杭州萧山国际机场",
                        "departure_time": "2026-06-10 08:00",
                        "arrival_time": "2026-06-10 10:30",
                        "price": 680,
                        "cabin_class": "经济舱",
                    }
                ],
                "error": None,
            },
            {
                "task": {"task_type": "tool_call", "tool": "search_hotels", "name": "搜索酒店"},
                "success": True,
                "result": [
                    {
                        "name": "杭州西湖国宾馆",
                        "address": "杭州市西湖区杨公堤18号",
                        "price_per_night": 1200,
                        "rating": 4.8,
                        "amenities": ["免费WiFi", "游泳池"],
                    }
                ],
                "error": None,
            },
            {
                "task": {"task_type": "tool_call", "tool": "search_attractions", "name": "搜索景点"},
                "success": True,
                "result": [
                    {
                        "name": "西湖",
                        "category": "自然风光",
                        "rating": 4.9,
                        "ticket_price": "免费",
                        "description": "西湖是中国著名的风景名胜区。",
                    }
                ],
                "error": None,
            },
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
                        "travelers": 2,
                        "preferences": ["慢节奏"],
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
                            "estimated_food": 1080,
                            "estimated_local_transport": 480,
                            "estimated_attractions": 720,
                            "estimated_total_without_flight": 3480,
                            "budget_note": "预算可能偏紧。",
                        },
                    },
                    "error": None,
                    "metadata": {"tool": "generate_itinerary"},
                },
                "error": None,
            },
        ],
    )

    assert "已为您规划郑州到杭州的3天旅行方案" in response
    assert "一、出行概览" in response
    assert "二、航班推荐" in response
    assert "三、酒店推荐" in response
    assert "四、景点推荐" in response
    assert "五、每日行程" in response
    assert "六、预算估算" in response
    assert "东方航空" in response
    assert "杭州西湖国宾馆" in response
    assert "Day 1" in response
    assert "{'" not in response
    assert "[" not in response


def test_build_policy_response():
    """政策查询输出答案和来源"""
    builder = ResponseBuilder()
    response = builder.build(
        intent={"intent": "policy_query"},
        task_results=[
            {
                "task": {"task_type": "tool_call", "tool": "retrieve_policy", "name": "检索政策文档"},
                "success": True,
                "result": {
                    "success": True,
                    "data": {
                        "query": "机票能退吗",
                        "answer": "根据本地政策文档，机票可按规则退改。",
                        "sources": [{"source": "rag/documents/policies/flight_policy.md"}],
                    },
                    "error": None,
                    "metadata": {"tool": "retrieve_policy"},
                },
                "error": None,
            }
        ],
    )

    assert "关于您的政策问题" in response
    assert "机票可按规则退改" in response
    assert "资料来源" in response
    assert "flight_policy.md" in response


def test_build_itinerary_revision_response():
    """行程调整输出修订后的每日安排"""
    builder = ResponseBuilder()
    response = builder.build(
        intent={"intent": "itinerary_revision"},
        task_results=[
            {
                "task": {"task_type": "revise_itinerary", "name": "修订旅行行程"},
                "success": True,
                "result": {
                    "success": True,
                    "summary": "已将西湖安排到第1天。",
                    "itinerary": [
                        {"day": 1, "title": "抵达杭州", "activities": ["抵达杭州", "西湖"], "notes": "轻松游览。"}
                    ],
                    "sources": ["上一轮旅行方案", "会话动态景点数据"],
                },
                "error": None,
            }
        ],
    )

    assert "已根据您的要求调整行程" in response
    assert "调整后的每日行程" in response
    assert "Day 1" in response
    assert "西湖" in response
    assert "资料依据" in response



def test_build_dynamic_knowledge_response():
    """动态外部知识追问输出答案和来源"""
    builder = ResponseBuilder()
    response = builder.build(
        intent={"intent": "dynamic_knowledge_query"},
        task_results=[
            {
                "task": {"task_type": "dynamic_rag_query", "name": "检索动态外部知识"},
                "success": True,
                "result": {
                    "query": "西湖在哪里？",
                    "answer": "根据刚才推荐的外部景点数据，我查到：\n\n西湖：\n- 地址：杭州市西湖区\n- 坐标：120.1551,30.2741",
                    "sources": [
                        {
                            "title": "西湖",
                            "source": "api/amap/attraction/mock-西湖",
                            "type": "attraction",
                        }
                    ],
                },
                "error": None,
            }
        ],
    )

    assert "杭州市西湖区" in response
    assert "120.1551,30.2741" in response
    assert "资料来源" in response
    assert "api/amap/attraction/mock-西湖" in response



def test_build_single_attraction_response_shows_external_sources():
    """单独景点查询展示外部POI来源"""
    builder = ResponseBuilder()
    response = builder.build(
        intent={"intent": "attraction_search"},
        task_results=[
            {
                "task": {"task_type": "tool_call", "tool": "search_attractions", "name": "搜索景点"},
                "success": True,
                "result": {
                    "success": True,
                    "data": {
                        "attractions": [
                            {
                                "name": "西湖",
                                "category": "风景名胜",
                                "rating": 4.8,
                                "ticket_price": "待定",
                                "description": "来自高德POI：杭州市西湖区。",
                            }
                        ],
                        "rag_documents": [
                            {
                                "title": "西湖",
                                "source": "api/amap/attraction/mock-西湖",
                                "type": "attraction",
                                "metadata": {"provider": "amap"},
                            }
                        ],
                    },
                    "error": None,
                    "metadata": {"tool": "search_attractions"},
                },
                "error": None,
            }
        ],
    )

    assert "景点推荐" in response
    assert "西湖" in response
    assert "资料来源" in response
    assert "api/amap/attraction/mock-西湖" in response



def test_build_single_flight_response_cleans_none_time():
    """单独航班查询不暴露None时间"""
    builder = ResponseBuilder()
    response = builder.build(
        intent={"intent": "flight_search"},
        task_results=[
            {
                "task": {"task_type": "tool_call", "tool": "search_flights", "name": "搜索航班"},
                "success": True,
                "result": {
                    "success": True,
                    "data": {
                        "flights": [
                            {
                                "flight_no": "MU1234",
                                "airline": "东方航空",
                                "departure_airport": "北京",
                                "arrival_airport": "上海",
                                "departure_time": "None 08:00",
                                "arrival_time": "None 10:30",
                                "price": 680,
                                "cabin_class": "经济舱",
                            }
                        ]
                    },
                    "error": None,
                    "metadata": {"tool": "search_flights"},
                },
                "error": None,
            }
        ],
    )

    assert "航班推荐" in response
    assert "东方航空" in response
    assert "None" not in response
