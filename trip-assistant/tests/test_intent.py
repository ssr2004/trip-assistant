"""
旅行需求解析模块测试
"""
from core.intent import IntentParser


def test_parse_travel_plan_with_budget_and_duration():
    """解析完整旅行规划需求"""
    parser = IntentParser()

    result = parser.parse("我要从郑州去杭州玩三天，预算3000，帮我安排一下")

    assert result["intent"] == "travel_plan"
    assert result["entities"]["origin"] == "郑州"
    assert result["entities"]["destination"] == "杭州"
    assert result["entities"]["duration"] == 3
    assert result["entities"]["budget"] == 3000
    assert "departure_date" in result["missing_slots"]
    assert result["followup_question"] is not None


def test_parse_flight_search_route():
    """解析航班搜索需求"""
    parser = IntentParser()

    result = parser.parse("郑州到杭州的航班")

    assert result["intent"] == "flight_search"
    assert result["entities"]["origin"] == "郑州"
    assert result["entities"]["destination"] == "杭州"


def test_parse_hotel_search():
    """解析酒店搜索需求"""
    parser = IntentParser()

    result = parser.parse("帮我找杭州的酒店")

    assert result["intent"] == "hotel_search"
    assert result["entities"]["destination"] == "杭州"
    assert result["missing_slots"] == []


def test_parse_attraction_search():
    """解析景点查询需求"""
    parser = IntentParser()

    result = parser.parse("杭州有什么好玩的")

    assert result["intent"] == "attraction_search"
    assert result["entities"]["destination"] == "杭州"


def test_parse_policy_query():
    """解析政策查询需求"""
    parser = IntentParser()

    result = parser.parse("机票能退吗")

    assert result["intent"] == "policy_query"
    assert result["missing_slots"] == []


def test_parse_flight_search_with_date():
    """解析带日期的航班搜索需求"""
    parser = IntentParser()

    result = parser.parse("我想从北京飞上海，6月10日出发")

    assert result["intent"] == "flight_search"
    assert result["entities"]["origin"] == "北京"
    assert result["entities"]["destination"] == "上海"
    assert result["entities"]["departure_date"].endswith("-06-10")


def test_parse_duration_and_budget():
    """解析预算和天数表达"""
    parser = IntentParser()

    result = parser.parse("预算5000，去成都玩4天3晚")

    assert result["intent"] == "travel_plan"
    assert result["entities"]["destination"] == "成都"
    assert result["entities"]["duration"] == 4
    assert result["entities"]["budget"] == 5000
    assert "origin" in result["missing_slots"]


def test_parse_preferences_and_travelers():
    """解析人数和偏好"""
    parser = IntentParser()

    result = parser.parse("两个人从郑州去杭州玩三天，酒店要干净，最好地铁附近")

    assert result["entities"]["travelers"] == 2
    assert "干净" in result["entities"]["preferences"]
    assert "地铁附近" in result["entities"]["preferences"]
