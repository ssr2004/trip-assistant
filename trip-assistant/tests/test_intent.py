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


def test_parse_weather_query():
    """解析天气查询需求"""
    parser = IntentParser()

    result = parser.parse("杭州明天天气怎么样？")

    assert result["intent"] == "weather_query"
    assert result["entities"]["destination"] == "杭州"
    assert result["missing_slots"] == []



def test_parse_rainy_itinerary_revision():
    """解析雨天行程调整需求"""
    parser = IntentParser()

    result = parser.parse("如果下雨怎么办？")

    assert result["intent"] == "itinerary_revision"
    assert result["missing_slots"] == []



def test_parse_itinerary_revision_move_attraction():
    """解析行程调整中的景点移动请求"""
    parser = IntentParser()

    result = parser.parse("把西湖安排到第一天")

    assert result["intent"] == "itinerary_revision"
    assert result["missing_slots"] == []



def test_parse_itinerary_revision_replace_attraction():
    """解析行程调整中的景点替换请求"""
    parser = IntentParser()

    result = parser.parse("不要去宋城，换一个自然风光景点")

    assert result["intent"] == "itinerary_revision"
    assert result["missing_slots"] == []



def test_parse_itinerary_revision_route_optimization():
    """解析行程调整中的路线优化请求"""
    parser = IntentParser()

    result = parser.parse("帮我按距离优化一下第二天行程")

    assert result["intent"] == "itinerary_revision"
    assert result["missing_slots"] == []



def test_parse_dynamic_knowledge_query_for_poi_detail():
    """解析外部动态知识景点详情追问"""
    parser = IntentParser()

    result = parser.parse("西湖在哪里？")

    assert result["intent"] == "dynamic_knowledge_query"
    assert result["missing_slots"] == []



def test_parse_dynamic_knowledge_query_for_previous_recommendations():
    """解析刚才推荐景点追问"""
    parser = IntentParser()

    result = parser.parse("刚才推荐的景点有哪些？")

    assert result["intent"] == "dynamic_knowledge_query"
    assert result["missing_slots"] == []



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


def test_parse_date_without_misreading_duration():
    """日期中的日不应被误识别为旅行天数"""
    parser = IntentParser()

    result = parser.parse("我要从郑州去杭州玩三天，预算3000，6月10日出发")

    assert result["entities"]["departure_date"].endswith("-06-10")
    assert result["entities"]["duration"] == 3


def test_parse_relative_date_route_and_region_destination():
    """解析明天从杭州去西藏玩5天这类组合句式"""
    parser = IntentParser()

    result = parser.parse("我要明天从杭州去西藏玩5天")

    assert result["intent"] == "travel_plan"
    assert result["entities"]["origin"] == "杭州"
    assert result["entities"]["destination"] == "西藏"
    assert result["entities"]["departure_date"] is not None
    assert result["entities"]["duration"] == 5
    assert result["missing_slots"] == []


def test_parse_hangzhou_to_yantai_route():
    """常见城市目的地应正确识别，避免把目的地错判为出发城市"""
    parser = IntentParser()

    result = parser.parse("我明天从杭州去烟台玩3天")

    assert result["intent"] == "travel_plan"
    assert result["entities"]["origin"] == "杭州"
    assert result["entities"]["destination"] == "烟台"
    assert result["entities"]["duration"] == 3
    assert result["missing_slots"] == []


def test_parse_taiyuan_to_zhengzhou_short_route():
    """短路线句也应抽取出发地和目的地，不能完全交给LLM兜底。"""
    parser = IntentParser()

    result = parser.parse("从太原去郑州")

    assert result["intent"] == "travel_plan"
    assert result["entities"]["origin"] == "太原"
    assert result["entities"]["destination"] == "郑州"
    assert "departure_date" in result["missing_slots"]
    assert "duration" in result["missing_slots"]
    assert "origin" not in result["missing_slots"]
    assert "destination" not in result["missing_slots"]


def test_parse_route_correction_with_region_origin():
    """解析用户纠正方向：我是从西藏去杭州"""
    parser = IntentParser()

    result = parser.parse("我是从西藏去杭州啊")

    assert result["intent"] == "travel_plan"
    assert result["entities"]["origin"] == "西藏"
    assert result["entities"]["destination"] == "杭州"
