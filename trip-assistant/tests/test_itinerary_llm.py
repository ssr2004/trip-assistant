"""
LLM行程生成测试
"""
import pytest

from core.llm import LLMResponse
from tools.itinerary import ItineraryTool


class FakeLLMClient:
    """可控LLM客户端"""

    def __init__(self, content: str = "", success: bool = True, available: bool = True):
        self.content = content
        self.success = success
        self.available = available
        self.calls = 0
        self.last_request = None

    async def chat(self, request):
        self.calls += 1
        self.last_request = request
        return LLMResponse(
            success=self.success,
            content=self.content,
            error=None if self.success else "mock itinerary failed",
            metadata={"mock": True},
        )


@pytest.mark.asyncio
async def test_itinerary_tool_defaults_to_template_generation():
    """行程工具默认使用模板生成"""
    llm_client = FakeLLMClient(available=True)
    tool = ItineraryTool(llm_client=llm_client)

    result = await tool.execute(origin="郑州", destination="杭州", duration=3, budget=3000)

    assert result["success"] is True
    assert result["data"]["generation_mode"] == "template"
    assert result["data"]["context_summary"]["flight_count"] == 0
    assert result["metadata"]["source"] == "template_itinerary_generator"
    assert len(result["data"]["itinerary"]) == 3
    assert llm_client.calls == 0


@pytest.mark.asyncio
async def test_itinerary_tool_falls_back_when_llm_unavailable():
    """启用LLM但客户端不可用时回退模板"""
    llm_client = FakeLLMClient(available=False)
    tool = ItineraryTool(llm_client=llm_client, llm_enabled=True)

    result = await tool.execute(origin="郑州", destination="杭州", duration=3, budget=3000)

    assert result["data"]["generation_mode"] == "template"
    assert len(result["data"]["itinerary"]) == 3
    assert llm_client.calls == 0


@pytest.mark.asyncio
async def test_itinerary_tool_uses_valid_llm_plan():
    """启用LLM且返回合法行程时使用LLM结果"""
    llm_content = """
    {
      "itinerary": [
        {
          "day": 1,
          "title": "抵达杭州与西湖慢游",
          "activities": ["抵达杭州", "入住地铁附近酒店", "西湖散步", "湖滨晚餐"],
          "notes": "第一天控制强度，适合抵达后慢慢适应。"
        },
        {
          "day": 2,
          "title": "灵隐寺与茶文化体验",
          "activities": ["灵隐寺", "龙井茶村", "杭帮菜晚餐"],
          "notes": "兼顾人文和慢节奏体验。"
        },
        {
          "day": 3,
          "title": "城市休闲与返程",
          "activities": ["河坊街", "购买伴手礼", "整理返程"],
          "notes": "最后一天预留返程缓冲。"
        }
      ],
      "summary": "已为您生成从郑州出发前往杭州的3天慢节奏行程。",
      "budget_tips": "预算建议选择中档酒店，并优先安排免费或低门票景点。"
    }
    """
    llm_client = FakeLLMClient(content=llm_content)
    tool = ItineraryTool(llm_client=llm_client, llm_enabled=True)

    result = await tool.execute(
        origin="郑州",
        destination="杭州",
        duration=3,
        budget=3000,
        travelers=2,
        preferences=["慢节奏", "地铁附近"],
    )

    assert result["success"] is True
    assert result["data"]["generation_mode"] == "llm"
    assert result["metadata"]["source"] == "llm_itinerary_generator"
    assert result["data"]["summary"] == "已为您生成从郑州出发前往杭州的3天慢节奏行程。"
    assert result["data"]["itinerary"][0]["title"] == "抵达杭州与西湖慢游"
    assert result["data"]["context_summary"]["hotel_count"] == 0
    assert "中档酒店" in result["data"]["budget_summary"]["budget_note"]
    assert llm_client.calls == 1
    assert llm_client.last_request.response_format == "json_object"


@pytest.mark.asyncio
async def test_itinerary_tool_supports_markdown_json_from_llm():
    """LLM返回Markdown JSON代码块时也可以解析"""
    llm_content = """```json
{
  "itinerary": [
    {"day": 1, "title": "厦门海边初体验", "activities": ["抵达厦门", "环岛路散步"], "notes": "轻松安排。"},
    {"day": 2, "title": "鼓浪屿慢游", "activities": ["鼓浪屿", "海边咖啡"], "notes": "注意预约船票。"}
  ],
  "summary": "已生成厦门2天海边放松行程。",
  "budget_tips": "优先选择公共交通和免费海边景点。"
}
```"""
    llm_client = FakeLLMClient(content=llm_content)
    tool = ItineraryTool(llm_client=llm_client, llm_enabled=True)

    result = await tool.execute(destination="厦门", duration=2, preferences=["海边", "放松"])

    assert result["data"]["generation_mode"] == "llm"
    assert len(result["data"]["itinerary"]) == 2
    assert result["data"]["itinerary"][1]["title"] == "鼓浪屿慢游"


@pytest.mark.asyncio
async def test_itinerary_tool_summarizes_injected_context():
    """行程工具可以汇总前置任务注入的上下文"""
    tool = ItineraryTool()
    context = {
        "flights": [{"flight_no": "CZ123"}, {"flight_no": "HU456"}],
        "hotels": [{"name": "西湖酒店"}],
        "attractions": [{"name": "西湖"}, {"name": "灵隐寺"}, {"name": "西溪湿地"}],
        "guide": {"answer": "杭州三天行程可围绕西湖和灵隐寺安排。"},
        "errors": {},
    }

    result = await tool.execute(destination="杭州", duration=3, context=context)

    context_summary = result["data"]["context_summary"]
    assert context_summary["flight_count"] == 2
    assert context_summary["hotel_count"] == 1
    assert context_summary["attraction_count"] == 3
    assert context_summary["has_guide"] is True


@pytest.mark.asyncio
async def test_itinerary_tool_passes_context_to_llm_prompt():
    """启用LLM时将前置任务上下文传入提示词"""
    llm_content = """
    {
      "itinerary": [
        {"day": 1, "title": "参考候选结果游杭州", "activities": ["西湖"], "notes": "参考候选景点安排。"}
      ],
      "summary": "参考前置工具结果生成的行程。",
      "budget_tips": "参考候选酒店控制预算。"
    }
    """
    llm_client = FakeLLMClient(content=llm_content)
    tool = ItineraryTool(llm_client=llm_client, llm_enabled=True)
    context = {
        "flights": [{"flight_no": "CZ123"}],
        "hotels": [{"name": "西湖酒店"}],
        "attractions": [{"name": "西湖"}],
        "guide": {"answer": "杭州攻略"},
    }

    result = await tool.execute(destination="杭州", duration=1, context=context)

    assert result["data"]["generation_mode"] == "llm"
    assert result["data"]["context_summary"]["flight_count"] == 1
    assert "CZ123" in llm_client.last_request.messages[1].content
    assert "西湖酒店" in llm_client.last_request.messages[1].content


@pytest.mark.asyncio
async def test_itinerary_tool_falls_back_when_llm_returns_invalid_json():
    """LLM返回非法JSON时回退模板"""
    llm_client = FakeLLMClient(content="不是JSON")
    tool = ItineraryTool(llm_client=llm_client, llm_enabled=True)

    result = await tool.execute(destination="杭州", duration=3)

    assert result["data"]["generation_mode"] == "template"
    assert result["data"]["itinerary"][0]["title"] == "抵达杭州与西湖初体验"
    assert llm_client.calls == 1


@pytest.mark.asyncio
async def test_itinerary_tool_falls_back_when_llm_duration_mismatch():
    """LLM返回天数不匹配时回退模板"""
    llm_content = """
    {
      "itinerary": [
        {"day": 1, "title": "只有一天", "activities": ["散步"], "notes": "天数不匹配。"}
      ],
      "summary": "错误行程",
      "budget_tips": "无"
    }
    """
    llm_client = FakeLLMClient(content=llm_content)
    tool = ItineraryTool(llm_client=llm_client, llm_enabled=True)

    result = await tool.execute(destination="杭州", duration=3)

    assert result["data"]["generation_mode"] == "template"
    assert len(result["data"]["itinerary"]) == 3


@pytest.mark.asyncio
async def test_itinerary_tool_falls_back_when_llm_schema_invalid():
    """LLM返回结构不合法时回退模板"""
    llm_client = FakeLLMClient(content='{"itinerary": "not-a-list", "summary": "bad"}')
    tool = ItineraryTool(llm_client=llm_client, llm_enabled=True)

    result = await tool.execute(destination="杭州", duration=3)

    assert result["data"]["generation_mode"] == "template"
    assert len(result["data"]["itinerary"]) == 3
