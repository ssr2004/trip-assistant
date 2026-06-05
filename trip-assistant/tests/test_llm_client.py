"""
LLM客户端模块测试
"""
import pytest

from app.config import Settings
from core.llm import LLMClient, LLMMessage, LLMRequest, LLMResponse
from core.llm.prompts import (
    INTENT_FALLBACK_SYSTEM_PROMPT,
    PLANNER_FALLBACK_SYSTEM_PROMPT,
    ITINERARY_GENERATION_SYSTEM_PROMPT,
    RESPONSE_POLISH_SYSTEM_PROMPT,
    PROMPT_REGISTRY,
    get_prompt_metadata,
)


class FakeMessage:
    """模拟OpenAI返回消息"""

    content = "这是LLM返回内容"


class FakeChoice:
    """模拟OpenAI返回选项"""

    message = FakeMessage()


class FakeUsage:
    """Mock OpenAI-compatible token usage."""

    prompt_tokens = 12
    completion_tokens = 8
    total_tokens = 20


class FakeCompletion:
    """模拟OpenAI完整响应"""

    choices = [FakeChoice()]
    usage = FakeUsage()


class FakeCompletions:
    """模拟chat.completions接口"""

    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeCompletion()


class FakeChat:
    """模拟chat对象"""

    def __init__(self):
        self.completions = FakeCompletions()


class FakeOpenAIClient:
    """模拟OpenAI客户端"""

    def __init__(self):
        self.chat = FakeChat()


class FakeFailingCompletions:
    """模拟失败的chat.completions接口"""

    async def create(self, **kwargs):
        raise RuntimeError("调用失败: sk-test-secret")


class FakeFailingChat:
    """模拟失败chat对象"""

    completions = FakeFailingCompletions()


class FakeFailingOpenAIClient:
    """模拟失败OpenAI客户端"""

    chat = FakeFailingChat()


class FakeRateLimitCompletions:
    """模拟限流错误"""

    async def create(self, **kwargs):
        raise RuntimeError("HTTP 429 rate limit exceeded")


class FakeRateLimitChat:
    """模拟限流chat对象"""

    completions = FakeRateLimitCompletions()


class FakeRateLimitOpenAIClient:
    """模拟限流OpenAI客户端"""

    chat = FakeRateLimitChat()


def test_llm_client_unavailable_without_api_key():
    """没有API Key时客户端不可用并返回降级结果"""
    settings = Settings(LLM_API_KEY="", LLM_BASE_URL="https://api.deepseek.com/v1")
    client = LLMClient(settings=settings)

    assert client.available is False


@pytest.mark.asyncio
async def test_llm_client_chat_fallback_without_api_key():
    """没有API Key时chat返回标准失败结构"""
    settings = Settings(LLM_API_KEY="", LLM_BASE_URL="https://api.deepseek.com/v1")
    client = LLMClient(settings=settings)

    response = await client.chat(LLMRequest(messages=[LLMMessage(role="user", content="你好")]))

    assert isinstance(response, LLMResponse)
    assert response.success is False
    assert response.content == ""
    assert "LLM_API_KEY未配置" in response.error
    assert response.metadata["fallback"] is True
    assert response.metadata["provider"] == "deepseek"
    assert response.metadata["execution_mode"] == "rule_fallback"
    assert response.metadata["error_type"] == "missing_api_key"


@pytest.mark.asyncio
async def test_llm_client_chat_with_mock_openai_client():
    """有API Key时可以通过OpenAI-compatible客户端发起请求并解析响应"""
    settings = Settings(
        LLM_API_KEY="sk-test-secret",
        LLM_BASE_URL="https://api.deepseek.com/v1",
        LLM_PROVIDER="deepseek",
        LLM_MODEL="deepseek-chat",
        LLM_TEMPERATURE=0.2,
    )
    fake_client = FakeOpenAIClient()
    client = LLMClient(settings=settings, openai_client=fake_client)

    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content="你是旅行助手"),
            LLMMessage(role="user", content="帮我规划杭州三天游"),
        ],
        response_format="json_object",
        metadata={**get_prompt_metadata("planner_fallback"), "fallback_for": "task_plan", "raw_user_input": "不应透传"},
    )
    response = await client.chat(request)

    assert client.available is True
    assert response.success is True
    assert response.content == "这是LLM返回内容"
    assert response.metadata["model"] == "deepseek-chat"
    assert response.metadata["response_format"] == "json_object"
    assert response.metadata["prompt_id"] == "planner_fallback"
    assert response.metadata["prompt_version"] == PROMPT_REGISTRY["planner_fallback"].version
    assert response.metadata["fallback_for"] == "task_plan"
    assert "raw_user_input" not in response.metadata
    assert response.metadata["execution_mode"] == "llm"
    assert isinstance(response.metadata["duration_ms"], int)
    assert response.metadata["duration_ms"] >= 0
    assert response.metadata["prompt_tokens"] == 12
    assert response.metadata["completion_tokens"] == 8
    assert response.metadata["total_tokens"] == 20

    kwargs = fake_client.chat.completions.kwargs
    assert kwargs["model"] == "deepseek-chat"
    assert kwargs["temperature"] == 0.2
    assert kwargs["messages"][0] == {"role": "system", "content": "你是旅行助手"}
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_llm_client_sanitizes_api_key_from_errors():
    """LLM调用失败时不暴露API Key"""
    settings = Settings(
        LLM_API_KEY="sk-test-secret",
        LLM_BASE_URL="https://api.deepseek.com/v1",
    )
    client = LLMClient(settings=settings, openai_client=FakeFailingOpenAIClient())

    response = await client.chat(LLMRequest(messages=[LLMMessage(role="user", content="你好")]))

    assert response.success is False
    assert "sk-test-secret" not in response.error
    assert "***" in response.error
    assert response.metadata["error_type"] == "provider_error"
    assert isinstance(response.metadata["duration_ms"], int)


@pytest.mark.asyncio
async def test_llm_client_classifies_rate_limit_errors():
    """LLM调用失败时提供稳定错误分类"""
    settings = Settings(
        LLM_API_KEY="sk-test-secret",
        LLM_BASE_URL="https://api.deepseek.com",
    )
    client = LLMClient(settings=settings, openai_client=FakeRateLimitOpenAIClient())

    response = await client.chat(LLMRequest(messages=[LLMMessage(role="user", content="你好")]))

    assert response.success is False
    assert response.metadata["error_type"] == "rate_limit"
    assert "sk-test-secret" not in response.error


def test_default_llm_settings_are_deepseek_compatible():
    """默认LLM配置面向DeepSeek OpenAI-compatible接口"""
    settings = Settings(_env_file=None)

    assert settings.LLM_PROVIDER == "deepseek"
    assert settings.LLM_MODEL == "deepseek-v4-flash"
    assert settings.LLM_BASE_URL == "https://api.deepseek.com"
    assert settings.LLM_PLANNER_ENABLED is False


def test_prompt_templates_are_ready_for_future_fallbacks():
    """Prompt模板为后续fallback场景预留"""
    assert "意图识别" in INTENT_FALLBACK_SYSTEM_PROMPT
    assert "任务规划" in PLANNER_FALLBACK_SYSTEM_PROMPT
    assert "旅行规划师" in ITINERARY_GENERATION_SYSTEM_PROMPT
    assert "最终回复" in RESPONSE_POLISH_SYSTEM_PROMPT


def test_prompt_registry_exposes_stable_versions():
    """Prompt注册表提供稳定版本元数据。"""
    for prompt_id in ["intent_fallback", "planner_fallback", "itinerary_generation", "json_repair"]:
        metadata = get_prompt_metadata(prompt_id)
        assert metadata["prompt_id"] == prompt_id
        assert metadata["prompt_version"]
        assert metadata["prompt_purpose"]
