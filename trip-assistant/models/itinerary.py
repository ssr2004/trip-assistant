"""
行程生成数据模型
定义LLM行程生成结果的结构化校验模型
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ItineraryDay(BaseModel):
    """每日行程"""

    day: int = Field(..., description="第几天")
    title: str = Field(..., description="当天主题")
    activities: List[str] = Field(default_factory=list, description="当天活动安排")
    notes: str = Field("", description="当天说明和注意事项")


class LLMItineraryPlan(BaseModel):
    """LLM生成的行程计划"""

    itinerary: List[ItineraryDay] = Field(default_factory=list, description="每日行程列表")
    summary: str = Field("", description="行程摘要")
    budget_tips: Optional[str] = Field(None, description="预算建议")
