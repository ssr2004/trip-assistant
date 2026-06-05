"""记忆数据模型。

定义用户长期偏好、规划画像和偏好使用证据等结构。
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class UserPreference(BaseModel):
    """用户旅行偏好"""

    travel_styles: List[str] = Field(default_factory=list, description="旅行节奏和风格偏好")
    hotel_preferences: List[str] = Field(default_factory=list, description="酒店和住宿偏好")
    transport_preferences: List[str] = Field(default_factory=list, description="交通偏好")
    attraction_preferences: List[str] = Field(default_factory=list, description="景点和玩法偏好")
    food_preferences: List[str] = Field(default_factory=list, description="饮食偏好")
    budget_preference: Optional[str] = Field(None, description="预算风格偏好")
    dietary_restrictions: List[str] = Field(default_factory=list, description="饮食禁忌或排除项")
    excluded_preferences: List[str] = Field(default_factory=list, description="用户明确排除的偏好")
    raw_preferences: List[str] = Field(default_factory=list, description="原始抽取偏好")
    updated_at: Optional[str] = Field(None, description="更新时间")
    preference_evidence: Dict[str, List[str]] = Field(default_factory=dict, description="偏好标签对应的命中文本证据")

    def has_preferences(self) -> bool:
        """判断是否抽取到了有效偏好"""
        return any([
            self.travel_styles,
            self.hotel_preferences,
            self.transport_preferences,
            self.attraction_preferences,
            self.food_preferences,
            self.budget_preference,
            self.dietary_restrictions,
            self.excluded_preferences,
            self.raw_preferences,
        ])


class PreferenceEvidence(BaseModel):
    """单条偏好证据。"""

    field: str = Field(..., description="偏好字段")
    value: str = Field(..., description="偏好值")
    evidence: List[str] = Field(default_factory=list, description="命中的文本证据")


class PlanningPreferenceProfile(BaseModel):
    """规划可直接消费的用户偏好画像。"""

    global_preferences: List[str] = Field(default_factory=list, description="跨工具通用偏好")
    travel_styles: List[str] = Field(default_factory=list, description="旅行风格偏好")
    budget_preference: Optional[str] = Field(None, description="预算偏好")
    hotel_preferences: List[str] = Field(default_factory=list, description="住宿偏好")
    transport_preferences: List[str] = Field(default_factory=list, description="交通偏好")
    attraction_preferences: List[str] = Field(default_factory=list, description="景点偏好")
    food_preferences: List[str] = Field(default_factory=list, description="饮食偏好")
    itinerary_constraints: List[str] = Field(default_factory=list, description="行程生成约束")
    excluded_preferences: List[str] = Field(default_factory=list, description="排除或禁忌偏好")
    tool_preferences: Dict[str, List[str]] = Field(default_factory=dict, description="按工具分流后的偏好")
    used_preference_count: int = Field(0, description="可用于规划的偏好数量")
    evidence: List[PreferenceEvidence] = Field(default_factory=list, description="偏好证据列表")
    conflicts: List[str] = Field(default_factory=list, description="潜在冲突偏好")
