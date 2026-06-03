"""
记忆数据模型
定义用户长期偏好等记忆结构
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class UserPreference(BaseModel):
    """用户旅行偏好"""

    travel_styles: List[str] = Field(default_factory=list, description="旅行节奏和风格偏好")
    hotel_preferences: List[str] = Field(default_factory=list, description="酒店和住宿偏好")
    transport_preferences: List[str] = Field(default_factory=list, description="交通偏好")
    attraction_preferences: List[str] = Field(default_factory=list, description="景点和玩法偏好")
    food_preferences: List[str] = Field(default_factory=list, description="饮食偏好")
    budget_preference: Optional[str] = Field(None, description="预算风格偏好")
    raw_preferences: List[str] = Field(default_factory=list, description="原始抽取偏好")
    updated_at: Optional[str] = Field(None, description="更新时间")

    def has_preferences(self) -> bool:
        """判断是否抽取到了有效偏好"""
        return any([
            self.travel_styles,
            self.hotel_preferences,
            self.transport_preferences,
            self.attraction_preferences,
            self.food_preferences,
            self.budget_preference,
            self.raw_preferences,
        ])
