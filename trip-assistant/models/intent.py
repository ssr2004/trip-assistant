"""
意图和实体数据模型
使用Pydantic定义结构化数据
"""
from typing import Literal, Optional, List
from pydantic import BaseModel, Field


IntentType = Literal[
    "travel_plan",
    "flight_search",
    "train_search",
    "hotel_search",
    "attraction_search",
    "policy_query",
    "guide_query",
    "dynamic_knowledge_query",
    "itinerary_revision",
    "weather_query",
    "general_chat",
]


class TravelEntities(BaseModel):
    """旅行实体"""

    origin: Optional[str] = Field(None, description="出发地")
    destination: Optional[str] = Field(None, description="目的地")
    departure_date: Optional[str] = Field(None, description="出发日期")
    return_date: Optional[str] = Field(None, description="返回日期")
    duration: Optional[int] = Field(None, description="旅行天数")
    budget: Optional[float] = Field(None, description="预算")
    travelers: Optional[int] = Field(None, description="出行人数")
    preferences: List[str] = Field(default_factory=list, description="偏好列表")


class TravelIntent(BaseModel):
    """旅行意图"""

    intent: IntentType = Field(..., description="意图类型")
    entities: TravelEntities = Field(default_factory=TravelEntities, description="实体信息")
    confidence: float = Field(0.0, description="置信度", ge=0, le=1)
    missing_slots: List[str] = Field(default_factory=list, description="缺失的关键信息")
    followup_question: Optional[str] = Field(None, description="针对缺失信息的追问")


class FlightInfo(BaseModel):
    """航班信息"""
    flight_no: str = Field(..., description="航班号")
    departure_airport: str = Field(..., description="出发机场")
    arrival_airport: str = Field(..., description="到达机场")
    departure_time: str = Field(..., description="出发时间")
    arrival_time: str = Field(..., description="到达时间")
    price: float = Field(..., description="价格")
    airline: str = Field(..., description="航空公司")


class HotelInfo(BaseModel):
    """酒店信息"""
    id: int = Field(..., description="酒店ID")
    name: str = Field(..., description="酒店名称")
    location: str = Field(..., description="位置")
    price_per_night: float = Field(..., description="每晚价格")
    rating: Optional[float] = Field(None, description="评分")
    amenities: List[str] = Field(default_factory=list, description="设施")


class AttractionInfo(BaseModel):
    """景点信息"""
    id: int = Field(..., description="景点ID")
    name: str = Field(..., description="景点名称")
    location: str = Field(..., description="位置")
    category: str = Field(..., description="类别")
    description: str = Field(..., description="描述")
    rating: Optional[float] = Field(None, description="评分")


class TravelPlan(BaseModel):
    """旅行计划"""
    origin: str = Field(..., description="出发地")
    destination: str = Field(..., description="目的地")
    duration: int = Field(..., description="旅行天数")
    flights: List[FlightInfo] = Field(default_factory=list, description="航班信息")
    hotels: List[HotelInfo] = Field(default_factory=list, description="酒店信息")
    itinerary: List[dict] = Field(default_factory=list, description="每日行程")
    total_budget: Optional[float] = Field(None, description="总预算")
