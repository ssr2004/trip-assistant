"""Typed frontend artifact contracts shared by the agent and API layer."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FlexibleArtifactModel(BaseModel):
    """Base model for artifact payloads that may carry provider-specific fields."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ItineraryDayArtifact(FlexibleArtifactModel):
    day: int | str | None = None
    title: str | None = None
    activities: list[str] = Field(default_factory=list)
    notes: str | None = None


class ItineraryArtifact(FlexibleArtifactModel):
    title: str | None = None
    origin: str | None = None
    destination: str | None = None
    duration: int | str | None = None
    budget: int | float | str | None = None
    summary: str | None = None
    days: list[ItineraryDayArtifact] = Field(default_factory=list)
    budget_summary: dict[str, Any] = Field(default_factory=dict)


class WeatherForecastArtifact(FlexibleArtifactModel):
    date: str | None = None
    weather: str | None = None
    temperature: str | int | float | None = None
    suitable_for_outdoor: bool | None = None


class WeatherArtifact(FlexibleArtifactModel):
    city: str | None = None
    forecasts: list[WeatherForecastArtifact] = Field(default_factory=list)
    travel_advice: list[str] = Field(default_factory=list)


class WeatherAdjustmentDayArtifact(FlexibleArtifactModel):
    day: int | str | None = None
    date: str | None = None
    weather: str | None = None
    temperature: str | int | float | None = None
    advice: str | None = None


class WeatherAdjustmentArtifact(FlexibleArtifactModel):
    city: str | None = None
    adjusted_days: list[WeatherAdjustmentDayArtifact] = Field(default_factory=list)
    forecasts: list[WeatherForecastArtifact] = Field(default_factory=list)


class RouteSegmentArtifact(FlexibleArtifactModel):
    from_place: str | None = Field(default=None, alias="from")
    to: str | None = None
    distance: int | float | None = None
    duration: int | float | None = None


class RouteArtifact(FlexibleArtifactModel):
    day: int | str | None = None
    ordered_places: list[str | dict[str, Any]] = Field(default_factory=list)
    segments: list[RouteSegmentArtifact] = Field(default_factory=list)
    total_distance: int | float = 0
    total_duration: int | float = 0
    mode: str | None = None


class AttractionItemArtifact(FlexibleArtifactModel):
    id: str | int | None = None
    name: str | None = None
    category: str | None = None
    rating: str | int | float | None = None
    address: str | None = None
    location: str | None = None


class ArtifactSource(FlexibleArtifactModel):
    title: str | None = None
    content: str | None = None
    source: str | None = None
    score: int | float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttractionsArtifact(FlexibleArtifactModel):
    location: str | None = None
    items: list[AttractionItemArtifact] = Field(default_factory=list)
    sources: list[ArtifactSource] = Field(default_factory=list)


class ChatArtifacts(BaseModel):
    """Structured artifacts returned by /api/chat for rich frontend rendering."""

    itinerary: ItineraryArtifact | None = None
    weather: WeatherArtifact | None = None
    weather_adjustment: WeatherAdjustmentArtifact | None = None
    route: RouteArtifact | None = None
    attractions: AttractionsArtifact | None = None

    def to_frontend_dict(self) -> dict[str, Any]:
        """Dump only populated artifact groups so the Vue contract stays stable."""

        return self.model_dump(by_alias=True, exclude_none=True)


def normalize_chat_artifacts(value: Any) -> dict[str, Any]:
    """Validate artifact payloads and return the JSON shape consumed by the frontend."""

    if isinstance(value, ChatArtifacts):
        return value.to_frontend_dict()
    return ChatArtifacts.model_validate(value or {}).to_frontend_dict()
