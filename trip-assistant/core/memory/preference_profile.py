"""Build planner-ready preference profiles from long-term memory."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from models.memory import PlanningPreferenceProfile, PreferenceEvidence


class PreferenceProfileBuilder:
    """Convert stored memory preferences into tool-scoped planning constraints."""

    PREFERENCE_FIELDS = [
        "travel_styles",
        "hotel_preferences",
        "transport_preferences",
        "attraction_preferences",
        "food_preferences",
        "dietary_restrictions",
        "excluded_preferences",
    ]

    def build(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Build a serializable planning profile."""
        preferences = preferences or {}
        travel_styles = self._as_list(preferences.get("travel_styles"))
        hotel_preferences = self._as_list(preferences.get("hotel_preferences"))
        transport_preferences = self._as_list(preferences.get("transport_preferences"))
        attraction_preferences = self._as_list(preferences.get("attraction_preferences"))
        food_preferences = self._as_list(preferences.get("food_preferences"))
        dietary_restrictions = self._as_list(preferences.get("dietary_restrictions"))
        excluded_preferences = self._dedupe([
            *dietary_restrictions,
            *self._as_list(preferences.get("excluded_preferences")),
        ])
        budget_preference = preferences.get("budget_preference")

        global_preferences = self._dedupe([
            *travel_styles,
            *hotel_preferences,
            *transport_preferences,
            *attraction_preferences,
            *food_preferences,
            *([budget_preference] if budget_preference else []),
            *excluded_preferences,
        ])
        itinerary_constraints = self._dedupe([
            *travel_styles,
            *attraction_preferences,
            *food_preferences,
            *transport_preferences,
            *hotel_preferences,
            *([budget_preference] if budget_preference else []),
            *excluded_preferences,
        ])

        tool_preferences = {
            "search_flights": self._dedupe([*transport_preferences, *travel_styles, *([budget_preference] if budget_preference else [])]),
            "search_hotels": self._dedupe([*travel_styles, *hotel_preferences, *([budget_preference] if budget_preference else [])]),
            "search_attractions": self._dedupe([*travel_styles, *attraction_preferences, *food_preferences, *excluded_preferences]),
            "retrieve_guide": self._dedupe([*global_preferences]),
            "generate_itinerary": itinerary_constraints,
            "recommend_destination": self._dedupe([*travel_styles, *attraction_preferences, *food_preferences, *([budget_preference] if budget_preference else [])]),
        }

        profile = PlanningPreferenceProfile(
            global_preferences=global_preferences,
            travel_styles=travel_styles,
            budget_preference=budget_preference,
            hotel_preferences=hotel_preferences,
            transport_preferences=transport_preferences,
            attraction_preferences=attraction_preferences,
            food_preferences=food_preferences,
            itinerary_constraints=itinerary_constraints,
            excluded_preferences=excluded_preferences,
            tool_preferences=tool_preferences,
            used_preference_count=len(global_preferences),
            evidence=self._build_evidence(preferences),
            conflicts=self._detect_conflicts(preferences),
        )
        return profile.model_dump()

    def _build_evidence(self, preferences: Dict[str, Any]) -> List[PreferenceEvidence]:
        """Build preference evidence entries without storing user raw messages."""
        evidence_map = preferences.get("preference_evidence") if isinstance(preferences.get("preference_evidence"), dict) else {}
        entries = []
        for field in [*self.PREFERENCE_FIELDS, "budget_preference"]:
            values = self._as_list(preferences.get(field))
            if field == "budget_preference" and preferences.get(field):
                values = [preferences.get(field)]
            evidence_values = self._as_list(evidence_map.get(field))
            for value in values:
                entries.append(PreferenceEvidence(
                    field=field,
                    value=str(value),
                    evidence=evidence_values if value in evidence_values else [str(value)],
                ))
        return entries

    def _detect_conflicts(self, preferences: Dict[str, Any]) -> List[str]:
        """Detect simple preference conflicts that should remain visible to planning."""
        conflicts = []
        budget = preferences.get("budget_preference")
        hotels = self._as_list(preferences.get("hotel_preferences"))
        transport = self._as_list(preferences.get("transport_preferences"))
        travel_styles = self._as_list(preferences.get("travel_styles"))

        if budget == "经济型" and "高星级酒店" in hotels:
            conflicts.append("经济型预算与高星级酒店偏好可能冲突")
        if "飞机优先" in transport and "高铁优先" in transport:
            conflicts.append("飞机优先与高铁优先同时存在")
        if "慢节奏" in travel_styles and "深度游" in travel_styles:
            conflicts.append("慢节奏与深度游需要在行程密度上权衡")
        return conflicts

    def _as_list(self, value: Any) -> List[str]:
        """Normalize a scalar or list into a list of non-empty strings."""
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        return [str(item) for item in values if item not in (None, "", [], {})]

    def _dedupe(self, values: List[Optional[str]]) -> List[str]:
        """Keep order while removing empty values and duplicates."""
        result = []
        for value in values:
            if value and value not in result:
                result.append(value)
        return result
