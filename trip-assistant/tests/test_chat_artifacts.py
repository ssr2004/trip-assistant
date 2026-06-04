"""Tests for structured chat artifact contracts."""

import pytest
from pydantic import ValidationError

from core.artifacts import ChatArtifacts, normalize_chat_artifacts


def test_empty_chat_artifacts_dump_as_empty_frontend_payload():
    assert normalize_chat_artifacts({}) == {}
    assert ChatArtifacts().to_frontend_dict() == {}


def test_chat_artifacts_keep_frontend_field_names_and_route_aliases():
    payload = normalize_chat_artifacts(
        {
            "itinerary": {
                "destination": "杭州",
                "days": [{"day": 1, "title": "西湖初体验", "activities": ["西湖"]}],
            },
            "route": {
                "day": 2,
                "segments": [{"from": "西湖", "to": "灵隐寺", "distance": 4300}],
                "total_distance": 4300,
                "total_duration": 1800,
            },
        }
    )

    assert payload["itinerary"]["destination"] == "杭州"
    assert payload["itinerary"]["days"][0]["activities"] == ["西湖"]
    assert payload["route"]["segments"][0]["from"] == "西湖"
    assert payload["route"]["segments"][0]["to"] == "灵隐寺"


def test_chat_artifacts_reject_invalid_known_field_types():
    with pytest.raises(ValidationError):
        ChatArtifacts.model_validate({"weather": {"forecasts": "not-a-list"}})
