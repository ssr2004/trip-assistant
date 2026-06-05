"""Quality gates for structured LLM outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from models.intent import TravelIntent
from models.itinerary import LLMItineraryPlan
from models.task import TaskPlan


FORBIDDEN_OPERATION_TERMS = ["支付", "下单", "取消订单", "出票", "预订成功", "已预订", "已付款"]
FORBIDDEN_ITINERARY_CLAIMS = ["已预订", "已出票", "已付款", "下单成功", "库存充足"]


@dataclass
class LLMQualityResult:
    """Structured quality audit result."""

    passed: bool
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def metadata(self, prefix: str = "llm_output") -> Dict[str, Any]:
        """Return trace-safe metadata."""
        return {
            f"{prefix}_quality_passed": self.passed,
            f"{prefix}_quality_issues": self.issues,
            f"{prefix}_quality_issue_count": len(self.issues),
            f"{prefix}_quality_metrics": self.metrics,
        }


def audit_intent_quality(intent: TravelIntent) -> LLMQualityResult:
    """Audit parsed intent quality after schema validation."""
    issues: List[str] = []
    entities = intent.entities
    preferences = entities.preferences or []
    if intent.intent == "travel_plan" and entities.destination is None and "destination" not in intent.missing_slots:
        issues.append("travel_plan_destination_missing_not_declared")
    if intent.intent in {"flight_search", "hotel_search", "attraction_search"} and intent.confidence < 0.35:
        issues.append("low_confidence_actionable_intent")
    if not isinstance(preferences, list):
        issues.append("preferences_not_list")
    if intent.followup_question and len(intent.followup_question) > 120:
        issues.append("followup_question_too_long")

    metrics = {
        "intent": intent.intent,
        "confidence": intent.confidence,
        "missing_slot_count": len(intent.missing_slots or []),
        "preference_count": len(preferences),
    }
    return LLMQualityResult(passed=not issues, issues=issues, metrics=metrics)


def audit_task_plan_quality(plan: TaskPlan) -> LLMQualityResult:
    """Audit LLM task plan quality after schema and safety validation."""
    issues: List[str] = []
    task_ids = set()
    tool_count = 0
    for task in plan.tasks:
        if task.task_id in task_ids:
            issues.append("duplicate_task_id")
        task_ids.add(task.task_id)
        if not task.name.strip():
            issues.append(f"empty_task_name:{task.task_id}")
        if not task.reason.strip():
            issues.append(f"empty_task_reason:{task.task_id}")
        if task.priority < 1:
            issues.append(f"invalid_priority:{task.task_id}")
        if task.tool:
            tool_count += 1
        text_blob = " ".join([task.name, task.reason, str(task.params)])
        if any(term in text_blob for term in FORBIDDEN_OPERATION_TERMS):
            issues.append(f"forbidden_operation_claim:{task.task_id}")

    if plan.intent == "travel_plan" and not plan.tasks and not plan.need_user_input:
        issues.append("empty_travel_plan_without_user_input")
    if plan.need_user_input and not any(task.task_type == "ask_user" for task in plan.tasks):
        issues.append("need_user_input_without_ask_user_task")

    metrics = {
        "intent": plan.intent,
        "task_count": len(plan.tasks),
        "tool_count": tool_count,
        "need_user_input": plan.need_user_input,
    }
    return LLMQualityResult(passed=not issues, issues=issues, metrics=metrics)


def audit_itinerary_quality(plan: LLMItineraryPlan, expected_duration: int) -> LLMQualityResult:
    """Audit LLM-generated itinerary quality after schema validation."""
    issues: List[str] = []
    days = plan.itinerary or []
    if len(days) != expected_duration:
        issues.append("duration_mismatch")
    for expected_day, day in enumerate(days, start=1):
        if day.day != expected_day:
            issues.append(f"non_contiguous_day:{day.day}")
        if not day.title.strip():
            issues.append(f"empty_day_title:{day.day}")
        if not day.activities:
            issues.append(f"empty_activities:{day.day}")
        text_blob = " ".join([day.title, " ".join(day.activities), day.notes or ""])
        if any(term in text_blob for term in FORBIDDEN_ITINERARY_CLAIMS):
            issues.append(f"forbidden_realtime_claim:{day.day}")
    if not plan.summary.strip():
        issues.append("empty_summary")

    metrics = {
        "expected_duration": expected_duration,
        "itinerary_day_count": len(days),
        "activity_count": sum(len(day.activities or []) for day in days),
        "has_budget_tips": bool(plan.budget_tips),
    }
    return LLMQualityResult(passed=not issues, issues=issues, metrics=metrics)
