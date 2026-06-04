<script setup lang="ts">
import { CheckCircle2, CircleAlert, ListTree } from "@lucide/vue";
import type { ExecutionTrace, TraceStep } from "../types";

withDefaults(
  defineProps<{
    trace?: ExecutionTrace;
  }>(),
  {
    trace: () => ({ steps: [], summary: {} }),
  },
);

function stageLabel(stage: string): string {
  const labels: Record<string, string> = {
    intent: "Intent",
    context: "Context",
    planning: "Plan",
    tool: "Tool",
    task: "Task",
    rag: "RAG",
  };
  return labels[stage] || stage;
}

function stepKey(step: TraceStep, index: number): string {
  return `${step.stage}-${step.tool || step.label}-${index}`;
}

function modeLabel(mode?: string | null): string {
  const labels: Record<string, string> = {
    real_api: "Real API",
    mock_fallback: "Mock",
    local_data: "Local",
    internal_rule: "Rule",
    internal_revision: "Revision",
    rule_fallback: "Rule Fallback",
    dynamic_rag: "Dynamic RAG",
    template: "Template",
    llm: "LLM",
    tool: "Tool",
  };
  return mode ? labels[mode] || mode : "";
}

function formatDuration(duration?: number | null): string {
  if (duration === null || duration === undefined) {
    return "";
  }
  return `${Math.max(Math.round(duration), 0)}ms`;
}

function summaryNumber(value: unknown): number {
  return typeof value === "number" ? value : 0;
}
</script>

<template>
  <section v-if="trace?.steps?.length" class="trace-panel">
    <div class="trace-header">
      <div>
        <strong>Execution Trace</strong>
        <span>
          {{ trace.summary.task_count || 0 }} tasks /
          {{ trace.summary.total_duration_ms || 0 }}ms /
          {{ trace.summary.source_count || 0 }} sources
        </span>
      </div>
      <ListTree :size="16" />
    </div>

    <div class="runtime-summary" aria-label="Runtime Metrics">
      <span>Planner {{ trace.summary.planner_mode || "template" }}</span>
      <span>LLM {{ summaryNumber(trace.summary.llm_call_count) }} calls</span>
      <span>{{ summaryNumber(trace.summary.llm_duration_ms) }}ms LLM</span>
      <span>{{ summaryNumber(trace.summary.llm_total_tokens) }} tokens</span>
      <span>{{ summaryNumber(trace.summary.llm_repair_count) }} repairs</span>
      <span>{{ summaryNumber(trace.summary.real_api_count) }} real API</span>
      <span>{{ summaryNumber(trace.summary.mock_fallback_count) }} mock</span>
      <span>{{ summaryNumber(trace.summary.tool_total_duration_ms) }}ms tools</span>
    </div>

    <ol class="trace-list">
      <li
        v-for="(step, index) in trace.steps"
        :key="stepKey(step, index)"
        class="trace-step"
        :class="step.status"
      >
        <CheckCircle2 v-if="step.status === 'success'" :size="15" />
        <CircleAlert v-else :size="15" />
        <div>
          <b>{{ stageLabel(step.stage) }} · {{ step.label }}</b>
          <span v-if="step.detail">{{ step.detail }}</span>
          <span v-else-if="step.result_summary">{{ step.result_summary }}</span>
          <div class="trace-badges">
            <em v-if="step.duration_ms !== undefined && step.duration_ms !== null">
              {{ formatDuration(step.duration_ms) }}
            </em>
            <em v-if="step.execution_mode">{{ modeLabel(step.execution_mode) }}</em>
            <em v-if="step.error_type" class="error">{{ step.error_type }}</em>
          </div>
          <small v-if="step.tool || step.source_count">
            {{ step.tool || step.task_type }}
            <template v-if="step.source_count"> / {{ step.source_count }} sources</template>
          </small>
        </div>
      </li>
    </ol>
  </section>
</template>
