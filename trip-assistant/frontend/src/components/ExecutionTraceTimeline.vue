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
</script>

<template>
  <section v-if="trace?.steps?.length" class="trace-panel">
    <div class="trace-header">
      <div>
        <strong>Execution Trace</strong>
        <span>
          {{ trace.summary.task_count || 0 }} tasks /
          {{ trace.summary.source_count || 0 }} sources
        </span>
      </div>
      <ListTree :size="16" />
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
          <small v-if="step.tool || step.source_count">
            {{ step.tool || step.task_type }}
            <template v-if="step.source_count"> / {{ step.source_count }} sources</template>
          </small>
        </div>
      </li>
    </ol>
  </section>
</template>
