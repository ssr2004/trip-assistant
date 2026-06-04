<script setup lang="ts">
import type { Component } from "vue";
import {
  BrainCircuit,
  CloudSun,
  MapPinned,
  Navigation,
  Plane,
  Route,
  ShieldCheck,
  Sparkles,
  Umbrella,
} from "@lucide/vue";
import type { ExternalStatusResponse, ExternalStatusSummary, LLMStatusResponse } from "../types";

withDefaults(
  defineProps<{
    externalStatus: ExternalStatusResponse | null;
    llmStatus: LLMStatusResponse | null;
    statusSummary: ExternalStatusSummary;
    statusLoading?: boolean;
    statusError?: string;
    sessionModeText: string;
    shortSessionId: string;
    loading?: boolean;
  }>(),
  {
    externalStatus: null,
    llmStatus: null,
    statusLoading: false,
    statusError: "",
    loading: false,
  },
);

const emit = defineEmits<{
  "run-quick": [prompt: string];
}>();

interface QuickAction {
  label: string;
  icon: Component;
  prompt: string;
}

const quickActions: QuickAction[] = [
  {
    label: "1 完整规划",
    icon: Plane,
    prompt: "我要从郑州去杭州玩三天，预算3000，6月10日出发",
  },
  {
    label: "2 雨天调整",
    icon: Umbrella,
    prompt: "如果下雨怎么办？",
  },
  {
    label: "3 路线优化",
    icon: Route,
    prompt: "帮我按距离优化一下第二天行程",
  },
  {
    label: "4 景点追问",
    icon: MapPinned,
    prompt: "西湖在哪里？",
  },
];

function modeLabel(mode: string) {
  const labels: Record<string, string> = {
    real_api: "真实 API",
    mock_fallback: "Mock 降级",
    unavailable: "不可用",
    real_llm: "真实 LLM",
    rule_fallback: "规则降级",
  };
  return labels[mode] || mode;
}

function capabilityLabel(capability: string) {
  const labels: Record<string, string> = {
    poi_search: "景点检索",
    route_distance: "路线距离",
    weather_forecast: "天气预报",
  };
  return labels[capability] || capability;
}
</script>

<template>
  <aside class="inspector" aria-label="演示状态面板">
    <section class="status-section">
      <div class="panel-header compact">
        <div>
          <p class="eyebrow">Session</p>
          <h2>会话状态</h2>
        </div>
        <ShieldCheck :size="20" />
      </div>
      <dl class="session-details">
        <div>
          <dt>状态</dt>
          <dd>{{ sessionModeText }}</dd>
        </div>
        <div>
          <dt>当前 ID</dt>
          <dd>{{ shortSessionId }}</dd>
        </div>
      </dl>
    </section>

    <section class="status-section">
      <div class="panel-header compact">
        <div>
          <p class="eyebrow">LLM Runtime</p>
          <h2>模型能力状态</h2>
        </div>
        <BrainCircuit :size="20" />
      </div>

      <article v-if="llmStatus" class="service-item llm-service" :class="llmStatus.mode">
        <div>
          <strong>{{ modeLabel(llmStatus.mode) }}</strong>
          <span>{{ llmStatus.provider }} / {{ llmStatus.model }}</span>
          <span>{{ llmStatus.key_source || "无 Key，使用规则 fallback" }}</span>
        </div>
        <b>{{ llmStatus.openai_compatible ? "OpenAI-compatible" : "Custom" }}</b>
      </article>
      <p v-else-if="statusLoading" class="status-muted">正在读取 LLM 状态...</p>
      <p v-else class="status-muted">LLM 状态待刷新。</p>
    </section>

    <section class="status-section">
      <div class="panel-header compact">
        <div>
          <p class="eyebrow">External APIs</p>
          <h2>外部能力状态</h2>
        </div>
        <Navigation :size="20" />
      </div>

      <div class="summary-strip">
        <span>{{ statusSummary.real_api_count }} 真实</span>
        <span>{{ statusSummary.mock_fallback_count }} 降级</span>
        <span>{{ statusSummary.unavailable_count }} 不可用</span>
      </div>

      <p v-if="statusError" class="status-error">{{ statusError }}</p>
      <p v-else-if="statusLoading" class="status-muted">正在读取外部 API 状态...</p>

      <div v-if="externalStatus" class="service-list">
        <article
          v-for="service in externalStatus.services"
          :key="service.name"
          class="service-item"
          :class="service.mode"
        >
          <div>
            <strong>{{ capabilityLabel(service.capability) }}</strong>
            <span>{{ service.provider }} / {{ service.key_source || "无 Key" }}</span>
          </div>
          <b>{{ modeLabel(service.mode) }}</b>
        </article>
      </div>
    </section>

    <section class="status-section">
      <div class="panel-header compact">
        <div>
          <p class="eyebrow">Demo Scripts</p>
          <h2>快捷演示</h2>
        </div>
        <Sparkles :size="20" />
      </div>
      <div class="quick-grid">
        <button
          v-for="action in quickActions"
          :key="action.label"
          type="button"
          class="quick-action"
          :disabled="loading"
          @click="emit('run-quick', action.prompt)"
        >
          <component :is="action.icon" :size="18" />
          <span>{{ action.label }}</span>
        </button>
      </div>
      <p class="hint">
        建议顺序：先看外部能力状态，再依次执行 1 到 4，并在每轮回答下方查看 artifacts 和 trace。
      </p>
    </section>

    <section class="status-section compact-note">
      <CloudSun :size="20" />
      <p>
        配置高德 Key 后，POI、路线和天气会显示为真实 API；测试环境会自动隔离真实 Key，保证结果稳定。
      </p>
    </section>
  </aside>
</template>
