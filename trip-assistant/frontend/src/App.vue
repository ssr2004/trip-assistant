<script setup>
import { computed, nextTick, onMounted, ref } from "vue";
import {
  CloudSun,
  MapPinned,
  MessageCircle,
  Navigation,
  Plane,
  Plus,
  RefreshCw,
  Route,
  Send,
  ShieldCheck,
  Sparkles,
  Umbrella,
} from "@lucide/vue";
import { fetchExternalStatus, sendChatMessage } from "./api";

const SESSION_STORAGE_KEY = "travelMindSessionId";

const input = ref("");
const sessionId = ref(window.localStorage.getItem(SESSION_STORAGE_KEY) || "");
const loading = ref(false);
const statusLoading = ref(false);
const statusError = ref("");
const externalStatus = ref(null);
const messages = ref([
  {
    id: crypto.randomUUID(),
    role: "assistant",
    content:
      "您好，我是 TravelMind。可以输入完整旅行需求，也可以用右侧快捷脚本演示多轮规划、路线优化和雨天调整。",
  },
]);
const messagesPanel = ref(null);

const quickActions = [
  {
    label: "完整规划",
    icon: Plane,
    prompt: "我要从郑州去杭州玩三天，预算3000，6月10日出发",
  },
  {
    label: "雨天调整",
    icon: Umbrella,
    prompt: "如果下雨怎么办？",
  },
  {
    label: "路线优化",
    icon: Route,
    prompt: "帮我按距离优化一下第二天行程",
  },
  {
    label: "景点追问",
    icon: MapPinned,
    prompt: "西湖在哪里？",
  },
];

const statusSummary = computed(() => {
  if (!externalStatus.value) {
    return {
      total: 0,
      real_api_count: 0,
      mock_fallback_count: 0,
      unavailable_count: 0,
      all_operational: false,
    };
  }
  return externalStatus.value.summary;
});

const shortSessionId = computed(() => {
  if (!sessionId.value) {
    return "未创建";
  }
  return `${sessionId.value.slice(0, 8)}...${sessionId.value.slice(-6)}`;
});

const sessionModeText = computed(() => (sessionId.value ? "会话保持中" : "首轮消息后自动创建"));

function modeLabel(mode) {
  const labels = {
    real_api: "真实 API",
    mock_fallback: "Mock 降级",
    unavailable: "不可用",
  };
  return labels[mode] || mode;
}

function capabilityLabel(capability) {
  const labels = {
    poi_search: "景点检索",
    route_distance: "路线距离",
    weather_forecast: "天气预报",
  };
  return labels[capability] || capability;
}

function addMessage(role, content) {
  messages.value.push({
    id: crypto.randomUUID(),
    role,
    content,
  });
  nextTick(() => {
    if (messagesPanel.value) {
      messagesPanel.value.scrollTop = messagesPanel.value.scrollHeight;
    }
  });
}

async function refreshExternalStatus() {
  statusLoading.value = true;
  statusError.value = "";
  try {
    externalStatus.value = await fetchExternalStatus();
  } catch (error) {
    statusError.value = error.message || "外部 API 状态获取失败";
  } finally {
    statusLoading.value = false;
  }
}

async function submitMessage(messageOverride = "") {
  const message = (messageOverride || input.value).trim();
  if (!message || loading.value) {
    return;
  }

  addMessage("user", message);
  input.value = "";
  loading.value = true;

  try {
    const data = await sendChatMessage(message, sessionId.value);
    if (data.session_id) {
      sessionId.value = data.session_id;
      window.localStorage.setItem(SESSION_STORAGE_KEY, data.session_id);
    }
    addMessage("assistant", data.response || "处理完成");
  } catch (error) {
    addMessage("assistant", error.message || "抱歉，处理请求时出现错误。");
  } finally {
    loading.value = false;
  }
}

function runQuickAction(prompt) {
  submitMessage(prompt);
}

function newSession() {
  window.localStorage.removeItem(SESSION_STORAGE_KEY);
  sessionId.value = "";
  messages.value = [
    {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "已开启新会话。请输入新的旅行需求，系统会在首轮请求后创建新的 session_id。",
    },
  ];
}

onMounted(() => {
  refreshExternalStatus();
});
</script>

<template>
  <main class="app-shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">TravelMind 演示控制台</p>
        <h1>多智能体旅行规划系统</h1>
      </div>
      <div class="topbar-actions">
        <button class="icon-button" type="button" title="刷新外部 API 状态" @click="refreshExternalStatus">
          <RefreshCw :size="18" />
        </button>
        <button class="primary-button" type="button" @click="newSession">
          <Plus :size="18" />
          新建会话
        </button>
      </div>
    </header>

    <section class="dashboard-grid">
      <section class="chat-workbench" aria-label="旅行规划对话">
        <div class="panel-header">
          <div>
            <p class="eyebrow">Agent Chat</p>
            <h2>旅行需求与多轮调整</h2>
          </div>
          <div class="session-chip" :title="sessionId || '等待后端创建 session_id'">
            <MessageCircle :size="16" />
            <span>{{ shortSessionId }}</span>
          </div>
        </div>

        <div ref="messagesPanel" class="messages-panel">
          <article v-for="message in messages" :key="message.id" class="message-row" :class="message.role">
            <div class="message-bubble">
              {{ message.content }}
            </div>
          </article>
          <article v-if="loading" class="message-row assistant">
            <div class="message-bubble pending">正在规划和调用工具...</div>
          </article>
        </div>

        <form class="composer" @submit.prevent="submitMessage()">
          <textarea
            v-model="input"
            rows="2"
            placeholder="输入旅行需求，例如：我要从郑州去杭州玩三天，预算3000，6月10日出发"
            @keydown.enter.exact.prevent="submitMessage()"
          />
          <button class="send-button" type="submit" :disabled="loading || !input.trim()">
            <Send :size="18" />
            发送
          </button>
        </form>
      </section>

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
              @click="runQuickAction(action.prompt)"
            >
              <component :is="action.icon" :size="18" />
              <span>{{ action.label }}</span>
            </button>
          </div>
          <p class="hint">
            建议顺序：完整规划 -> 雨天调整 -> 路线优化 -> 景点追问。
          </p>
        </section>

        <section class="status-section compact-note">
          <CloudSun :size="20" />
          <p>
            配置高德 Key 后，POI、路线和天气会显示为真实 API；测试环境会自动隔离真实 Key，保证结果稳定。
          </p>
        </section>
      </aside>
    </section>
  </main>
</template>
