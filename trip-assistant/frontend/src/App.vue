<script setup>
import { computed, nextTick, onMounted, ref } from "vue";
import { Plus, RefreshCw } from "@lucide/vue";
import { fetchExternalStatus, sendChatMessage } from "./api";
import ChatPanel from "./components/ChatPanel.vue";
import StatusPanel from "./components/StatusPanel.vue";

const SESSION_STORAGE_KEY = "travelMindSessionId";

const input = ref("");
const sessionId = ref(window.localStorage.getItem(SESSION_STORAGE_KEY) || "");
const loading = ref(false);
const statusLoading = ref(false);
const statusError = ref("");
const externalStatus = ref(null);
const chatPanel = ref(null);
const messages = ref([
  {
    id: crypto.randomUUID(),
    role: "assistant",
    content:
      "您好，我是 TravelMind。可以输入完整旅行需求，也可以用右侧快捷脚本演示多轮规划、路线优化和雨天调整。",
    artifacts: {},
  },
]);

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

function addMessage(role, content, artifacts = {}) {
  messages.value.push({
    id: crypto.randomUUID(),
    role,
    content,
    artifacts: artifacts || {},
  });
  nextTick(() => {
    chatPanel.value?.scrollToBottom();
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
    addMessage("assistant", data.response || "处理完成", data.artifacts || {});
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
      artifacts: {},
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
      <ChatPanel
        ref="chatPanel"
        :messages="messages"
        :loading="loading"
        :session-id="sessionId"
        :short-session-id="shortSessionId"
        @send="submitMessage"
      />

      <StatusPanel
        :external-status="externalStatus"
        :status-summary="statusSummary"
        :status-loading="statusLoading"
        :status-error="statusError"
        :session-mode-text="sessionModeText"
        :short-session-id="shortSessionId"
        :loading="loading"
        @run-quick="runQuickAction"
      />
    </section>
  </main>
</template>
