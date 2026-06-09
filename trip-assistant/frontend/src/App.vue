<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from "vue";
import { Plus, RefreshCw } from "@lucide/vue";
import { fetchExternalStatus, fetchLLMStatus, sendChatMessage } from "./api";
import ChatPanel from "./components/ChatPanel.vue";
import StatusPanel from "./components/StatusPanel.vue";
import type { ChatArtifacts, ChatMessage, ExecutionTrace, ExternalStatusResponse, LLMStatusResponse, MessageRole } from "./types";

const SESSION_STORAGE_KEY = "menglvSessionId";

const input = ref("");
window.localStorage.removeItem(SESSION_STORAGE_KEY);
const sessionId = ref("");
const loading = ref(false);
const statusLoading = ref(false);
const statusError = ref("");
const externalStatus = ref<ExternalStatusResponse | null>(null);
const llmStatus = ref<LLMStatusResponse | null>(null);
const chatPanel = ref<{ scrollToBottom: () => void } | null>(null);
const messages = ref<ChatMessage[]>([
  {
    id: crypto.randomUUID(),
    role: "assistant",
    content:
      "您好，我是梦旅。可以输入完整旅行需求，也可以用右侧快捷脚本演示多轮规划、路线优化和雨天调整。",
    artifacts: {},
    execution_trace: { steps: [], summary: {} },
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

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function addMessage(
  role: MessageRole,
  content: string,
  artifacts: ChatArtifacts = {},
  executionTrace: ExecutionTrace = { steps: [], summary: {} },
) {
  messages.value.push({
    id: crypto.randomUUID(),
    role,
    content,
    artifacts: artifacts || {},
    execution_trace: executionTrace || { steps: [], summary: {} },
  });
  nextTick(() => {
    chatPanel.value?.scrollToBottom();
  });
}

async function refreshExternalStatus() {
  statusLoading.value = true;
  statusError.value = "";
  try {
    const [externalData, llmData] = await Promise.all([
      fetchExternalStatus(),
      fetchLLMStatus(),
    ]);
    externalStatus.value = externalData;
    llmStatus.value = llmData;
  } catch (error) {
    statusError.value = error.message || "系统状态获取失败";
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
    }
    addMessage("assistant", data.response || "处理完成", data.artifacts || {}, data.execution_trace || { steps: [], summary: {} });
  } catch (error) {
    addMessage("assistant", error.message || "抱歉，处理请求时出现错误。");
  } finally {
    loading.value = false;
  }
}

function runQuickAction(prompt: string) {
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
      execution_trace: { steps: [], summary: {} },
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
        <h1>梦旅</h1>
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
        :llm-status="llmStatus"
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
