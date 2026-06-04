<script setup lang="ts">
import { nextTick, ref } from "vue";
import { MessageCircle, Send } from "@lucide/vue";
import ArtifactCards from "./ArtifactCards.vue";
import ExecutionTraceTimeline from "./ExecutionTraceTimeline.vue";
import type { ChatMessage } from "../types";

withDefaults(
  defineProps<{
    messages: ChatMessage[];
    loading?: boolean;
    sessionId?: string;
    shortSessionId: string;
  }>(),
  {
    loading: false,
    sessionId: "",
  },
);

const emit = defineEmits<{
  send: [message: string];
}>();
const input = ref("");
const messagesPanel = ref<HTMLDivElement | null>(null);

function submitMessage() {
  const message = input.value.trim();
  if (!message) {
    return;
  }
  emit("send", message);
  input.value = "";
}

function hasArtifacts(message: ChatMessage) {
  return message.role === "assistant" && message.artifacts && Object.keys(message.artifacts).length > 0;
}

function hasTrace(message: ChatMessage) {
  return message.role === "assistant" && Boolean(message.execution_trace?.steps?.length);
}

function scrollToBottom() {
  nextTick(() => {
    if (messagesPanel.value) {
      messagesPanel.value.scrollTop = messagesPanel.value.scrollHeight;
    }
  });
}

defineExpose({ scrollToBottom });
</script>

<template>
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
        <div class="message-stack">
          <div class="message-bubble">
            {{ message.content }}
          </div>
          <ExecutionTraceTimeline v-if="hasTrace(message)" :trace="message.execution_trace" />
          <ArtifactCards v-if="hasArtifacts(message)" :artifacts="message.artifacts" />
        </div>
      </article>
      <article v-if="loading" class="message-row assistant">
        <div class="message-bubble pending">正在规划和调用工具...</div>
      </article>
    </div>

    <form class="composer" @submit.prevent="submitMessage">
      <textarea
        v-model="input"
        rows="2"
        placeholder="输入旅行需求，例如：我要从郑州去杭州玩三天，预算3000，6月10日出发"
        @keydown.enter.exact.prevent="submitMessage"
      />
      <button class="send-button" type="submit" :disabled="loading || !input.trim()">
        <Send :size="18" />
        发送
      </button>
    </form>
  </section>
</template>
