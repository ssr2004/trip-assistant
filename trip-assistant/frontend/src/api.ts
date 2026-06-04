import type { ChatResponse, ExternalStatusResponse, LLMStatusResponse } from "./types";

const JSON_HEADERS = {
  "Content-Type": "application/json",
};

export async function sendChatMessage(message: string, sessionId: string): Promise<ChatResponse> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({
      message,
      session_id: sessionId || undefined,
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || data.error || "请求处理失败");
  }
  return data;
}

export async function fetchExternalStatus(): Promise<ExternalStatusResponse> {
  const response = await fetch("/api/external/status");
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || data.error || "外部 API 状态获取失败");
  }
  return data;
}

export async function fetchLLMStatus(): Promise<LLMStatusResponse> {
  const response = await fetch("/api/llm/status");
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || data.error || "LLM 状态获取失败");
  }
  return data;
}
