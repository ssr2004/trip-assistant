import type {
  ApiErrorResponse,
  ChatResponse,
  ClearHistoryResponse,
  ExternalStatusResponse,
  HistoryResponse,
  LLMStatusResponse,
  SessionRunsResponse,
} from "./types";

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
    throw new Error(errorMessage(data, "请求处理失败"));
  }
  return data;
}

export async function fetchExternalStatus(): Promise<ExternalStatusResponse> {
  const response = await fetch("/api/external/status");
  const data = await response.json();
  if (!response.ok) {
    throw new Error(errorMessage(data, "外部 API 状态获取失败"));
  }
  return data;
}

export async function fetchLLMStatus(): Promise<LLMStatusResponse> {
  const response = await fetch("/api/llm/status");
  const data = await response.json();
  if (!response.ok) {
    throw new Error(errorMessage(data, "LLM 状态获取失败"));
  }
  return data;
}

export async function fetchHistory(sessionId: string): Promise<HistoryResponse> {
  const response = await fetch(`/api/history/${encodeURIComponent(sessionId)}`);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(errorMessage(data, "历史记录获取失败"));
  }
  return data;
}

export async function fetchSessionRuns(sessionId: string, limit = 20): Promise<SessionRunsResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  const response = await fetch(`/api/history/${encodeURIComponent(sessionId)}/runs?${params.toString()}`);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(errorMessage(data, "会话运行历史获取失败"));
  }
  return data;
}

export async function clearHistory(sessionId: string): Promise<ClearHistoryResponse> {
  const response = await fetch(`/api/history/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(errorMessage(data, "历史记录清理失败"));
  }
  return data;
}

function errorMessage(payload: ApiErrorResponse | Record<string, unknown>, fallback: string): string {
  const error = "error" in payload && typeof payload.error === "object" ? payload.error as { message?: unknown } : null;
  if (typeof error?.message === "string" && error.message) {
    return error.message;
  }
  if ("detail" in payload && typeof payload.detail === "string" && payload.detail) {
    return payload.detail;
  }
  if ("error" in payload && typeof payload.error === "string" && payload.error) {
    return payload.error;
  }
  return fallback;
}
