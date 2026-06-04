import { expect, test } from "@playwright/test";
import type { Page, Route } from "@playwright/test";

const demoPrompts = {
  plan: "我要从郑州去杭州玩三天，预算3000，6月10日出发",
  weather: "如果下雨怎么办？",
  route: "帮我按距离优化一下第二天行程",
  followup: "西湖在哪里？",
};

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => window.localStorage.clear());
  await mockBackend(page);
});

test("runs the resume demo script with artifacts and trace", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "多智能体旅行规划系统" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "模型能力状态" })).toBeVisible();
  await expect(page.getByText("真实 LLM")).toBeVisible();
  await expect(page.getByText("deepseek / deepseek-v4-flash")).toBeVisible();
  await expect(page.getByRole("heading", { name: "外部能力状态" })).toBeVisible();
  await expect(page.getByText("3 真实")).toBeVisible();
  await expect(page.getByText("AMAP_API_KEY")).toHaveCount(3);
  await expect(page.getByRole("button", { name: /1 完整规划/ })).toBeVisible();

  await page.getByRole("button", { name: /1 完整规划/ }).click();
  await expect(page.getByText("已为您规划郑州到杭州的3天旅行方案")).toBeVisible();
  await expect(page.getByText("每日行程").first()).toBeVisible();
  await expect(page.getByText("景点推荐").first()).toBeVisible();
  await expect(page.getByText("Execution Trace").first()).toBeVisible();
  await expect(page.getByText("LLM 1 calls").first()).toBeVisible();
  await expect(page.getByText("Tool · 搜索景点")).toBeVisible();

  await page.getByRole("button", { name: /2 雨天调整/ }).click();
  await expect(page.getByText("已根据您的要求调整行程：雨天优先安排室内景点。")).toBeVisible();
  await expect(page.getByText("雨天调整依据")).toBeVisible();
  await expect(page.getByText("Revision").first()).toBeVisible();

  await page.getByRole("button", { name: /3 路线优化/ }).click();
  await expect(page.getByText("已根据您的要求调整行程：第二天已按距离重新排序。")).toBeVisible();
  await expect(page.getByText("路线优化摘要")).toBeVisible();
  await expect(page.getByText("西湖 -> 灵隐寺：5.2 公里")).toBeVisible();

  await page.getByRole("button", { name: /4 景点追问/ }).click();
  await expect(page.getByText("根据刚才推荐的外部景点数据，我查到：")).toBeVisible();
  await expect(page.getByText("Dynamic RAG").first()).toBeVisible();
  await expect(page.getByText("西湖").first()).toBeVisible();
});

test("keeps the demo controls usable on mobile viewport", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "多智能体旅行规划系统" })).toBeVisible();
  await expect(page.getByRole("button", { name: /1 完整规划/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /4 景点追问/ })).toBeVisible();

  await page.getByPlaceholder("输入旅行需求，例如：我要从郑州去杭州玩三天，预算3000，6月10日出发").fill(demoPrompts.plan);
  await page.getByRole("button", { name: "发送" }).click();

  await expect(page.getByText("已为您规划郑州到杭州的3天旅行方案")).toBeVisible();
  await expect(page.getByText("Execution Trace").first()).toBeVisible();
});

async function mockBackend(page: Page) {
  await page.route("**/api/external/status", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        services: [
          buildServiceStatus("amap_poi", "poi_search"),
          buildServiceStatus("amap_route", "route_distance"),
          buildServiceStatus("weather", "weather_forecast"),
        ],
        summary: {
          total: 3,
          real_api_count: 3,
          mock_fallback_count: 0,
          unavailable_count: 0,
          all_operational: true,
        },
      }),
    });
  });

  await page.route("**/api/llm/status", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        provider: "deepseek",
        model: "deepseek-v4-flash",
        base_url: "https://api.deepseek.com",
        api_key_configured: true,
        key_source: "LLM_API_KEY",
        mode: "real_llm",
        fallback_enabled: true,
        openai_compatible: true,
      }),
    });
  });

  await page.route("**/api/chat", async (route) => {
    const body = route.request().postDataJSON() as { message?: string; session_id?: string };
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(buildChatResponse(body.message || "")),
    });
  });
}

function buildServiceStatus(name: string, capability: string) {
  return {
    name,
    provider: "amap",
    capability,
    api_key_configured: true,
    key_source: "AMAP_API_KEY",
    mock_enabled: true,
    mode: "real_api",
    probe_type: "configuration",
  };
}

function buildChatResponse(message: string) {
  if (message === demoPrompts.weather) {
    return chatResponse(
      "已根据您的要求调整行程：雨天优先安排室内景点。",
      {
        itinerary: itineraryArtifact("调整后的每日行程"),
        weather_adjustment: {
          city: "杭州",
          adjusted_days: [
            { day: 2, date: "2026-06-11", weather: "小雨", temperature: "22-27℃", advice: "优先安排室内展馆和低强度活动" },
          ],
        },
      },
      trace("itinerary_revision", [{ stage: "task", label: "修订旅行行程", execution_mode: "internal_revision", detail: "weather_adjusted_days=1" }]),
    );
  }

  if (message === demoPrompts.route) {
    return chatResponse(
      "已根据您的要求调整行程：第二天已按距离重新排序。",
      {
        itinerary: itineraryArtifact("调整后的每日行程"),
        route: {
          day: 2,
          ordered_places: ["西湖", "灵隐寺"],
          segments: [{ from: "西湖", to: "灵隐寺", distance: 5200, duration: 1600 }],
          total_distance: 5200,
          total_duration: 1600,
          mode: "walking",
        },
      },
      trace("itinerary_revision", [{ stage: "task", label: "修订旅行行程", execution_mode: "internal_revision", detail: "route_segments=1" }]),
    );
  }

  if (message === demoPrompts.followup) {
    return chatResponse(
      "根据刚才推荐的外部景点数据，我查到：\n\n西湖：位于杭州市西湖区，适合安排半天游览。",
      {
        attractions: {
          location: "西湖在哪里？",
          items: [{ name: "西湖", category: "自然风光", rating: 4.9, address: "杭州市西湖区" }],
          sources: [{ title: "西湖", content: "杭州市西湖区", source: "amap_poi" }],
        },
      },
      trace("dynamic_knowledge_query", [{ stage: "task", label: "检索动态外部知识", execution_mode: "dynamic_rag", detail: "sources=1" }]),
    );
  }

  return chatResponse(
    "已为您规划郑州到杭州的3天旅行方案。",
    {
      itinerary: itineraryArtifact("每日行程"),
      attractions: {
        location: "杭州",
        items: [
          { name: "西湖", category: "自然风光", rating: 4.9 },
          { name: "灵隐寺", category: "人文景点", rating: 4.7 },
        ],
        sources: [{ title: "杭州POI", source: "amap_poi" }],
      },
    },
    trace("travel_plan", [
      { stage: "tool", label: "搜索景点", tool: "search_attractions", execution_mode: "real_api", detail: "attractions=2" },
      { stage: "tool", label: "生成旅行行程", tool: "generate_itinerary", execution_mode: "template", detail: "itinerary_days=3" },
    ]),
  );
}

function chatResponse(response: string, artifacts: Record<string, unknown>, execution_trace: Record<string, unknown>) {
  return {
    session_id: "demo-session-module38",
    response,
    artifacts,
    execution_trace,
  };
}

function itineraryArtifact(title: string) {
  return {
    title,
    origin: "郑州",
    destination: "杭州",
    duration: 3,
    budget: 3000,
    days: [
      { day: 1, title: "西湖初体验", activities: ["西湖", "南宋御街"], notes: "低强度到达日" },
      { day: 2, title: "经典景点", activities: ["西湖", "灵隐寺"], notes: "适合路线优化" },
      { day: 3, title: "返程休整", activities: ["城市漫步", "返程"], notes: "保留机动时间" },
    ],
  };
}

function trace(intent: string, taskSteps: Array<Record<string, unknown>>) {
  const steps = [
    { stage: "intent", label: intent, status: "success", detail: "confidence=0.91" },
    { stage: "context", label: "RAG context", status: "success", detail: "static_sources=3", source_count: 3 },
    { stage: "planning", label: "Task plan", status: "success", detail: `tasks=${taskSteps.length}` },
    ...taskSteps.map((step, index) => ({
      status: "success",
      duration_ms: 12 + index,
      result_summary: step.detail,
      source_count: step.execution_mode === "dynamic_rag" ? 1 : undefined,
      ...step,
    })),
  ];
  return {
    steps,
    summary: {
      intent,
      task_count: taskSteps.length,
      tool_count: taskSteps.filter((step) => step.stage === "tool").length,
      failed_count: 0,
      source_count: 3,
      total_duration_ms: 26,
      llm_call_count: 1,
      llm_success_count: 1,
      llm_failure_count: 0,
      llm_fallback_count: 0,
      llm_repair_count: 0,
      llm_repair_success_count: 0,
      llm_duration_ms: 42,
      llm_prompt_tokens: 128,
      llm_completion_tokens: 64,
      llm_total_tokens: 192,
      llm_token_usage_available: true,
      llm_cost_basis: "provider_token_usage",
      tool_total_duration_ms: 26,
      real_api_count: taskSteps.filter((step) => step.execution_mode === "real_api").length,
      mock_fallback_count: taskSteps.filter((step) => step.execution_mode === "mock_fallback").length,
      template_task_count: taskSteps.filter((step) => step.execution_mode === "template").length,
      dynamic_rag_count: taskSteps.filter((step) => step.execution_mode === "dynamic_rag").length,
      internal_task_count: taskSteps.filter((step) => step.execution_mode === "internal_revision").length,
    },
  };
}
