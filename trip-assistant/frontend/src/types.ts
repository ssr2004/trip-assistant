export type MessageRole = "assistant" | "user";

export interface ItineraryDayArtifact {
  day?: number | string | null;
  title?: string | null;
  activities?: string[];
  notes?: string | null;
  [key: string]: unknown;
}

export interface ItineraryArtifact {
  title?: string | null;
  origin?: string | null;
  destination?: string | null;
  duration?: number | string | null;
  budget?: number | string | null;
  summary?: string | null;
  days?: ItineraryDayArtifact[];
  budget_summary?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface WeatherForecastArtifact {
  date?: string | null;
  weather?: string | null;
  temperature?: number | string | null;
  suitable_for_outdoor?: boolean | null;
  [key: string]: unknown;
}

export interface WeatherArtifact {
  city?: string | null;
  forecasts?: WeatherForecastArtifact[];
  travel_advice?: string[];
  [key: string]: unknown;
}

export interface WeatherAdjustmentDayArtifact {
  day?: number | string | null;
  date?: string | null;
  weather?: string | null;
  temperature?: number | string | null;
  advice?: string | null;
  [key: string]: unknown;
}

export interface WeatherAdjustmentArtifact {
  city?: string | null;
  adjusted_days?: WeatherAdjustmentDayArtifact[];
  forecasts?: WeatherForecastArtifact[];
  [key: string]: unknown;
}

export interface RouteSegmentArtifact {
  from?: string | null;
  to?: string | null;
  distance?: number | null;
  duration?: number | null;
  [key: string]: unknown;
}

export interface RouteArtifact {
  day?: number | string | null;
  ordered_places?: Array<string | Record<string, unknown>>;
  segments?: RouteSegmentArtifact[];
  total_distance?: number;
  total_duration?: number;
  mode?: string | null;
  [key: string]: unknown;
}

export interface AttractionItemArtifact {
  id?: number | string | null;
  name?: string | null;
  category?: string | null;
  rating?: number | string | null;
  address?: string | null;
  location?: string | null;
  [key: string]: unknown;
}

export interface ArtifactSource {
  title?: string | null;
  content?: string | null;
  source?: string | null;
  score?: number | null;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface AttractionsArtifact {
  location?: string | null;
  items?: AttractionItemArtifact[];
  sources?: ArtifactSource[];
  [key: string]: unknown;
}

export interface ChatArtifacts {
  itinerary?: ItineraryArtifact | null;
  weather?: WeatherArtifact | null;
  weather_adjustment?: WeatherAdjustmentArtifact | null;
  route?: RouteArtifact | null;
  attractions?: AttractionsArtifact | null;
}

export interface TraceStep {
  stage: "intent" | "context" | "planning" | "tool" | "task" | "rag" | string;
  label: string;
  status: "success" | "failed" | string;
  detail?: string | null;
  task_type?: string | null;
  tool?: string | null;
  duration_ms?: number | null;
  execution_mode?: string | null;
  error_type?: string | null;
  result_summary?: string | null;
  source_count?: number | null;
  dependency_ids?: string[] | null;
  resolved_dependencies?: string[] | null;
  missing_dependencies?: string[] | null;
  failed_dependencies?: string[] | null;
  dependency_context_keys?: string[] | null;
  dependency_error_count?: number | null;
  failure_category?: string | null;
  recoverable?: boolean | null;
  degraded?: boolean | null;
  fallback_used?: boolean | null;
  recovery_strategy?: string | null;
  degradation_reason?: string | null;
  provider?: string | null;
  api_status?: string | null;
  cache_hit?: boolean | null;
  cache_backend?: string | null;
  cache_write?: boolean | null;
  memory_preference_source?: string | null;
  memory_used_preferences?: string[] | null;
  memory_preference_count?: number | null;
}

export interface ExecutionTrace {
  steps: TraceStep[];
  summary: {
    intent?: string;
    task_count?: number;
    tool_count?: number;
    failed_count?: number;
    source_count?: number;
    total_duration_ms?: number;
    planner_mode?: string;
    planner_mode_config?: string;
    llm_planner_auto_route?: boolean;
    llm_planner_complexity_score?: number;
    llm_planner_complexity_signals?: string[];
    llm_planner_route_decision?: string;
    llm_planner_enabled?: boolean;
    llm_planner_available?: boolean;
    llm_planner_attempted?: boolean;
    llm_planner_adopted?: boolean;
    llm_planner_fallback_reason?: string;
    llm_planner_duration_ms?: number;
    llm_planner_total_tokens?: number;
    llm_call_count?: number;
    llm_success_count?: number;
    llm_failure_count?: number;
    llm_fallback_count?: number;
    llm_repair_count?: number;
    llm_repair_success_count?: number;
    llm_duration_ms?: number;
    llm_prompt_tokens?: number;
    llm_completion_tokens?: number;
    llm_total_tokens?: number;
    llm_token_usage_available?: boolean;
    llm_cost_basis?: string;
    tool_total_duration_ms?: number;
    real_api_count?: number;
    mock_fallback_count?: number;
    template_task_count?: number;
    dynamic_rag_count?: number;
    internal_task_count?: number;
    degraded_count?: number;
    recoverable_failure_count?: number;
    fallback_used_count?: number;
    unrecoverable_failure_count?: number;
    external_api_degraded_count?: number;
    external_api_failed_count?: number;
    api_cache_hit_count?: number;
    api_cache_write_count?: number;
    recovery_strategy_counts?: Record<string, number>;
    memory_personalization_applied?: boolean;
    memory_preference_count?: number;
    memory_conflict_count?: number;
    memory_preference_fields?: string[];
    [key: string]: number | string | boolean | string[] | Record<string, number> | undefined;
  };
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  artifacts: ChatArtifacts;
  execution_trace: ExecutionTrace;
}

export interface ChatResponse {
  session_id: string;
  response: string;
  artifacts?: ChatArtifacts | null;
  execution_trace?: ExecutionTrace | null;
}

export interface ApiErrorPayload {
  code: string;
  message: string;
  recoverable: boolean;
}

export interface ApiErrorResponse {
  detail: unknown;
  error: ApiErrorPayload;
}

export interface HistoryMessage {
  role: MessageRole | string;
  content: string;
  timestamp?: string | null;
  run_id?: string | null;
  source?: string | null;
}

export interface HistoryResponse {
  session_id: string;
  history: HistoryMessage[];
}

export interface TaskSummaryItem {
  task_id?: string | null;
  task_type?: string | null;
  tool?: string | null;
  name?: string | null;
  success: boolean;
  error?: string | null;
  duration_ms?: number | null;
  execution_mode?: string | null;
  degraded?: boolean | null;
  fallback_used?: boolean | null;
  result_summary?: string | null;
}

export interface SessionRunRecord {
  run_id: string;
  session_id: string;
  user_message: string;
  ai_message: string;
  intent_type?: string | null;
  task_count: number;
  failed_count: number;
  artifact_keys: string[];
  trace_summary: Record<string, unknown>;
  artifacts: ChatArtifacts;
  execution_trace: ExecutionTrace;
  task_summary: TaskSummaryItem[];
  created_at: string;
}

export interface SessionRunsResponse {
  session_id: string;
  runs: SessionRunRecord[];
  count: number;
}

export interface ClearHistoryResponse {
  message: string;
}

export interface ExternalServiceStatus {
  name: string;
  provider: string;
  capability: "poi_search" | "route_distance" | "weather_forecast" | string;
  api_key_configured: boolean;
  key_source?: string | null;
  mock_enabled: boolean;
  mode: "real_api" | "mock_fallback" | "unavailable" | string;
  probe_type: string;
}

export interface ExternalStatusSummary {
  total: number;
  real_api_count: number;
  mock_fallback_count: number;
  unavailable_count: number;
  all_operational: boolean;
}

export interface ExternalStatusResponse {
  services: ExternalServiceStatus[];
  summary: ExternalStatusSummary;
}

export interface LLMStatusResponse {
  provider: string;
  model: string;
  base_url: string;
  api_key_configured: boolean;
  key_source?: string | null;
  mode: "real_llm" | "rule_fallback" | string;
  fallback_enabled: boolean;
  openai_compatible: boolean;
}
