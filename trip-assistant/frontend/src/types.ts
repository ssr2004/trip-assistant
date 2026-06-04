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

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  artifacts: ChatArtifacts;
}

export interface ChatResponse {
  session_id: string;
  response: string;
  artifacts?: ChatArtifacts | null;
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
