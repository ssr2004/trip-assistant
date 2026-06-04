<script setup lang="ts">
import type { ChatArtifacts } from "../types";

withDefaults(
  defineProps<{
    artifacts?: ChatArtifacts;
  }>(),
  {
    artifacts: () => ({}),
  },
);

function formatDistance(distance?: number | string | null) {
  const meters = Number(distance || 0);
  if (meters >= 1000) {
    return `${(meters / 1000).toFixed(1)} 公里`;
  }
  return `${Math.round(meters)} 米`;
}

function formatDuration(duration?: number | string | null) {
  const seconds = Number(duration || 0);
  if (!seconds) {
    return "0 分钟";
  }
  return `${Math.max(Math.round(seconds / 60), 1)} 分钟`;
}
function artifactKey(value: unknown, fallback: string): string {
  if (typeof value === "string" || typeof value === "number") {
    return String(value);
  }
  return fallback;
}
</script>

<template>
  <div class="artifact-panel">
    <section v-if="artifacts.itinerary" class="artifact-card itinerary-card">
      <div class="artifact-header">
        <strong>{{ artifacts.itinerary.title || "每日行程" }}</strong>
        <span v-if="artifacts.itinerary.destination">
          {{ artifacts.itinerary.destination }}
        </span>
      </div>
      <div class="day-grid">
        <article
          v-for="day in artifacts.itinerary.days"
          :key="artifactKey(day.day, `day-${day.title || 'untitled'}`)"
          class="day-card"
        >
          <b>Day {{ day.day }} · {{ day.title }}</b>
          <p v-if="day.activities?.length">{{ day.activities.join(" -> ") }}</p>
          <small v-if="day.notes">{{ day.notes }}</small>
        </article>
      </div>
    </section>

    <section v-if="artifacts.weather" class="artifact-card">
      <div class="artifact-header">
        <strong>{{ artifacts.weather.city || "目的地" }}天气</strong>
        <span>旅行建议</span>
      </div>
      <div class="weather-list">
        <article
          v-for="forecast in artifacts.weather.forecasts"
          :key="artifactKey(forecast.date, `forecast-${forecast.weather || 'unknown'}`)"
          class="weather-item"
          :class="{ rainy: forecast.suitable_for_outdoor === false }"
        >
          <b>{{ forecast.date }}</b>
          <span>{{ forecast.weather }} · {{ forecast.temperature }}</span>
        </article>
      </div>
    </section>

    <section v-if="artifacts.weather_adjustment" class="artifact-card">
      <div class="artifact-header">
        <strong>雨天调整依据</strong>
        <span>{{ artifacts.weather_adjustment.city || "目的地" }}</span>
      </div>
      <ul class="compact-list">
        <li
          v-for="item in artifacts.weather_adjustment.adjusted_days || []"
          :key="`${item.day}-${item.date}`"
        >
          第{{ item.day }}天：{{ item.weather }}，{{ item.advice }}
        </li>
      </ul>
    </section>

    <section v-if="artifacts.route" class="artifact-card">
      <div class="artifact-header">
        <strong>路线优化摘要</strong>
        <span>
          {{ formatDistance(artifacts.route.total_distance) }} /
          {{ formatDuration(artifacts.route.total_duration) }}
        </span>
      </div>
      <ul class="compact-list">
        <li
          v-for="segment in artifacts.route.segments || []"
          :key="`${segment.from}-${segment.to}`"
        >
          {{ segment.from }} -> {{ segment.to }}：{{ formatDistance(segment.distance) }}
        </li>
      </ul>
    </section>

    <section v-if="artifacts.attractions" class="artifact-card">
      <div class="artifact-header">
        <strong>景点推荐</strong>
        <span>{{ artifacts.attractions.location || "目的地" }}</span>
      </div>
      <div class="attraction-grid">
        <article
          v-for="item in artifacts.attractions.items || []"
          :key="artifactKey(item.id || item.name, `attraction-${item.name || 'unknown'}`)"
          class="attraction-card"
        >
          <b>{{ item.name }}</b>
          <span>{{ item.category }} · {{ item.rating || "暂无评分" }}</span>
        </article>
      </div>
    </section>
  </div>
</template>
