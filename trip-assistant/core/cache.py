"""
外部API缓存基础设施
支持Redis缓存，并在Redis不可用时降级到内存缓存
"""
import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple

from app.config import settings

try:  # pragma: no cover - 依赖是否安装由运行环境决定
    from redis.asyncio import Redis
except Exception:  # pragma: no cover
    Redis = None


SENSITIVE_PARAM_NAMES = {"key", "api_key", "apiKey", "secret", "token", "access_token"}


class CacheBackend:
    """缓存后端协议"""

    name = "base"

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """读取缓存"""
        raise NotImplementedError

    async def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        """写入缓存"""
        raise NotImplementedError


class MemoryCacheBackend(CacheBackend):
    """内存缓存后端，作为Redis不可用时的降级实现"""

    name = "memory"

    def __init__(self):
        self._store: Dict[str, Tuple[Dict[str, Any], float]] = {}

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """读取未过期的内存缓存"""
        cached = self._store.get(key)
        if not cached:
            return None
        value, expires_at = cached
        if expires_at and expires_at < time.time():
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        """写入内存缓存"""
        expires_at = time.time() + ttl if ttl > 0 else 0
        self._store[key] = (value, expires_at)


class RedisCacheBackend(CacheBackend):
    """Redis缓存后端"""

    name = "redis"
    _unavailable_until = 0.0
    _cooldown_seconds = 60

    def __init__(self, redis_url: str):
        if Redis is None:
            raise RuntimeError("redis依赖未安装，无法启用Redis缓存。")
        self.redis_url = redis_url
        self.client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.2,
            socket_timeout=0.2,
            retry_on_timeout=False,
        )

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """从Redis读取JSON缓存"""
        self._ensure_available()
        try:
            cached = await self.client.get(key)
        except Exception:
            self._mark_unavailable()
            raise
        if not cached:
            return None
        return json.loads(cached)

    async def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        """将JSON缓存写入Redis"""
        self._ensure_available()
        payload = json.dumps(value, ensure_ascii=False)
        try:
            if ttl > 0:
                await self.client.set(key, payload, ex=ttl)
                return
            await self.client.set(key, payload)
        except Exception:
            self._mark_unavailable()
            raise

    def _ensure_available(self) -> None:
        """Redis短时间不可用时快速失败，避免本地无Redis导致测试和Agent变慢"""
        if self._unavailable_until > time.time():
            raise RuntimeError("redis temporarily unavailable")

    def _mark_unavailable(self) -> None:
        """记录Redis短时间不可用"""
        self.__class__._unavailable_until = time.time() + self._cooldown_seconds


class ExternalAPICache:
    """外部API缓存服务"""

    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        fallback_backend: Optional[CacheBackend] = None,
        enabled: Optional[bool] = None,
        ttl: Optional[int] = None,
        prefix: str = "travelmind:external_api",
    ):
        self.backend = backend or MemoryCacheBackend()
        self.fallback_backend = fallback_backend or MemoryCacheBackend()
        self.enabled = settings.EXTERNAL_API_CACHE_ENABLED if enabled is None else enabled
        self.ttl = settings.EXTERNAL_API_CACHE_TTL if ttl is None else ttl
        self.prefix = prefix

    def build_key(self, provider: str, resource: str, params: Dict[str, Any]) -> str:
        """基于提供方、资源类型和归一化参数构建缓存Key"""
        normalized_params = self._normalize_params(params)
        payload = json.dumps(normalized_params, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"{self.prefix}:{provider}:{resource}:{digest}"

    async def get(
        self,
        provider: str,
        resource: str,
        params: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """读取外部API缓存，并返回缓存元数据"""
        key = self.build_key(provider, resource, params)
        if not self.enabled:
            return None, self._metadata(key=key, hit=False, backend="disabled", enabled=False)

        try:
            value = await self.backend.get(key)
            return value, self._metadata(key=key, hit=value is not None, backend=self.backend.name)
        except Exception as exc:
            fallback_value = await self.fallback_backend.get(key)
            return fallback_value, self._metadata(
                key=key,
                hit=fallback_value is not None,
                backend=f"{self.fallback_backend.name}_fallback",
                error=str(exc),
            )

    async def set(
        self,
        provider: str,
        resource: str,
        params: Dict[str, Any],
        value: Dict[str, Any],
    ) -> Dict[str, Any]:
        """写入外部API缓存，并返回缓存写入元数据"""
        key = self.build_key(provider, resource, params)
        if not self.enabled:
            return self._metadata(key=key, hit=False, backend="disabled", enabled=False, write=False)

        try:
            await self.backend.set(key, value, self.ttl)
            return self._metadata(key=key, hit=False, backend=self.backend.name, write=True)
        except Exception as exc:
            await self.fallback_backend.set(key, value, self.ttl)
            return self._metadata(
                key=key,
                hit=False,
                backend=f"{self.fallback_backend.name}_fallback",
                error=str(exc),
                write=True,
            )

    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """归一化缓存参数，避免API Key等敏感信息进入缓存Key"""
        normalized = {}
        for key, value in (params or {}).items():
            if key in SENSITIVE_PARAM_NAMES:
                continue
            normalized[key] = value
        return normalized

    def _metadata(
        self,
        key: str,
        hit: bool,
        backend: str,
        enabled: bool = True,
        error: Optional[str] = None,
        write: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """构建缓存元数据"""
        metadata = {
            "cache_enabled": enabled,
            "cache_hit": hit,
            "cache_backend": backend,
            "cache_key": key,
            "cache_ttl": self.ttl,
        }
        if error:
            metadata["cache_error"] = error
        if write is not None:
            metadata["cache_write"] = write
        return metadata


def create_external_api_cache() -> ExternalAPICache:
    """根据配置创建外部API缓存服务"""
    backend_name = settings.EXTERNAL_API_CACHE_BACKEND.lower()
    fallback_backend = MemoryCacheBackend()

    if backend_name == "redis":
        try:
            backend: CacheBackend = RedisCacheBackend(settings.REDIS_URL)
        except RuntimeError:
            backend = fallback_backend
    else:
        backend = MemoryCacheBackend()

    return ExternalAPICache(
        backend=backend,
        fallback_backend=fallback_backend,
        enabled=settings.EXTERNAL_API_CACHE_ENABLED,
        ttl=settings.EXTERNAL_API_CACHE_TTL,
    )
