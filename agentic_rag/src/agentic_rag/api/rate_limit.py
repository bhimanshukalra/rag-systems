import math
import threading
import time
from collections.abc import Callable
from functools import lru_cache

from fastapi import Depends, HTTPException

from agentic_rag.config import Settings, get_settings


class FixedWindowLimiter:
    """In-memory fixed-window counter.

    Process-local on purpose: this project runs a single uvicorn worker, so
    no shared store is needed. Running more than one worker would give each
    its own counter and silently multiply the effective limit.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._window_start = clock()
        self._count = 0

    def acquire(self) -> float | None:
        """Take one slot. Returns None if allowed, else seconds until reset."""
        with self._lock:
            now = self._clock()
            if now - self._window_start >= self.window_seconds:
                self._window_start = now
                self._count = 0
            if self._count >= self.limit:
                return self._window_start + self.window_seconds - now
            self._count += 1
            return None


@lru_cache
def _limiter_for(limit: int) -> FixedWindowLimiter:
    return FixedWindowLimiter(limit)


def enforce_rate_limit(settings: Settings = Depends(get_settings)) -> None:
    retry_after = _limiter_for(settings.rate_limit_requests_per_minute).acquire()
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded",
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )
