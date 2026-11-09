"""Token-bucket rate limiter used as a sample code artifact for local RAG indexing."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitPolicy:
    """Configuration for a token-bucket rate limiter."""

    capacity: int
    refill_rate_per_sec: float
    burst_allowance: int = 0

    @property
    def max_tokens(self) -> float:
        return float(self.capacity + self.burst_allowance)


class TokenBucketRateLimiter:
    """Thread-safe token-bucket rate limiter for API and MCP tool throttling.

    Tokens replenish continuously at `policy.refill_rate_per_sec` up to
    `policy.max_tokens`. Each `acquire()` call consumes `cost` tokens if
    available, or returns `(False, retry_after_seconds)` when throttled.
    """

    def __init__(self, policy: RateLimitPolicy) -> None:
        if policy.capacity <= 0 or policy.refill_rate_per_sec <= 0:
            raise ValueError("capacity and refill_rate_per_sec must be positive")
        self._policy = policy
        self._tokens = float(policy.capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._last_refill)
        replenished = elapsed * self._policy.refill_rate_per_sec
        self._tokens = min(self._policy.max_tokens, self._tokens + replenished)
        self._last_refill = now

    def acquire(self, cost: int = 1) -> tuple[bool, float]:
        """Attempt to consume `cost` tokens.

        Returns:
            A `(allowed, retry_after_sec)` tuple. When `allowed` is True,
            `retry_after_sec` is `0.0`.
        """
        if cost <= 0:
            raise ValueError("cost must be >= 1")

        with self._lock:
            now = time.monotonic()
            self._refill(now)
            if self._tokens >= cost:
                self._tokens -= cost
                return True, 0.0

            deficit = cost - self._tokens
            retry_after = deficit / self._policy.refill_rate_per_sec
            return False, retry_after
