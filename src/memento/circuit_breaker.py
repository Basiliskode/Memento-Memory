"""In-memory circuit breaker for LLM calls.

Tracks consecutive failures and enters a cooldown period after the configured
threshold is reached. On cooldown expiry or successful call, the counter resets.

Thread-safe for a single instance (not shared across processes — in-memory only).
"""

import logging
import time

logger = logging.getLogger(__name__)


class LLMCircuitBreaker:
    """In-memory circuit breaker that protects LLM-dependent code from cascading failure.

    After ``max_failures`` consecutive failures the breaker "opens" (enters cooldown).
    Calls to ``is_available()`` return ``False`` until the cooldown expires.
    A single ``record_success()`` resets the failure counter and exits cooldown immediately.

    Args:
        max_failures: Consecutive failures before cooldown (default: 3).
        cooldown_seconds: Duration of cooldown in seconds (default: 60).
    """

    def __init__(self, max_failures: int = 3, cooldown_seconds: int = 60) -> None:
        self._max = max_failures
        self._cooldown = cooldown_seconds
        self._failures = 0
        self._cooldown_until: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_failure(self) -> None:
        """Record a failure.

        If failures reach ``max_failures``, enter cooldown.
        """
        self._failures += 1
        if self._failures >= self._max:
            self._cooldown_until = time.monotonic() + self._cooldown
            logger.warning(
                "Circuit breaker opened after %d failures, "
                "cooldown until monotonic=%.2f",
                self._failures,
                self._cooldown_until,
            )

    def record_success(self) -> None:
        """Record a success — reset failure counter and exit cooldown."""
        self._failures = 0
        self._cooldown_until = 0.0

    def is_available(self) -> bool:
        """Check whether the breaker allows calls.

        Returns ``True`` if no cooldown is active, or if the cooldown has
        already expired (auto-recovery).
        """
        if self._cooldown_until == 0.0:
            return True
        if time.monotonic() >= self._cooldown_until:
            self._failures = 0
            self._cooldown_until = 0.0
            logger.info("Circuit breaker auto-recovered after cooldown expiry")
            return True
        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def failures(self) -> int:
        """Current consecutive failure count."""
        return self._failures

    def __repr__(self) -> str:
        status = "open" if not self.is_available() else "closed"
        return f"LLMCircuitBreaker(failures={self._failures}/{self._max}, status={status})"

    @property
    def in_cooldown(self) -> bool:
        """Whether the breaker is currently in cooldown (open)."""
        return not self.is_available()
