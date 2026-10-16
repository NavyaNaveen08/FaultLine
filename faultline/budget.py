"""Run-level budget and deadline tracking for the Faultline harness."""

from __future__ import annotations

import time
from typing import Any, Optional

from faultline.config import settings
from faultline.models import TokenUsage


class BudgetExceededError(Exception):
    """Raised when the shared model call count or token budget is exhausted."""


class GlobalTimeoutError(Exception):
    """Raised when the global execution deadline has expired."""


class RunBudget:
    """Tracks global deadline, model call count, and token usage across all agents."""

    def __init__(
        self,
        max_calls: Optional[int] = None,
        max_tokens: Optional[int] = None,
        overall_timeout: Optional[float] = None,
    ) -> None:
        self.max_calls = max_calls if max_calls is not None else settings.max_total_model_calls
        self.max_tokens = max_tokens if max_tokens is not None else settings.max_total_tokens
        self.overall_timeout = (
            overall_timeout if overall_timeout is not None else settings.overall_timeout_seconds
        )
        self.start_time = time.monotonic()
        self.deadline = self.start_time + self.overall_timeout

        self.current_calls: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.has_unknown_tokens: bool = False

    def remaining_time(self) -> float:
        """Seconds remaining before global deadline."""
        return max(0.0, self.deadline - time.monotonic())

    def check_budget(self) -> None:
        """Assert that deadline and call/token budgets have not been breached.
        
        Whenever prior token usage is known, enforces max_tokens strictly.
        Note on Unknown Usage: If provider token counts are unavailable (has_unknown_tokens=True),
        a strict upper-bound token guarantee cannot be proven without fabricating counts.
        Faultline never invents numbers or treats unknown usage as zero; instead, call limit
        (max_calls) and execution deadline (overall_timeout) serve as hard safeguards.
        """
        if time.monotonic() >= self.deadline:
            raise GlobalTimeoutError(
                f"Global investigation deadline of {self.overall_timeout:.1f}s expired."
            )
        if self.current_calls >= self.max_calls:
            raise BudgetExceededError(
                f"Model call budget of {self.max_calls} calls exhausted (current: {self.current_calls})."
            )
        if self.max_tokens is not None:
            known_total = self.prompt_tokens + self.completion_tokens
            if known_total >= self.max_tokens:
                raise BudgetExceededError(
                    f"Token budget of {self.max_tokens} exhausted by known usage ({known_total} tokens)."
                )

    def record_usage(self, prompt_tokens: Any, completion_tokens: Any) -> None:
        """Record model call and token consumption or flag unknown usage."""
        self.current_calls += 1
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
        else:
            self.has_unknown_tokens = True

    def get_token_usage(self) -> TokenUsage:
        """Return TokenUsage with actual counts or 'unknown'."""
        if self.has_unknown_tokens:
            if self.prompt_tokens == 0 and self.completion_tokens == 0:
                return TokenUsage(
                    prompt_tokens="unknown",
                    completion_tokens="unknown",
                    total_tokens="unknown",
                )
            return TokenUsage(
                prompt_tokens=f">={self.prompt_tokens} (partial)",
                completion_tokens=f">={self.completion_tokens} (partial)",
                total_tokens="unknown",
            )
        return TokenUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
        )
