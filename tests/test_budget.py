"""Tests for shared token budget tracking, honest unknown representation, and enforcement."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from faultline.agents.base import BaseAgent
from faultline.budget import BudgetExceededError, RunBudget
from faultline.models import BaseModel, TokenUsage
from faultline.tracer import ExecutionTracer


class DummyResult(BaseModel):
    status: str


class DummyAgent(BaseAgent):
    async def run(self, *args, **kwargs):
        return await self.call_llm_structured("prompt", DummyResult)


def make_mock_response(prompt_tokens=None, completion_tokens=None):
    choice = MagicMock()
    choice.message.content = '{"status": "ok"}'
    response = MagicMock()
    response.choices = [choice]
    if prompt_tokens is not None and completion_tokens is not None:
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        usage.total_tokens = prompt_tokens + completion_tokens
        response.usage = usage
    else:
        response.usage = None
    return response


def test_subsequent_call_blocked_when_known_usage_exhausts_budget():
    """Verify that a subsequent call is blocked when prior known usage reaches or exceeds max_tokens."""
    budget = RunBudget(max_tokens=100)

    # First call consumes 60 + 40 = 100 tokens
    budget.record_usage(prompt_tokens=60, completion_tokens=40)
    assert budget.prompt_tokens == 60
    assert budget.completion_tokens == 40
    assert budget.current_calls == 1

    # Check budget before subsequent call: must raise BudgetExceededError
    with pytest.raises(BudgetExceededError) as exc_info:
        budget.check_budget()
    assert "exhausted by known usage (100 tokens)" in str(exc_info.value)


def test_token_usage_accumulated_correctly():
    """Verify that actual token counts from multiple calls are preserved and accumulated accurately."""
    budget = RunBudget(max_tokens=1000)

    # Call 1
    budget.record_usage(prompt_tokens=50, completion_tokens=30)
    # Call 2
    budget.record_usage(prompt_tokens=70, completion_tokens=45)

    assert budget.prompt_tokens == 120
    assert budget.completion_tokens == 75
    assert budget.current_calls == 2

    usage = budget.get_token_usage()
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 75
    assert usage.total_tokens == 195


def test_unknown_token_usage_recorded_honestly():
    """Verify that unavailable token usage is recorded as unknown, never fabricated or assumed zero."""
    budget = RunBudget(max_tokens=500)

    # Provider reports no usage metadata
    budget.record_usage(prompt_tokens="unknown", completion_tokens="unknown")
    assert budget.has_unknown_tokens is True

    usage = budget.get_token_usage()
    assert usage.prompt_tokens == "unknown"
    assert usage.completion_tokens == "unknown"
    assert usage.total_tokens == "unknown"

    # Now add partial known usage: partial must be reported honestly as >=
    budget.record_usage(prompt_tokens=50, completion_tokens=25)
    partial_usage = budget.get_token_usage()
    assert ">=50" in str(partial_usage.prompt_tokens)
    assert ">=25" in str(partial_usage.completion_tokens)
    assert partial_usage.total_tokens == "unknown"


@pytest.mark.asyncio
async def test_agent_subsequent_call_blocked_by_shared_token_budget(tmp_path: Path):
    """Verify that an agent's subsequent model call is blocked when shared token budget is exhausted."""
    budget = RunBudget(max_tokens=80, max_calls=10)
    tracer = ExecutionTracer(trace_file=tmp_path / "budget_trace.jsonl")

    # Mock client returns 50 prompt + 35 completion = 85 tokens (exceeds 80)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=make_mock_response(prompt_tokens=50, completion_tokens=35)
    )

    agent = DummyAgent(
        name="test_agent",
        sandbox_dir=tmp_path,
        tracer=tracer,
        groq_client=mock_client,
        budget=budget,
    )

    # First call succeeds and consumes 85 tokens
    res1 = await agent.run()
    assert res1.status == "ok"
    assert budget.prompt_tokens == 50
    assert budget.completion_tokens == 35

    # Second call must be blocked before making Groq call
    with pytest.raises(BudgetExceededError) as exc_info:
        await agent.run()
    assert "exhausted by known usage (85 tokens)" in str(exc_info.value)
    # The API was only called once
    assert mock_client.chat.completions.create.call_count == 1


def test_tracer_preserves_token_counts_and_redacts_secrets(tmp_path: Path):
    """Verify that actual token counts are preserved in trace while credentials remain redacted."""
    trace_file = tmp_path / "test_redact.jsonl"
    tracer = ExecutionTracer(trace_file=trace_file)

    tracer.log_event(
        agent="test_agent",
        event_type="llm_call_success",
        payload={
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "total_tokens": 165,
            "groq_api_key": "gsk_secret_12345",
            "auth_token": "secret_bearer_token",
        },
    )

    content = trace_file.read_text(encoding="utf-8")
    # Token counts must be preserved as real integers
    assert '"prompt_tokens": 120' in content
    assert '"completion_tokens": 45' in content
    assert '"total_tokens": 165' in content
    # Secrets must be redacted
    assert '"groq_api_key": "[REDACTED]"' in content
    assert '"auth_token": "[REDACTED]"' in content
