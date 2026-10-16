"""Mocked tests for BaseAgent structured parsing, retries, timeouts, and Checker verification."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest
import groq
from pydantic import BaseModel

from faultline.agents.base import BaseAgent
from faultline.agents.checker import CheckerAgent
from faultline.budget import RunBudget
from faultline.models import EvidenceCitation, Hypothesis
from faultline.tracer import ExecutionTracer


class DummyModel(BaseModel):
    message: str
    code: int


class DummyAgent(BaseAgent):
    async def run(self, *args, **kwargs):
        return await self.call_llm_structured("prompt", DummyModel)


def make_mock_completion(content: str, prompt_tokens: int = 50, completion_tokens: int = 25):
    """Helper to mock Groq chat completion response."""
    choice = MagicMock()
    choice.message.content = content

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


@pytest.mark.asyncio
async def test_base_agent_structured_parsing(tmp_path: Path):
    tracer = ExecutionTracer(trace_file=tmp_path / "trace.jsonl")
    budget = RunBudget(max_calls=5)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=make_mock_completion('{"message": "success", "code": 200}')
    )

    agent = DummyAgent(
        name="test_agent",
        sandbox_dir=tmp_path,
        tracer=tracer,
        groq_client=mock_client,
        budget=budget,
    )

    result = await agent.run()
    assert isinstance(result, DummyModel)
    assert result.message == "success"
    assert result.code == 200
    assert budget.current_calls == 1
    assert budget.prompt_tokens == 50
    assert budget.completion_tokens == 25


@pytest.mark.asyncio
async def test_base_agent_validation_retry(tmp_path: Path):
    tracer = ExecutionTracer(trace_file=tmp_path / "trace.jsonl")

    # First attempt: invalid schema (missing required code)
    # Second attempt: valid schema
    invalid_resp = make_mock_completion('{"message": "missing_code"}')
    valid_resp = make_mock_completion('{"message": "recovered", "code": 100}')

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=[invalid_resp, valid_resp])

    agent = DummyAgent(
        name="test_agent",
        sandbox_dir=tmp_path,
        tracer=tracer,
        groq_client=mock_client,
        timeout_seconds=5.0,
    )

    result = await agent.run()
    assert result.message == "recovered"
    assert result.code == 100
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_base_agent_transient_api_retry(tmp_path: Path):
    tracer = ExecutionTracer(trace_file=tmp_path / "trace.jsonl")

    # First attempt: 429 RateLimitError
    # Second attempt: success
    rate_limit_err = groq.RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )
    valid_resp = make_mock_completion('{"message": "rate_recovered", "code": 200}')

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=[rate_limit_err, valid_resp])

    agent = DummyAgent(
        name="test_agent",
        sandbox_dir=tmp_path,
        tracer=tracer,
        groq_client=mock_client,
        timeout_seconds=5.0,
    )

    result = await agent.run()
    assert result.message == "rate_recovered"
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_base_agent_per_agent_timeout(tmp_path: Path):
    tracer = ExecutionTracer(trace_file=tmp_path / "trace.jsonl")

    async def slow_create(*args, **kwargs):
        await asyncio.sleep(0.5)
        return make_mock_completion('{"message": "slow", "code": 200}')

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=slow_create)

    agent = DummyAgent(
        name="slow_agent",
        sandbox_dir=tmp_path,
        tracer=tracer,
        groq_client=mock_client,
        timeout_seconds=0.1,  # Short timeout
    )

    with pytest.raises(asyncio.TimeoutError):
        await agent.run()


@pytest.mark.asyncio
async def test_base_agent_unknown_usage(tmp_path: Path):
    tracer = ExecutionTracer(trace_file=tmp_path / "trace.jsonl")
    budget = RunBudget()

    choice = MagicMock()
    choice.message.content = '{"message": "no_usage", "code": 0}'
    response = MagicMock()
    response.choices = [choice]
    response.usage = None  # Missing usage

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)

    agent = DummyAgent(
        name="test_agent",
        sandbox_dir=tmp_path,
        tracer=tracer,
        groq_client=mock_client,
        budget=budget,
    )

    await agent.run()
    token_usage = budget.get_token_usage()
    assert token_usage.total_tokens == "unknown"
    assert token_usage.prompt_tokens == "unknown"


def test_checker_physical_citation_verification(sample_incident_dir: Path):
    checker = CheckerAgent(
        name="checker_agent",
        sandbox_dir=sample_incident_dir,
    )

    # 1. Valid citation: correct file, correct 1-based line number (line 3), exact snippet
    cit_valid = EvidenceCitation(
        file_path="logs/payment_service.log",
        line_number=3,
        snippet="Connection timeout after 5000ms",
    )
    assert checker.verify_citation_physically(cit_valid) is True
    assert cit_valid.verified is True

    # 2. Wrong line number: snippet exists on line 3, but citation claims line 2 (must not pass)
    cit_wrong_line = EvidenceCitation(
        file_path="logs/payment_service.log",
        line_number=2,
        snippet="Connection timeout after 5000ms",
    )
    assert checker.verify_citation_physically(cit_wrong_line) is False
    assert cit_wrong_line.verified is False

    # 3. Wrong snippet: line 3 exists, but snippet is wrong
    cit_wrong_snippet = EvidenceCitation(
        file_path="logs/payment_service.log",
        line_number=3,
        snippet="Service started without errors",
    )
    assert checker.verify_citation_physically(cit_wrong_snippet) is False
    assert cit_wrong_snippet.verified is False

    # 4. Out-of-range line number (file only has 4 lines)
    cit_out_of_range = EvidenceCitation(
        file_path="logs/payment_service.log",
        line_number=999,
        snippet="Connection timeout after 5000ms",
    )
    assert checker.verify_citation_physically(cit_out_of_range) is False
    assert cit_out_of_range.verified is False

    # 5. Non-existent file
    cit_missing_file = EvidenceCitation(
        file_path="logs/non_existent.log",
        line_number=1,
        snippet="Any snippet",
    )
    assert checker.verify_citation_physically(cit_missing_file) is False
    assert cit_missing_file.verified is False

    # 6. Absolute path (must be rejected even if pointing inside incident root)
    cit_abs = EvidenceCitation(
        file_path=str((sample_incident_dir / "logs" / "payment_service.log").resolve()),
        line_number=3,
        snippet="Connection timeout after 5000ms",
    )
    assert checker.verify_citation_physically(cit_abs) is False
    assert cit_abs.verified is False

    # 7. Path traversal outside incident root
    cit_trav = EvidenceCitation(
        file_path="../../.env",
        line_number=1,
        snippet="GROQ",
    )
    assert checker.verify_citation_physically(cit_trav) is False
    assert cit_trav.verified is False

    # 8. Whitespace-mismatched snippet (extra or normalized spacing must not pass literal match)
    cit_whitespace_mismatch = EvidenceCitation(
        file_path="logs/payment_service.log",
        line_number=3,
        snippet="Connection   timeout   after   5000ms",
    )
    assert checker.verify_citation_physically(cit_whitespace_mismatch) is False
    assert cit_whitespace_mismatch.verified is False
