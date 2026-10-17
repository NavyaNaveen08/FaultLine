"""Tests for single-agent baseline, task fixtures, and comparative evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from faultline.baseline import SingleAgentBaseline
from faultline.evaluator import (
    DEFAULT_TASKS,
    RubricScore,
    evaluate_rubric,
    evaluate_task,
    run_evaluation_suite,
)
from faultline.models import (
    EvidenceCitation,
    Hypothesis,
    InvestigationReport,
)
from faultline.tracer import ExecutionTracer


def test_task_fixtures_exist_and_conform_to_schema():
    """Verify that all three incident task fixtures exist with valid logs, configs, and incident descriptions."""
    assert len(DEFAULT_TASKS) == 3

    for task in DEFAULT_TASKS:
        assert task.incident_file.exists(), f"Missing incident file for {task.task_id}"
        assert task.evidence_dir.exists(), f"Missing evidence dir for {task.task_id}"

        incident_content = task.incident_file.read_text(encoding="utf-8")
        assert len(incident_content.strip()) > 30

        # Check logs directory
        logs_dir = task.evidence_dir / "logs"
        assert logs_dir.exists() and logs_dir.is_dir()
        log_files = list(logs_dir.glob("*.log"))
        assert len(log_files) >= 1

        # Check config directory
        config_dir = task.evidence_dir / "config"
        assert config_dir.exists() and config_dir.is_dir()
        services_file = config_dir / "services.json"
        assert services_file.exists()
        config_data = json.loads(services_file.read_text(encoding="utf-8"))
        assert "service" in config_data


def test_rubric_scoring_deterministic():
    """Verify that evaluate_rubric computes objective dimensions without fabrication."""
    report = InvestigationReport(
        incident_title="Redis Pool Outage",
        executive_summary="Redis connection pool exhausted under load.",
        root_cause="Redis connection pool was capped at 5 connections.",
        confidence_level="High",
        primary_hypothesis=Hypothesis(
            id="hyp-1",
            title="Redis Connection Starvation",
            mechanism="Connection pool exhausted causing read timeouts.",
            likelihood="high",
            confidence_score=0.9,
            supporting_evidence=[
                EvidenceCitation(
                    file_path="logs/checkout.log",
                    line_number=6,
                    snippet="Connection pool exhausted for redis-cache. Available: 0, Max: 5",
                    verified=True,
                )
            ],
        ),
        timeline_summary=["10:00 init", "10:05 spike"],
        service_impacts=["checkout-service 504s"],
        alternative_hypotheses=[],
        uncertainties_and_gaps=["No network packet capture available"],
        verified_evidence=[
            EvidenceCitation(
                file_path="logs/checkout.log",
                line_number=6,
                snippet="Connection pool exhausted for redis-cache. Available: 0, Max: 5",
                verified=True,
            )
        ],
    )

    rubric = evaluate_rubric(
        report=report,
        expected_keywords=["redis", "pool", "exhaust", "connection"],
        total_tool_calls=3,
        budget_exhausted=False,
    )

    assert rubric.correctness > 0.5
    assert rubric.evidence_grounding == 1.0  # 100% verified citations
    assert rubric.completeness == 0.75  # 3 of 4 sections present (no alternative hypotheses)
    assert rubric.safe_handling == 1.0
    assert 0.0 <= rubric.overall_score <= 1.0


@pytest.mark.asyncio
async def test_single_agent_baseline_execution_and_citation_check(tmp_path: Path):
    """Verify that SingleAgentBaseline runs in one pass and checks citations physically."""
    tracer = ExecutionTracer(trace_file=tmp_path / "baseline_trace.jsonl")

    # Mock single report output with 1 valid citation and 1 invalid citation
    mock_report = InvestigationReport(
        incident_title="JWKS Cache OOM",
        executive_summary="JWKS token cache expansion caused JVM heap exhaustion.",
        root_cause="JVM heap memory limit exceeded by JWKS signature cache.",
        confidence_level="High",
        primary_hypothesis=Hypothesis(
            id="hyp-1",
            title="JWKS Cache Heap Exhaustion",
            mechanism="Heap exhausted by 85000 cache entries causing OOMKilled exit code 137.",
            likelihood="high",
            confidence_score=0.95,
            supporting_evidence=[
                EvidenceCitation(
                    file_path="logs/auth.log",
                    line_number=5,
                    snippet="java.lang.OutOfMemoryError: Java heap space during JWKS signature cache expansion",
                ),
                EvidenceCitation(
                    file_path="logs/auth.log",
                    line_number=999,  # Out of range line
                    snippet="Non-existent line snippet",
                ),
            ],
        ),
        timeline_summary=["08:00 started", "08:14 OOM"],
        service_impacts=["auth-service HTTP 503"],
        verified_evidence=[
            EvidenceCitation(
                file_path="logs/auth.log",
                line_number=5,
                snippet="java.lang.OutOfMemoryError: Java heap space during JWKS signature cache expansion",
            ),
        ],
        uncertainties_and_gaps=["Need JVM heap dump analysis"],
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content=mock_report.model_dump_json()))],
            usage=MagicMock(prompt_tokens=400, completion_tokens=250),
        )
    )

    baseline = SingleAgentBaseline(
        incident_description="Auth service OOM crashes",
        evidence_dir=Path("sample_incident_auth"),
        tracer=tracer,
        groq_client=mock_client,
    )

    report = await baseline.run()

    # Model was called exactly once in monolithic pass
    assert mock_client.chat.completions.create.call_count == 1

    # Out of range citation was rejected and only valid citation kept in verified_evidence
    assert len(report.verified_evidence) == 1
    assert report.verified_evidence[0].verified is True
    assert report.verified_evidence[0].line_number == 5
    assert any("failed physical line verification" in s for s in report.failed_steps)


@pytest.mark.asyncio
async def test_run_evaluation_suite_offline_with_mock_client(tmp_path: Path):
    """Verify that run_evaluation_suite runs both approaches across all tasks offline."""
    mock_report = InvestigationReport(
        incident_title="Offline Evaluation Incident",
        executive_summary="Root cause identified as database deadlock.",
        root_cause="Postgres transaction deadlock detected on orders table.",
        confidence_level="Medium",
        primary_hypothesis=Hypothesis(
            id="hyp-db",
            title="Database Deadlock",
            mechanism="Deadlock detected between inventory and orders table.",
            likelihood="high",
            confidence_score=0.85,
        ),
        timeline_summary=["14:20 start", "14:24 deadlock"],
        service_impacts=["order-service HTTP 500"],
        alternative_hypotheses=[],
        uncertainties_and_gaps=["Migration lock trace needed"],
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content=mock_report.model_dump_json()))],
            usage=MagicMock(prompt_tokens=300, completion_tokens=150),
        )
    )

    eval_out = tmp_path / "eval_results"
    summary = await run_evaluation_suite(
        tasks=[DEFAULT_TASKS[2]],  # Run on db deadlock task
        output_dir=eval_out,
        groq_client=mock_client,
        auto_approve=True,
    )

    assert "runs" in summary
    assert len(summary["runs"]) == 2  # 1 multi_agent run + 1 single_agent run
    approaches = [r["approach"] for r in summary["runs"]]
    assert "multi_agent" in approaches
    assert "single_agent" in approaches

    # Verify summary JSON artifact written to disk
    json_artifact = eval_out / "evaluation_summary.json"
    assert json_artifact.exists()
    saved_data = json.loads(json_artifact.read_text(encoding="utf-8"))
    assert len(saved_data["runs"]) == 2
