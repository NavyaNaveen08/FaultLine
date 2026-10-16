"""Tests for Pydantic data models and validation contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from faultline.models import (
    EvidenceCitation,
    Hypothesis,
    InvestigationPlan,
    InvestigationReport,
    Subtask,
    TaskStatus,
)


def test_evidence_citation_creation() -> None:
    citation = EvidenceCitation(
        file_path="logs/app.log",
        line_number=42,
        snippet="500 Internal Server Error",
    )
    assert citation.file_path == "logs/app.log"
    assert citation.line_number == 42
    assert not citation.verified


def test_investigation_plan_validation() -> None:
    subtask1 = Subtask(
        id="subtask-1",
        target_agent="timeline_specialist",
        objective="Analyze error spikes between 10:00 and 10:10 UTC",
    )
    plan = InvestigationPlan(
        incident_summary="Database connection starvation incident",
        initial_assessment="Likely DB pool issue",
        subtasks=[subtask1],
    )
    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].status == TaskStatus.PENDING


def test_hypothesis_confidence_bounds() -> None:
    # Confidence must be between 0.0 and 1.0
    with pytest.raises(ValidationError):
        Hypothesis(
            id="hyp-1",
            title="Invalid",
            mechanism="Broken",
            likelihood="high",
            confidence_score=1.5,
        )


def test_investigation_report_serialization() -> None:
    primary = Hypothesis(
        id="hyp-1",
        title="DB connection pool exhaustion",
        mechanism="Connection pool size was too small under load",
        likelihood="high",
        confidence_score=0.9,
    )
    report = InvestigationReport(
        incident_title="Outage 2026-10-14",
        executive_summary="Service degraded due to DB pool exhaustion",
        root_cause="DB connection pool size reached limit",
        confidence_level="High",
        primary_hypothesis=primary,
        uncertainties_and_gaps=["No metrics on external gateway latency"],
    )
    data = report.model_dump()
    assert data["primary_hypothesis"]["id"] == "hyp-1"
    assert "No metrics on external gateway latency" in data["uncertainties_and_gaps"]
