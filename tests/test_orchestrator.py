"""Mocked tests for async orchestration, parallel execution, failure recovery, and budget enforcement."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from faultline.budget import BudgetExceededError, GlobalTimeoutError, RunBudget
from faultline.models import (
    CheckerReview,
    EvidenceCitation,
    Hypothesis,
    HypothesisEvaluation,
    InvestigationPlan,
    InvestigationReport,
    SpecialistFinding,
    Subtask,
)
from faultline.orchestrator import IncidentOrchestrator
from faultline.tracer import ExecutionTracer


@pytest.mark.asyncio
async def test_budget_call_limit_enforcement():
    budget = RunBudget(max_calls=2)
    budget.record_usage(10, 10)
    budget.record_usage(10, 10)
    assert budget.current_calls == 2

    # Third check must raise BudgetExceededError
    with pytest.raises(BudgetExceededError) as exc_info:
        budget.check_budget()
    assert "budget of 2 calls exhausted" in str(exc_info.value)


@pytest.mark.asyncio
async def test_budget_global_deadline_enforcement():
    budget = RunBudget(max_calls=10, overall_timeout=0.01)
    await asyncio.sleep(0.02)

    with pytest.raises(GlobalTimeoutError) as exc_info:
        budget.check_budget()
    assert "Global investigation deadline" in str(exc_info.value)


@pytest.mark.asyncio
async def test_orchestrator_parallel_execution_and_happy_path(sample_incident_dir: Path, tmp_path: Path):
    tracer = ExecutionTracer(trace_file=tmp_path / "trace.jsonl")
    orchestrator = IncidentOrchestrator(
        incident_description="Test DB outage",
        evidence_dir=sample_incident_dir,
        tracer=tracer,
    )

    # Mock agent methods directly to avoid live LLM calls and verify orchestration flow
    plan = InvestigationPlan(
        incident_summary="Test DB outage",
        initial_assessment="Likely DB pool issue",
        subtasks=[
            Subtask(id="t1", target_agent="timeline_specialist", objective="Log errors"),
            Subtask(id="d1", target_agent="dependency_specialist", objective="Config limits"),
        ],
    )
    timeline_finding = SpecialistFinding(
        agent_name="timeline_specialist",
        summary="Errors began at 10:05",
        key_observations=["10:05:22 DB pool connection threshold reached"],
        evidence=[EvidenceCitation(file_path="logs/payment_service.log", line_number=3, snippet="Connection timeout after 5000ms")],
    )
    dependency_finding = SpecialistFinding(
        agent_name="dependency_specialist",
        summary="DB pool size is 10",
        key_observations=["pool_size: 10 in database.yaml"],
        evidence=[EvidenceCitation(file_path="config/database.yaml", line_number=4, snippet="pool_size: 10")],
    )
    hypothesis_eval = HypothesisEvaluation(
        hypotheses=[
            Hypothesis(
                id="hyp-1",
                title="Database Pool Starvation",
                mechanism="Connection pool size 10 was too small for traffic spike.",
                likelihood="high",
                confidence_score=0.9,
                supporting_evidence=[
                    EvidenceCitation(file_path="logs/payment_service.log", line_number=3, snippet="Connection timeout after 5000ms")
                ],
            )
        ],
        rationale="Clear evidence of pool exhaustion.",
    )
    checker_reviews = [
        CheckerReview(
            hypothesis_id="hyp-1",
            is_sound=True,
            evidence_verified=True,
            unsupported_claims=[],
            suggested_revisions="None",
        )
    ]

    orchestrator.lead.plan_investigation = AsyncMock(return_value=plan)
    orchestrator.lead.call_llm_structured = AsyncMock(
        return_value=InvestigationReport(
            incident_title="Test DB outage",
            executive_summary="DB pool exhausted",
            root_cause="Database Pool Starvation",
            confidence_level="High",
            primary_hypothesis=hypothesis_eval.hypotheses[0],
            uncertainties_and_gaps=[],
        )
    )
    orchestrator.timeline.run = AsyncMock(return_value=timeline_finding)
    orchestrator.dependency.run = AsyncMock(return_value=dependency_finding)
    orchestrator.hypothesis.evaluate_hypotheses = AsyncMock(return_value=hypothesis_eval)
    orchestrator.checker.verify_evidence_and_challenge = AsyncMock(return_value=checker_reviews)

    report = await orchestrator.run_investigation()

    assert isinstance(report, InvestigationReport)
    assert report.primary_hypothesis.title == "Database Pool Starvation"
    assert len(report.failed_steps) == 0
    assert report.execution_stats is not None

    # Verify both specialists were called
    assert orchestrator.timeline.run.called
    assert orchestrator.dependency.run.called


@pytest.mark.asyncio
async def test_orchestrator_handles_failing_specialist(sample_incident_dir: Path, tmp_path: Path):
    tracer = ExecutionTracer(trace_file=tmp_path / "trace.jsonl")
    orchestrator = IncidentOrchestrator(
        incident_description="Test incident with specialist failure",
        evidence_dir=sample_incident_dir,
        tracer=tracer,
    )

    # Timeline specialist fails with an unexpected exception
    orchestrator.timeline.run = AsyncMock(side_effect=RuntimeError("Disk read failure on log volume"))

    # Dependency specialist succeeds
    dependency_finding = SpecialistFinding(
        agent_name="dependency_specialist",
        summary="Config inspected",
        key_observations=["Database timeout is 5000ms"],
    )
    orchestrator.dependency.run = AsyncMock(return_value=dependency_finding)

    # Lead planning succeeds
    orchestrator.lead.plan_investigation = AsyncMock(
        return_value=InvestigationPlan(
            incident_summary="Incident",
            initial_assessment="Assessment",
            subtasks=[],
        )
    )

    # Mock lead synthesis LLM call to guarantee 0 live API calls during unit test
    orchestrator.lead.call_llm_structured = AsyncMock(
        return_value=InvestigationReport(
            incident_title="Test incident with specialist failure",
            executive_summary="Summary with partial results",
            root_cause="Config misconfiguration",
            confidence_level="Medium",
            primary_hypothesis=Hypothesis(
                id="hyp-1",
                title="Config misconfiguration",
                mechanism="Timeout too aggressive",
                likelihood="medium",
                confidence_score=0.5,
            ),
            uncertainties_and_gaps=["Initial uncertainty"],
        )
    )

    # Hypothesis evaluates with available findings
    orchestrator.hypothesis.evaluate_hypotheses = AsyncMock(
        return_value=HypothesisEvaluation(
            hypotheses=[
                Hypothesis(
                    id="hyp-1",
                    title="Config misconfiguration",
                    mechanism="Timeout too aggressive",
                    likelihood="medium",
                    confidence_score=0.5,
                )
            ],
            rationale="Based only on dependency config.",
        )
    )

    orchestrator.checker.verify_evidence_and_challenge = AsyncMock(return_value=[])

    # Run investigation: must not crash!
    report = await orchestrator.run_investigation()

    assert isinstance(report, InvestigationReport)
    # The failed step must be explicitly recorded
    assert any("timeline_specialist failed" in step for step in report.failed_steps)
    assert any("timeline_specialist failed" in u for u in report.uncertainties_and_gaps)


@pytest.mark.asyncio
async def test_orchestrator_handles_timed_out_specialist(sample_incident_dir: Path, tmp_path: Path):
    tracer = ExecutionTracer(trace_file=tmp_path / "trace.jsonl")
    orchestrator = IncidentOrchestrator(
        incident_description="Test incident with specialist timeout",
        evidence_dir=sample_incident_dir,
        tracer=tracer,
    )

    # Dependency specialist raises asyncio.TimeoutError
    orchestrator.dependency.run = AsyncMock(side_effect=asyncio.TimeoutError())

    # Timeline specialist succeeds
    orchestrator.timeline.run = AsyncMock(
        return_value=SpecialistFinding(
            agent_name="timeline_specialist",
            summary="Events found",
            key_observations=["10:00:00 start"],
        )
    )

    orchestrator.lead.plan_investigation = AsyncMock(
        return_value=InvestigationPlan(
            incident_summary="Incident",
            initial_assessment="Assessment",
            subtasks=[],
        )
    )

    orchestrator.lead.call_llm_structured = AsyncMock(
        return_value=InvestigationReport(
            incident_title="Test incident with specialist timeout",
            executive_summary="Summary with timeout",
            root_cause="Unknown",
            confidence_level="Low",
            primary_hypothesis=Hypothesis(
                id="hyp-1",
                title="Unknown root cause",
                mechanism="Lack of dependency data",
                likelihood="low",
                confidence_score=0.3,
            ),
            uncertainties_and_gaps=[],
        )
    )

    orchestrator.hypothesis.evaluate_hypotheses = AsyncMock(
        return_value=HypothesisEvaluation(
            hypotheses=[
                Hypothesis(
                    id="hyp-1",
                    title="Unknown root cause",
                    mechanism="Lack of dependency data",
                    likelihood="low",
                    confidence_score=0.3,
                )
            ],
            rationale="Partial investigation",
        )
    )

    orchestrator.checker.verify_evidence_and_challenge = AsyncMock(return_value=[])

    report = await orchestrator.run_investigation()

    assert isinstance(report, InvestigationReport)
    assert any("dependency_specialist timed out" in step for step in report.failed_steps)


@pytest.mark.asyncio
async def test_checker_revision_loop_success(sample_incident_dir: Path, tmp_path: Path):
    """Test that a material critique triggers exactly one revision attempt and succeeds."""
    trace_file = tmp_path / "trace_revision_success.jsonl"
    tracer = ExecutionTracer(trace_file=trace_file)
    orchestrator = IncidentOrchestrator(
        incident_description="Outage with unverified hypothesis",
        evidence_dir=sample_incident_dir,
        tracer=tracer,
    )

    # 1. Specialists succeed
    orchestrator.lead.plan_investigation = AsyncMock(
        return_value=InvestigationPlan(
            incident_summary="Outage",
            initial_assessment="Initial",
            subtasks=[],
        )
    )
    orchestrator.timeline.run = AsyncMock(
        return_value=SpecialistFinding(
            agent_name="timeline_specialist",
            summary="Timeline logs",
            evidence=[EvidenceCitation(file_path="logs/payment_service.log", line_number=3, snippet="Connection timeout after 5000ms")],
        )
    )
    orchestrator.dependency.run = AsyncMock(
        return_value=SpecialistFinding(
            agent_name="dependency_specialist",
            summary="Topology",
            evidence=[EvidenceCitation(file_path="config/database.yaml", line_number=4, snippet="pool_size: 10")],
        )
    )

    # 2. Initial hypothesis contains ungrounded claim
    hyp_initial = Hypothesis(
        id="hyp-1",
        title="Unverified hypothesis",
        mechanism="Ungrounded claim",
        likelihood="high",
        confidence_score=0.9,
    )
    orchestrator.hypothesis.evaluate_hypotheses = AsyncMock(
        return_value=HypothesisEvaluation(hypotheses=[hyp_initial], rationale="Initial guess")
    )

    # 3. Checker first review: material critique (is_sound=False, unsupported claims)
    review_initial = CheckerReview(
        hypothesis_id="hyp-1",
        is_sound=False,
        evidence_verified=False,
        unsupported_claims=["Mechanism is missing verified causal chain"],
        suggested_revisions="Anchor mechanism in database pool exhaustion",
    )

    # 4. Revised hypothesis produced by revision step
    hyp_revised = Hypothesis(
        id="hyp-1",
        title="Database Pool Starvation",
        mechanism="Verified connection pool exhaustion",
        likelihood="high",
        confidence_score=0.85,
        supporting_evidence=[
            EvidenceCitation(file_path="logs/payment_service.log", line_number=3, snippet="Connection timeout after 5000ms")
        ],
    )
    revised_eval = HypothesisEvaluation(hypotheses=[hyp_revised], rationale="Grounded in evidence")
    orchestrator.hypothesis.revise_hypotheses = AsyncMock(return_value=revised_eval)

    # 5. Checker recheck review: sound and verified!
    review_recheck = CheckerReview(
        hypothesis_id="hyp-1",
        is_sound=True,
        evidence_verified=True,
        unsupported_claims=[],
        suggested_revisions=None,
    )

    orchestrator.checker.verify_evidence_and_challenge = AsyncMock(
        side_effect=[[review_initial], [review_recheck]]
    )

    orchestrator.lead.call_llm_structured = AsyncMock(
        return_value=InvestigationReport(
            incident_title="Outage",
            executive_summary="Resolved via revision",
            root_cause="Database Pool Starvation",
            confidence_level="High",
            primary_hypothesis=hyp_revised,
            uncertainties_and_gaps=[],
        )
    )

    report = await orchestrator.run_investigation()

    # Assert revision was invoked exactly once
    assert orchestrator.hypothesis.revise_hypotheses.call_count == 1
    assert orchestrator.checker.verify_evidence_and_challenge.call_count == 2
    assert report.primary_hypothesis.title == "Database Pool Starvation"

    # Verify trace contains the revision lifecycle events
    lines = [line for line in trace_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    events = [line for line in lines]
    assert any("checker_original_result" in line for line in events)
    assert any("checker_critique" in line for line in events)
    assert any("revision_attempt" in line for line in events)
    assert any("recheck_outcome" in line for line in events)


@pytest.mark.asyncio
async def test_checker_revision_loop_failed_recheck(sample_incident_dir: Path, tmp_path: Path):
    """Test that if a revision fails recheck, exactly one revision is attempted and partial results are labeled."""
    trace_file = tmp_path / "trace_revision_failed.jsonl"
    tracer = ExecutionTracer(trace_file=trace_file)
    orchestrator = IncidentOrchestrator(
        incident_description="Outage where revision fails recheck",
        evidence_dir=sample_incident_dir,
        tracer=tracer,
    )

    orchestrator.lead.plan_investigation = AsyncMock(
        return_value=InvestigationPlan(incident_summary="Outage", initial_assessment="Init", subtasks=[])
    )
    orchestrator.timeline.run = AsyncMock(
        return_value=SpecialistFinding(agent_name="timeline_specialist", summary="Timeline")
    )
    orchestrator.dependency.run = AsyncMock(
        return_value=SpecialistFinding(agent_name="dependency_specialist", summary="Dependency")
    )

    hyp_initial = Hypothesis(
        id="hyp-1",
        title="Flawed hypothesis",
        mechanism="Speculation",
        likelihood="medium",
        confidence_score=0.4,
    )
    orchestrator.hypothesis.evaluate_hypotheses = AsyncMock(
        return_value=HypothesisEvaluation(hypotheses=[hyp_initial], rationale="Initial")
    )

    # First critique
    review_1 = CheckerReview(
        hypothesis_id="hyp-1",
        is_sound=False,
        evidence_verified=False,
        unsupported_claims=["Speculative claim 1"],
    )

    # Revision still produces a flawed hypothesis
    hyp_still_flawed = Hypothesis(
        id="hyp-1",
        title="Still flawed hypothesis",
        mechanism="Still speculative",
        likelihood="low",
        confidence_score=0.3,
    )
    orchestrator.hypothesis.revise_hypotheses = AsyncMock(
        return_value=HypothesisEvaluation(hypotheses=[hyp_still_flawed], rationale="Revised guess")
    )

    # Recheck still fails
    review_2 = CheckerReview(
        hypothesis_id="hyp-1",
        is_sound=False,
        evidence_verified=False,
        unsupported_claims=["Still speculative and ungrounded"],
    )

    orchestrator.checker.verify_evidence_and_challenge = AsyncMock(
        side_effect=[[review_1], [review_2]]
    )

    orchestrator.lead.call_llm_structured = AsyncMock(
        return_value=InvestigationReport(
            incident_title="Outage",
            executive_summary="Completed with partial unverified results",
            root_cause="Still flawed hypothesis",
            confidence_level="Low",
            primary_hypothesis=hyp_still_flawed,
            uncertainties_and_gaps=[],
        )
    )

    report = await orchestrator.run_investigation()

    # Bounded to exactly ONE revision attempt
    assert orchestrator.hypothesis.revise_hypotheses.call_count == 1
    assert orchestrator.checker.verify_evidence_and_challenge.call_count == 2
    # Partial result must explain what remains unverified
    assert any("Revised hypothesis remained unverified" in step for step in report.failed_steps)
    assert any("Unverified claims in hyp-1" in step for step in report.failed_steps)


@pytest.mark.asyncio
async def test_checker_revision_loop_budget_blocked(sample_incident_dir: Path, tmp_path: Path):
    """Test that when the budget is exhausted, revision is blocked and labeled as partial."""
    trace_file = tmp_path / "trace_revision_budget_blocked.jsonl"
    tracer = ExecutionTracer(trace_file=trace_file)
    budget = RunBudget(max_calls=0)  # Call budget already exhausted

    orchestrator = IncidentOrchestrator(
        incident_description="Outage with budget block",
        evidence_dir=sample_incident_dir,
        tracer=tracer,
        budget=budget,
    )

    orchestrator.lead.plan_investigation = AsyncMock(
        return_value=InvestigationPlan(incident_summary="Outage", initial_assessment="Init", subtasks=[])
    )
    orchestrator.timeline.run = AsyncMock(
        return_value=SpecialistFinding(agent_name="timeline_specialist", summary="Timeline")
    )
    orchestrator.dependency.run = AsyncMock(
        return_value=SpecialistFinding(agent_name="dependency_specialist", summary="Dependency")
    )

    hyp_initial = Hypothesis(
        id="hyp-1",
        title="Unverified hypothesis",
        mechanism="Ungrounded",
        likelihood="high",
        confidence_score=0.8,
    )
    orchestrator.hypothesis.evaluate_hypotheses = AsyncMock(
        return_value=HypothesisEvaluation(hypotheses=[hyp_initial], rationale="Initial")
    )
    orchestrator.hypothesis.revise_hypotheses = AsyncMock()

    # Checker finds issue
    critique = CheckerReview(
        hypothesis_id="hyp-1",
        is_sound=False,
        evidence_verified=False,
        unsupported_claims=["Causal link unverified"],
    )
    orchestrator.checker.verify_evidence_and_challenge = AsyncMock(return_value=[critique])

    orchestrator.lead.call_llm_structured = AsyncMock(
        return_value=InvestigationReport(
            incident_title="Outage",
            executive_summary="Budget blocked report",
            root_cause="Unverified hypothesis",
            confidence_level="Low",
            primary_hypothesis=hyp_initial,
            uncertainties_and_gaps=[],
        )
    )

    report = await orchestrator.run_investigation()

    # Revision must NOT be called when budget is exhausted
    assert orchestrator.hypothesis.revise_hypotheses.call_count == 0
    # Must report that revision was blocked by budget limit
    assert any("Hypothesis revision blocked by budget limit" in step for step in report.failed_steps)
    assert any("budget exhausted" in step for step in report.failed_steps)

    # Trace must record revision_blocked event
    lines = [line for line in trace_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any("revision_blocked" in line for line in lines)
