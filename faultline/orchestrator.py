"""Asynchronous multi-agent incident investigation orchestrator.

Coordinates Lead, Timeline, Dependency, Hypothesis, and Checker agents,
enforcing global deadlines, call/token budgets, parallel execution,
and graceful degradation upon partial failures.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, List, Optional

from groq import AsyncGroq

from faultline.agents.checker import CheckerAgent
from faultline.agents.dependency import DependencySpecialist
from faultline.agents.hypothesis import HypothesisSpecialist
from faultline.agents.lead import LeadInvestigator
from faultline.agents.timeline import TimelineSpecialist
from faultline.budget import BudgetExceededError, GlobalTimeoutError, RunBudget
from faultline.config import settings
from faultline.models import (
    CheckerReview,
    ExecutionStats,
    HypothesisEvaluation,
    InvestigationReport,
    SpecialistFinding,
    Subtask,
    TaskStatus,
)
from faultline.tools import list_directory_safe
from faultline.tracer import ExecutionTracer


class IncidentOrchestrator:
    """Manages the lifecycle of an end-to-end incident investigation."""

    def __init__(
        self,
        incident_description: str,
        evidence_dir: Path,
        incident_title: str = "Incident Investigation",
        tracer: Optional[ExecutionTracer] = None,
        groq_client: Optional[AsyncGroq] = None,
        budget: Optional[RunBudget] = None,
    ) -> None:
        self.incident_description = incident_description
        self.evidence_dir = evidence_dir
        self.incident_title = incident_title
        self.tracer = tracer or ExecutionTracer()
        self.groq_client = groq_client
        self.budget = budget or RunBudget()

        # Instantiate agents with shared budget, sandbox, and tracer
        self.lead = LeadInvestigator(
            name="lead_investigator",
            sandbox_dir=self.evidence_dir,
            tracer=self.tracer,
            groq_client=self.groq_client,
            budget=self.budget,
        )
        self.timeline = TimelineSpecialist(
            name="timeline_specialist",
            sandbox_dir=self.evidence_dir,
            tracer=self.tracer,
            groq_client=self.groq_client,
            budget=self.budget,
        )
        self.dependency = DependencySpecialist(
            name="dependency_specialist",
            sandbox_dir=self.evidence_dir,
            tracer=self.tracer,
            groq_client=self.groq_client,
            budget=self.budget,
        )
        self.hypothesis = HypothesisSpecialist(
            name="hypothesis_specialist",
            sandbox_dir=self.evidence_dir,
            tracer=self.tracer,
            groq_client=self.groq_client,
            budget=self.budget,
        )
        self.checker = CheckerAgent(
            name="checker_agent",
            sandbox_dir=self.evidence_dir,
            tracer=self.tracer,
            groq_client=self.groq_client,
            budget=self.budget,
        )

    def _get_file_manifest(self) -> List[dict]:
        """Obtain safe manifest of available files in evidence directory."""
        try:
            return list_directory_safe(self.evidence_dir)
        except Exception:
            return []

    async def run_investigation(self) -> InvestigationReport:
        """Run the end-to-end multi-agent investigation pipeline."""
        start_time = time.monotonic()
        failed_steps: List[str] = []

        self.tracer.log_event(
            agent="orchestrator",
            event_type="investigation_start",
            payload={
                "incident_title": self.incident_title,
                "evidence_dir": str(self.evidence_dir),
                "budget_calls": self.budget.max_calls,
                "budget_timeout": self.budget.overall_timeout,
            },
        )

        # Step 1: Lead Planning
        manifest = self._get_file_manifest()
        try:
            plan = await self.lead.plan_investigation(self.incident_description, manifest)
        except Exception as e:
            failed_steps.append(f"Lead planning failed: {e}")
            plan = None

        # Resolve subtasks or fallbacks
        timeline_subtask = Subtask(
            id="subtask-timeline",
            target_agent="timeline_specialist",
            objective="Reconstruct chronological sequence of errors from log files.",
        )
        dependency_subtask = Subtask(
            id="subtask-dependency",
            target_agent="dependency_specialist",
            objective="Map service configurations, connection pools, and blast radius.",
        )
        if plan and plan.subtasks:
            for st in plan.subtasks:
                if "timeline" in st.target_agent.lower():
                    timeline_subtask = st
                elif "dependency" in st.target_agent.lower():
                    dependency_subtask = st

        # Step 2: Parallel Specialist Execution (Timeline + Dependency)
        async def _run_specialist(agent: Any, subtask: Subtask, agent_label: str) -> SpecialistFinding:
            try:
                subtask.status = TaskStatus.IN_PROGRESS
                finding = await agent.run(subtask)
                subtask.status = TaskStatus.COMPLETED
                return finding
            except asyncio.TimeoutError:
                subtask.status = TaskStatus.TIMED_OUT
                subtask.error_message = "Agent execution timed out"
                failed_steps.append(f"{agent_label} timed out")
                return SpecialistFinding(
                    agent_name=agent_label,
                    summary=f"{agent_label} timed out during log/config analysis.",
                    failed=True,
                    failure_reason="Timed out",
                )
            except Exception as exc:
                subtask.status = TaskStatus.FAILED
                subtask.error_message = str(exc)
                failed_steps.append(f"{agent_label} failed: {exc}")
                return SpecialistFinding(
                    agent_name=agent_label,
                    summary=f"{agent_label} encountered an error.",
                    failed=True,
                    failure_reason=str(exc),
                )

        timeline_finding, dependency_finding = await asyncio.gather(
            _run_specialist(self.timeline, timeline_subtask, "timeline_specialist"),
            _run_specialist(self.dependency, dependency_subtask, "dependency_specialist"),
            return_exceptions=False,
        )

        # Step 3: Hypothesis Formulation
        findings_pool = [timeline_finding, dependency_finding]
        try:
            hypothesis_eval = await self.hypothesis.evaluate_hypotheses(
                self.incident_description,
                findings_pool,
            )
        except Exception as exc:
            failed_steps.append(f"Hypothesis formulation failed: {exc}")
            hypothesis_eval = HypothesisEvaluation(
                hypotheses=[],
                rationale=f"Hypothesis evaluation failed: {exc}",
            )

        # Step 4: Checker Adversarial Review & Bounded Revision Loop (At most ONE revision)
        checker_reviews: List[CheckerReview] = []
        if hypothesis_eval and hypothesis_eval.hypotheses:
            try:
                checker_reviews = await self.checker.verify_evidence_and_challenge(
                    hypothesis_eval.hypotheses
                )
            except Exception as exc:
                failed_steps.append(f"Checker review failed: {exc}")

            # Check if any hypothesis has material unsupported claims or unverified evidence
            needs_revision = any(
                (not r.is_sound or not r.evidence_verified or bool(r.unsupported_claims))
                for r in checker_reviews
            )

            if needs_revision:
                # 1. Record original result and checker critique in trace
                self.tracer.log_event(
                    agent="orchestrator",
                    event_type="checker_original_result",
                    payload={"hypotheses": [h.model_dump() for h in hypothesis_eval.hypotheses]},
                )
                self.tracer.log_event(
                    agent="orchestrator",
                    event_type="checker_critique",
                    payload={"reviews": [r.model_dump() for r in checker_reviews]},
                )

                # 2. Check if revision can proceed within budget limits
                can_revise = True
                budget_block_reason = None
                try:
                    self.budget.check_budget()
                except Exception as b_exc:
                    can_revise = False
                    budget_block_reason = str(b_exc)

                if not can_revise:
                    # Revision blocked by budget limit
                    self.tracer.log_event(
                        agent="orchestrator",
                        event_type="revision_blocked",
                        payload={"reason": budget_block_reason},
                    )
                    failed_steps.append(f"Hypothesis revision blocked by budget limit: {budget_block_reason}")
                    for r in checker_reviews:
                        if not r.is_sound or not r.evidence_verified or r.unsupported_claims:
                            unverified_detail = ", ".join(r.unsupported_claims) if r.unsupported_claims else "Unsound or unverified citations"
                            failed_steps.append(f"Unverified claims in {r.hypothesis_id}: {unverified_detail} (budget exhausted)")
                else:
                    # 3. Allow at most ONE revision attempt
                    self.tracer.log_event(
                        agent="orchestrator",
                        event_type="revision_attempt",
                        payload={"critique_count": len(checker_reviews)},
                    )
                    try:
                        revised_eval = await self.hypothesis.revise_hypotheses(
                            incident_description=self.incident_description,
                            findings=findings_pool,
                            original_evaluation=hypothesis_eval,
                            checker_reviews=checker_reviews,
                        )
                        hypothesis_eval = revised_eval

                        # 4. Recheck the revised result
                        recheck_reviews = await self.checker.verify_evidence_and_challenge(
                            revised_eval.hypotheses
                        )
                        self.tracer.log_event(
                            agent="orchestrator",
                            event_type="recheck_outcome",
                            payload={"reviews": [r.model_dump() for r in recheck_reviews]},
                        )
                        checker_reviews = recheck_reviews

                        # Check if revised result still remains invalid
                        still_invalid = [
                            r for r in recheck_reviews
                            if not r.is_sound or not r.evidence_verified or r.unsupported_claims
                        ]
                        if still_invalid:
                            failed_steps.append("Revised hypothesis remained unverified after checker recheck")
                            for r in still_invalid:
                                details = ", ".join(r.unsupported_claims) if r.unsupported_claims else "Unsound or unverified citations"
                                failed_steps.append(f"Unverified claims in {r.hypothesis_id} after revision: {details}")

                    except Exception as rev_exc:
                        self.tracer.log_event(
                            agent="orchestrator",
                            event_type="revision_failed",
                            payload={"error": str(rev_exc)},
                        )
                        failed_steps.append(f"Hypothesis revision failed: {rev_exc}")
                        for r in checker_reviews:
                            if not r.is_sound or not r.evidence_verified or r.unsupported_claims:
                                details = ", ".join(r.unsupported_claims) if r.unsupported_claims else "Unsound or unverified citations"
                                failed_steps.append(f"Unverified claims in {r.hypothesis_id}: {details}")
        else:
            failed_steps.append("No hypotheses available for Checker agent to review.")

        # Step 5: Final Report Synthesis by Lead Investigator
        total_duration = time.monotonic() - start_time
        exec_stats = ExecutionStats(
            total_model_calls=self.budget.current_calls,
            token_usage=self.budget.get_token_usage(),
            duration_seconds=round(total_duration, 2),
            failed_steps=failed_steps,
        )

        report = await self.lead.synthesize_report(
            incident_title=self.incident_title,
            incident_description=self.incident_description,
            timeline_finding=timeline_finding,
            dependency_finding=dependency_finding,
            hypothesis_evaluation=hypothesis_eval,
            checker_reviews=checker_reviews,
            failed_steps=failed_steps,
            duration_seconds=total_duration,
            execution_stats=exec_stats,
        )

        self.tracer.log_event(
            agent="orchestrator",
            event_type="investigation_end",
            payload={
                "status": "completed",
                "duration_seconds": round(total_duration, 2),
                "failed_steps_count": len(failed_steps),
                "primary_hypothesis": report.primary_hypothesis.title,
            },
        )

        return report
