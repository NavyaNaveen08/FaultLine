"""Lead Investigator agent.

Responsible for:
1. Formulating structured subtasks for Timeline and Dependency specialists.
2. Synthesizing all findings, checker reviews, and explicit uncertainties into an InvestigationReport.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from faultline.agents.base import BaseAgent
from faultline.models import (
    CheckerReview,
    EvidenceCitation,
    ExecutionStats,
    Hypothesis,
    HypothesisEvaluation,
    InvestigationPlan,
    InvestigationReport,
    SpecialistFinding,
    Subtask,
    TaskStatus,
)


class LeadInvestigator(BaseAgent):
    """Lead Investigator agent coordinating incident decomposition and final report synthesis."""

    async def plan_investigation(
        self,
        incident_description: str,
        file_manifest: List[Dict[str, Any]],
    ) -> InvestigationPlan:
        """Analyze the incident description and file manifest to emit structured subtasks."""
        manifest_summary = json.dumps(file_manifest, indent=2)
        prompt = (
            f"INCIDENT REPORT:\n{incident_description}\n\n"
            f"AVAILABLE INCIDENT FILES:\n```json\n{manifest_summary}\n```\n\n"
            "TASK: Create an investigation plan. Decompose the incident into at least two distinct subtasks:\n"
            "1. target_agent='timeline_specialist': Investigate logs and reconstruct the timeline of events/errors.\n"
            "2. target_agent='dependency_specialist': Investigate configs and service topologies to identify bottlenecks and blast radius."
        )

        system_prompt = (
            "You are the Lead Incident Commander. Formulate precise, focused investigation subtasks. "
            "Do not jump to conclusions before evidence is gathered."
        )

        try:
            return await self.call_llm_structured(
                prompt=prompt,
                response_model=InvestigationPlan,
                system_prompt=system_prompt,
            )
        except Exception as exc:
            # Deterministic fallback plan if LLM call fails
            return InvestigationPlan(
                incident_summary=incident_description[:120],
                initial_assessment="Automatic fallback plan generated due to planning failure.",
                subtasks=[
                    Subtask(
                        id="task-timeline",
                        target_agent="timeline_specialist",
                        objective="Reconstruct chronological sequence of errors from log files.",
                        status=TaskStatus.PENDING,
                    ),
                    Subtask(
                        id="task-dependency",
                        target_agent="dependency_specialist",
                        objective="Analyze service configurations and connection limits.",
                        status=TaskStatus.PENDING,
                    ),
                ],
            )

    async def synthesize_report(
        self,
        incident_title: str,
        incident_description: str,
        timeline_finding: Optional[SpecialistFinding],
        dependency_finding: Optional[SpecialistFinding],
        hypothesis_evaluation: Optional[HypothesisEvaluation],
        checker_reviews: List[CheckerReview],
        failed_steps: List[str],
        duration_seconds: float,
        execution_stats: Optional[ExecutionStats] = None,
    ) -> InvestigationReport:
        """Synthesize all specialist findings and checker critiques into the final structured report."""
        context_data = {
            "incident_description": incident_description,
            "timeline_finding": timeline_finding.model_dump() if timeline_finding else None,
            "dependency_finding": dependency_finding.model_dump() if dependency_finding else None,
            "hypothesis_evaluation": hypothesis_evaluation.model_dump() if hypothesis_evaluation else None,
            "checker_reviews": [r.model_dump() for r in checker_reviews],
            "failed_steps": failed_steps,
        }

        prompt = (
            f"INCIDENT TITLE: {incident_title}\n\n"
            f"INVESTIGATION EVIDENCE & FINDINGS:\n```json\n{json.dumps(context_data, indent=2, default=str)}\n```\n\n"
            "TASK: Synthesize the final InvestigationReport.\n"
            "- Select the most plausible root cause as primary_hypothesis.\n"
            "- Include alternative hypotheses.\n"
            "- Consolidate verified timeline items and service impacts.\n"
            "- Explicitly record any uncertainties, missing data, and failed investigation steps in uncertainties_and_gaps.\n"
            "- Provide a clear executive summary and checker verdict."
        )

        system_prompt = (
            "You are the Lead Investigator synthesizing the final incident report. "
            "Be rigorously honest about uncertainty and gaps in the evidence. "
            "Never invent facts that are not present in the specialist findings or checker reviews."
        )

        try:
            report = await self.call_llm_structured(
                prompt=prompt,
                response_model=InvestigationReport,
                system_prompt=system_prompt,
            )
            report.failed_steps = failed_steps
            for step in failed_steps:
                if not any(step in u for u in report.uncertainties_and_gaps):
                    report.uncertainties_and_gaps.append(f"Failed step: {step}")
            # Ensure unverified claims/failures cannot be presented as fully verified
            if failed_steps:
                if report.confidence_level.lower() == "high":
                    report.confidence_level = "Medium" if len(failed_steps) <= 1 else "Low"
                if not report.checker_verdict or "verified" in report.checker_verdict.lower():
                    report.checker_verdict = "Partial: Some claims or citations remained unverified."
            report.verified_evidence = [e for e in report.verified_evidence if getattr(e, "verified", False)]
            report.investigation_duration_seconds = round(duration_seconds, 2)
            report.execution_stats = execution_stats
            return report
        except Exception as exc:
            # Deterministic fallback report preserving all partial results
            primary = (
                hypothesis_evaluation.hypotheses[0]
                if hypothesis_evaluation and hypothesis_evaluation.hypotheses
                else Hypothesis(
                    id="hyp-fallback",
                    title="Inconclusive Investigation",
                    mechanism="Investigation ended with partial results or errors.",
                    likelihood="low",
                    confidence_score=0.2,
                )
            )
            alternatives = (
                hypothesis_evaluation.hypotheses[1:]
                if hypothesis_evaluation and len(hypothesis_evaluation.hypotheses) > 1
                else []
            )

            all_evidence = []
            if timeline_finding:
                all_evidence.extend(timeline_finding.evidence)
            if dependency_finding:
                all_evidence.extend(dependency_finding.evidence)

            uncertainties = list(failed_steps)
            uncertainties.append(f"Report synthesized via fallback due to: {exc}")

            return InvestigationReport(
                incident_title=incident_title,
                executive_summary="Incident investigation completed with partial results.",
                root_cause=primary.title,
                confidence_level="Low" if failed_steps else "Medium",
                primary_hypothesis=primary,
                alternative_hypotheses=alternatives,
                timeline_summary=timeline_finding.key_observations if timeline_finding else [],
                service_impacts=dependency_finding.key_observations if dependency_finding else [],
                verified_evidence=[e for e in all_evidence if e.verified],
                uncertainties_and_gaps=uncertainties,
                checker_verdict="Adversarial review completed" if checker_reviews else "Incomplete review",
                failed_steps=failed_steps,
                investigation_duration_seconds=round(duration_seconds, 2),
                execution_stats=execution_stats,
            )

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use plan_investigation or synthesize_report directly.")
