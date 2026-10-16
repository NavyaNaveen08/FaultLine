"""Hypothesis Specialist agent for candidate root cause evaluation."""

from __future__ import annotations

import json
from typing import Any, List, Optional

from faultline.agents.base import BaseAgent
from faultline.models import HypothesisEvaluation, SpecialistFinding


class HypothesisSpecialist(BaseAgent):
    """Correlates timeline events and dependency topologies to evaluate candidate root causes."""

    async def evaluate_hypotheses(
        self,
        incident_description: str,
        findings: List[SpecialistFinding],
    ) -> HypothesisEvaluation:
        """Formulate, compare, and rank candidate root causes based on specialist evidence."""
        findings_payload = [f.model_dump() for f in findings]

        prompt = (
            f"INCIDENT CONTEXT:\n{incident_description}\n\n"
            f"SPECIALIST FINDINGS & EVIDENCE:\n```json\n{json.dumps(findings_payload, indent=2, default=str)}\n```\n\n"
            "TASK: Evaluate candidate root cause hypotheses.\n"
            "1. Formulate at least two distinct hypotheses explaining why the incident occurred.\n"
            "2. For each hypothesis, supply:\n"
            "   - id: Unique ID (e.g. 'hyp-1')\n"
            "   - title: Clear title of the failure mode\n"
            "   - mechanism: Causal explanation connecting evidence to user-facing symptoms\n"
            "   - likelihood: 'high', 'medium', or 'low'\n"
            "   - supporting_evidence: Evidence citations that support this mechanism\n"
            "   - contradicting_evidence: Evidence citations or counter-indications against this mechanism\n"
            "   - confidence_score: Float between 0.0 and 1.0\n"
            "3. Rank the most likely explanation first and explain the rationale."
        )

        system_prompt = (
            "You are the Hypothesis Specialist. Rigorously evaluate alternative explanations. "
            "Do not favor simple correlation without causal plausibility. "
            "Identify counter-evidence and explicit uncertainties for every hypothesis."
        )

        return await self.call_llm_structured(
            prompt=prompt,
            response_model=HypothesisEvaluation,
            system_prompt=system_prompt,
        )

    async def revise_hypotheses(
        self,
        incident_description: str,
        findings: List[SpecialistFinding],
        original_evaluation: HypothesisEvaluation,
        checker_reviews: List[Any],
    ) -> HypothesisEvaluation:
        """Revise candidate hypotheses based on adversarial critique and citation verification from the Checker."""
        findings_payload = [f.model_dump() for f in findings]
        hypotheses_payload = [h.model_dump() for h in original_evaluation.hypotheses]
        critique_payload = [
            r.model_dump() if hasattr(r, "model_dump") else str(r)
            for r in checker_reviews
        ]

        prompt = (
            f"INCIDENT CONTEXT:\n{incident_description}\n\n"
            f"SPECIALIST FINDINGS & EVIDENCE:\n```json\n{json.dumps(findings_payload, indent=2, default=str)}\n```\n\n"
            f"ORIGINAL HYPOTHESES:\n```json\n{json.dumps(hypotheses_payload, indent=2, default=str)}\n```\n\n"
            f"CHECKER ADVERSARIAL CRITIQUE & UNVERIFIED CLAIMS:\n```json\n{json.dumps(critique_payload, indent=2, default=str)}\n```\n\n"
            "TASK: Revise the hypotheses to resolve the Checker's critiques.\n"
            "- Correct or remove unsupported claims.\n"
            "- Fix or replace invalid or unverified citations with exact snippets and 1-based line numbers from the specialist evidence.\n"
            "- Recalibrate confidence scores honestly.\n"
            "Return a revised HypothesisEvaluation."
        )

        system_prompt = (
            "You are the Hypothesis Specialist performing a single revision cycle. "
            "Rigorously address the Checker's critique, eliminate hallucinated or invalid citations, "
            "and ground every retained claim in verified evidence."
        )

        return await self.call_llm_structured(
            prompt=prompt,
            response_model=HypothesisEvaluation,
            system_prompt=system_prompt,
        )

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use evaluate_hypotheses or revise_hypotheses directly.")
