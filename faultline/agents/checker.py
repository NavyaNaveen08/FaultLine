"""Checker agent for adversarial challenge and physical citation verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from faultline.agents.base import BaseAgent
from faultline.models import CheckerResult, CheckerReview, EvidenceCitation, Hypothesis
from faultline.tools import read_file_safe


class CheckerAgent(BaseAgent):
    """Adversarially critiques hypotheses and verifies cited evidence against raw sandbox files."""

    def verify_citation_physically(self, citation: EvidenceCitation) -> bool:
        """Physically verify if the cited excerpt exists on the specified 1-based line in the target sandbox file."""
        self.tracer.log_event(
            agent=self.name,
            event_type="verify_citation_check",
            payload={
                "file_path": citation.file_path,
                "line_number": getattr(citation, "line_number", None),
                "snippet": citation.snippet,
            },
        )
        try:
            # 0. Path must be a non-empty relative path (reject absolute paths)
            if not citation.file_path or Path(citation.file_path).is_absolute():
                citation.verified = False
                self.tracer.log_event(
                    agent=self.name,
                    event_type="verify_citation_result",
                    payload={"file_path": citation.file_path, "verified": False, "reason": "Path must be a non-empty relative path"},
                )
                return False

            # 1. Line number must be a valid 1-based integer
            if citation.line_number is None or citation.line_number < 1:
                citation.verified = False
                self.tracer.log_event(
                    agent=self.name,
                    event_type="verify_citation_result",
                    payload={"file_path": citation.file_path, "verified": False, "reason": "Invalid or missing 1-based line number"},
                )
                return False

            # 2. Snippet must be non-empty
            if not citation.snippet or not citation.snippet.strip():
                citation.verified = False
                self.tracer.log_event(
                    agent=self.name,
                    event_type="verify_citation_result",
                    payload={"file_path": citation.file_path, "verified": False, "reason": "Empty snippet"},
                )
                return False

            # 3. Read target file content safely
            content = read_file_safe(self.sandbox_dir, citation.file_path)
            lines = content.splitlines()

            # 4. Reject out-of-range line numbers
            if citation.line_number > len(lines):
                citation.verified = False
                self.tracer.log_event(
                    agent=self.name,
                    event_type="verify_citation_result",
                    payload={
                        "file_path": citation.file_path,
                        "line_number": citation.line_number,
                        "verified": False,
                        "reason": f"Line number {citation.line_number} is out of range (file has {len(lines)} lines)",
                    },
                )
                return False

            # 5. Verify against the specific 1-based line only as an exact literal substring.
            # No whitespace normalization is permitted; different spacing or representation must fail.
            target_line = lines[citation.line_number - 1]
            is_present = bool(citation.snippet and citation.snippet in target_line)

            citation.verified = is_present

            self.tracer.log_event(
                agent=self.name,
                event_type="verify_citation_result",
                payload={
                    "file_path": citation.file_path,
                    "line_number": citation.line_number,
                    "verified": is_present,
                    "reason": "Snippet verified on specified line" if is_present else f"Snippet not found on line {citation.line_number}",
                },
            )
            return is_present
        except Exception as exc:
            citation.verified = False
            self.tracer.log_event(
                agent=self.name,
                event_type="verify_citation_error",
                payload={"file_path": citation.file_path, "error": str(exc)},
            )
            return False

    async def verify_evidence_and_challenge(
        self,
        hypotheses: List[Hypothesis],
    ) -> List[CheckerReview]:
        """Verify all citations physically, then adversarially audit hypothesis claims."""
        # Step 1: Deterministic physical verification of all citations
        verification_summary = []
        for hyp in hypotheses:
            hyp_citations = hyp.supporting_evidence + hyp.contradicting_evidence
            verified_count = 0
            for cit in hyp_citations:
                if self.verify_citation_physically(cit):
                    verified_count += 1
            verification_summary.append({
                "hypothesis_id": hyp.id,
                "total_citations": len(hyp_citations),
                "verified_citations": verified_count,
            })

        # Step 2: Adversarial critique via structured LLM prompt
        hypotheses_payload = [h.model_dump() for h in hypotheses]
        prompt = (
            f"PHYSICAL CITATION VERIFICATION SUMMARY:\n"
            f"```json\n{json.dumps(verification_summary, indent=2)}\n```\n\n"
            f"CANDIDATE HYPOTHESES:\n"
            f"```json\n{json.dumps(hypotheses_payload, indent=2, default=str)}\n```\n\n"
            "TASK: Adversarially challenge each hypothesis.\n"
            "For each hypothesis, produce a CheckerReview:\n"
            "- hypothesis_id: ID matching the candidate hypothesis\n"
            "- is_sound: True only if the causal mechanism is logical and backed by verified citations\n"
            "- evidence_verified: True only if all key supporting citations were physically verified\n"
            "- unsupported_claims: List of claims or causal leaps that lack verified proof\n"
            "- suggested_revisions: Constructive feedback or alternative explanations to consider\n"
            "Be strict. Flag any citation that failed physical verification or makes ungrounded assertions."
        )

        system_prompt = (
            "You are the Checker Agent — an adversarial auditor. Your goal is to find flaws, "
            "ungrounded assumptions, and hallucinated evidence in the proposed hypotheses. "
            "Reject claims that rely on unverified citations or logical fallacies."
        )

        try:
            checker_result = await self.call_llm_structured(
                prompt=prompt,
                response_model=CheckerResult,
                system_prompt=system_prompt,
            )
            return checker_result.reviews
        except Exception as exc:
            # Fallback deterministic review based solely on physical verification
            fallback_reviews = []
            for summary in verification_summary:
                hyp_id = summary["hypothesis_id"]
                all_verified = (
                    summary["total_citations"] > 0
                    and summary["verified_citations"] == summary["total_citations"]
                )
                fallback_reviews.append(
                    CheckerReview(
                        hypothesis_id=hyp_id,
                        is_sound=all_verified,
                        evidence_verified=all_verified,
                        unsupported_claims=(
                            [] if all_verified
                            else [f"{summary['total_citations'] - summary['verified_citations']} citations failed physical verification"]
                        ),
                        suggested_revisions=f"Fallback checker evaluation due to: {exc}",
                    )
                )
            return fallback_reviews

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use verify_evidence_and_challenge directly.")
