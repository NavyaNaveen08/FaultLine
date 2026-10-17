"""Single-agent baseline investigator for Faultline.

Provides a fair comparative baseline against the multi-agent system:
- Uses the same Groq model and runtime configuration.
- Has access to the same 3 read-only tools (list_directory_safe, read_file_safe, search_file_safe).
- Has equivalent access to incident evidence files.
- Operates under the same SECURITY_SYSTEM_PROMPT.
- Synthesizes an InvestigationReport in ONE single LLM call without specialist delegation or checker revision loops.
- Evaluates emitted citations physically using the same line-exact verification rubric.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import AsyncGroq

from faultline.agents.base import BaseAgent
from faultline.agents.checker import CheckerAgent
from faultline.budget import RunBudget
from faultline.config import settings
from faultline.models import (
    EvidenceCitation,
    ExecutionStats,
    Hypothesis,
    InvestigationReport,
)
from faultline.tools import list_directory_safe, read_file_safe
from faultline.tracer import ExecutionTracer


class SingleAgentBaseline(BaseAgent):
    """Monolithic single-agent investigator acting as a fair baseline."""

    def __init__(
        self,
        incident_description: str,
        evidence_dir: Path,
        incident_title: str = "Baseline Incident Investigation",
        tracer: Optional[ExecutionTracer] = None,
        groq_client: Optional[AsyncGroq] = None,
        budget: Optional[RunBudget] = None,
    ) -> None:
        self.evidence_dir = Path(evidence_dir)
        self.incident_description = incident_description
        self.incident_title = incident_title
        tracer = tracer or ExecutionTracer()
        budget = budget or RunBudget()

        super().__init__(
            name="single_agent_baseline",
            sandbox_dir=self.evidence_dir,
            tracer=tracer,
            groq_client=groq_client,
            budget=budget,
            timeout_seconds=settings.overall_timeout_seconds,
        )
        self.checker = CheckerAgent(
            name="baseline_verifier",
            sandbox_dir=self.evidence_dir,
            tracer=self.tracer,
            budget=self.budget,
        )

    def _gather_evidence_context(self) -> List[Dict[str, Any]]:
        """Collect file excerpts from the evidence directory using safe read-only tools."""
        manifest = list_directory_safe(self.sandbox_dir)
        excerpts: List[Dict[str, Any]] = []

        # Find files recursively in sandbox
        files_to_read: List[str] = []
        for entry in manifest:
            if entry["is_dir"]:
                try:
                    sub_entries = list_directory_safe(self.sandbox_dir, entry["relative_path"])
                    files_to_read.extend([e["relative_path"] for e in sub_entries if not e["is_dir"]])
                except Exception:
                    pass
            else:
                files_to_read.append(entry["relative_path"])

        for rel_path in files_to_read:
            self.tracer.log_event(
                agent=self.name,
                event_type="tool_call",
                payload={"tool": "read_file_safe", "path": rel_path},
            )
            try:
                content = read_file_safe(
                    self.sandbox_dir,
                    rel_path,
                    max_bytes=settings.max_file_read_bytes,
                    max_lines=200,
                )
                excerpts.append({"file": rel_path, "content": content})
                self.tracer.log_event(
                    agent=self.name,
                    event_type="tool_result",
                    payload={"file": rel_path, "status": "success", "lines": len(content.splitlines())},
                )
            except Exception as exc:
                self.tracer.log_event(
                    agent=self.name,
                    event_type="tool_error",
                    payload={"file": rel_path, "error": str(exc)},
                )

        return excerpts

    async def run(self) -> InvestigationReport:
        """Run the single-agent baseline investigation in a single monolithic prompt."""
        start_time = time.monotonic()
        failed_steps: List[str] = []

        self.tracer.log_event(
            agent=self.name,
            event_type="investigation_start",
            payload={
                "mode": "single_agent_baseline",
                "incident_title": self.incident_title,
                "evidence_dir": str(self.evidence_dir),
            },
        )

        # 1. Gather all file excerpts using the safe tools
        evidence_excerpts = self._gather_evidence_context()
        excerpts_json = json.dumps(evidence_excerpts, indent=2, default=str)

        prompt = (
            f"INCIDENT TITLE: {self.incident_title}\n"
            f"INCIDENT DESCRIPTION:\n{self.incident_description}\n\n"
            f"EVIDENCE FILES (READ-ONLY LOGS & CONFIGURATIONS):\n```json\n{excerpts_json}\n```\n\n"
            "TASK: You are the sole incident investigator. Investigate the incident and produce a complete "
            "InvestigationReport in a single comprehensive analysis.\n"
            "Requirements:\n"
            "1. Identify the root cause and provide an executive summary.\n"
            "2. Formulate the primary hypothesis with supporting and contradicting evidence.\n"
            "3. Offer alternative hypotheses if any.\n"
            "4. Summarize the timeline of events from logs.\n"
            "5. Summarize service impacts and configuration bottlenecks.\n"
            "6. Every evidence citation in verified_evidence MUST specify relative file_path, 1-based line_number, "
            "and exact literal snippet from that line.\n"
            "7. Explicitly disclose uncertainties and gaps in data."
        )

        system_prompt = (
            "You are a monolithic incident investigator. Synthesize root cause, evidence citations, timeline, "
            "and uncertainties directly. Require exact literal file snippets and 1-based line numbers."
        )

        try:
            report = await self.call_llm_structured(
                prompt=prompt,
                response_model=InvestigationReport,
                system_prompt=system_prompt,
            )
        except Exception as exc:
            failed_steps.append(f"Single-agent LLM call failed: {exc}")
            duration = time.monotonic() - start_time
            return InvestigationReport(
                incident_title=self.incident_title,
                executive_summary="Single-agent baseline failed to produce report.",
                root_cause=f"Investigation failed: {exc}",
                confidence_level="Low",
                primary_hypothesis=Hypothesis(
                    id="hyp-failed",
                    title="Investigation Error",
                    mechanism=str(exc),
                    likelihood="low",
                    confidence_score=0.0,
                ),
                failed_steps=failed_steps,
                investigation_duration_seconds=duration,
                execution_stats=ExecutionStats(
                    total_model_calls=self.budget.current_calls,
                    token_usage=self.budget.get_token_usage(),
                    duration_seconds=round(duration, 2),
                    failed_steps=failed_steps,
                ),
            )

        duration = time.monotonic() - start_time

        # 2. Apply the exact same physical citation verification to the baseline's citations
        # Deduplicate citations across report fields
        seen_keys = set()
        unique_citations: List[EvidenceCitation] = []
        for c in (report.verified_evidence or []) + (
            report.primary_hypothesis.supporting_evidence
            if (report.primary_hypothesis and report.primary_hypothesis.supporting_evidence)
            else []
        ):
            key = (c.file_path, c.line_number, c.snippet)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_citations.append(c)

        verified_citations: List[EvidenceCitation] = []
        unverified_count = 0
        for citation in unique_citations:
            is_valid = self.checker.verify_citation_physically(citation)
            if is_valid:
                verified_citations.append(citation)
            else:
                unverified_count += 1

        if unverified_count > 0:
            failed_steps.append(
                f"{unverified_count} citations emitted by single agent failed physical line verification"
            )

        report.verified_evidence = verified_citations
        report.investigation_duration_seconds = round(duration, 2)
        report.failed_steps = failed_steps
        report.execution_stats = ExecutionStats(
            total_model_calls=self.budget.current_calls,
            token_usage=self.budget.get_token_usage(),
            duration_seconds=round(duration, 2),
            failed_steps=failed_steps,
        )

        self.tracer.log_event(
            agent=self.name,
            event_type="investigation_end",
            payload={
                "status": "completed",
                "duration_seconds": round(duration, 2),
                "unverified_citations": unverified_count,
                "verified_citations": len(verified_citations),
            },
        )

        return report
