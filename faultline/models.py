"""Pydantic data models defining the communication contracts across agents."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class EvidenceCitation(BaseModel):
    """Structured evidence pointing to a verified line or excerpt in synthetic logs."""

    file_path: str = Field(description="Relative path of file within the incident directory")
    line_number: int = Field(ge=1, description="Real 1-based line number in target file")
    snippet: str = Field(description="Exact snippet or key log line cited")
    verified: bool = Field(default=False, description="Whether the checker verified this excerpt")


class Subtask(BaseModel):
    """Subtask created by the Lead Investigator for a specialist agent."""

    id: str
    target_agent: str
    objective: str
    status: TaskStatus = TaskStatus.PENDING
    error_message: Optional[str] = None


class InvestigationPlan(BaseModel):
    """Structured investigation plan created by the Lead Investigator."""

    incident_summary: str
    initial_assessment: str
    subtasks: List[Subtask] = Field(default_factory=list)


class SpecialistFinding(BaseModel):
    """Output produced by a specialist agent (Timeline or Dependency)."""

    agent_name: str
    summary: str
    key_observations: List[str] = Field(default_factory=list)
    evidence: List[EvidenceCitation] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    failed: bool = Field(default=False)
    failure_reason: Optional[str] = None


class Hypothesis(BaseModel):
    """Candidate root cause hypothesis evaluated by the Hypothesis Specialist."""

    id: str
    title: str
    mechanism: str
    likelihood: str = Field(description="high, medium, or low")
    supporting_evidence: List[EvidenceCitation] = Field(default_factory=list)
    contradicting_evidence: List[EvidenceCitation] = Field(default_factory=list)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)


class HypothesisEvaluation(BaseModel):
    """Collection of candidate hypotheses ranked by likelihood."""

    hypotheses: List[Hypothesis] = Field(default_factory=list)
    rationale: str


class CheckerReview(BaseModel):
    """Critique and validation output from the Checker Agent for a single hypothesis."""

    hypothesis_id: str
    is_sound: bool
    evidence_verified: bool
    unsupported_claims: List[str] = Field(default_factory=list)
    suggested_revisions: Optional[str] = None


class CheckerResult(BaseModel):
    """Collection of checker reviews and overall adversarial evaluation."""

    reviews: List[CheckerReview] = Field(default_factory=list)
    overall_assessment: str = ""


class TokenUsage(BaseModel):
    """Token consumption statistics or 'unknown' if unavailable from provider."""

    prompt_tokens: Any = "unknown"
    completion_tokens: Any = "unknown"
    total_tokens: Any = "unknown"


class ExecutionStats(BaseModel):
    """Runtime execution statistics and audit totals."""

    total_model_calls: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    duration_seconds: float = 0.0
    failed_steps: List[str] = Field(default_factory=list)


class InvestigationReport(BaseModel):
    """Final incident investigation report synthesized by the Lead Investigator."""

    incident_title: str
    executive_summary: str
    root_cause: str
    confidence_level: str  # e.g. "High", "Medium", "Low"
    primary_hypothesis: Hypothesis
    alternative_hypotheses: List[Hypothesis] = Field(default_factory=list)
    timeline_summary: List[str] = Field(default_factory=list)
    service_impacts: List[str] = Field(default_factory=list)
    verified_evidence: List[EvidenceCitation] = Field(default_factory=list)
    uncertainties_and_gaps: List[str] = Field(default_factory=list)
    checker_verdict: Optional[str] = None
    failed_steps: List[str] = Field(default_factory=list)
    investigation_duration_seconds: float = 0.0
    execution_stats: Optional[ExecutionStats] = None


class TraceEvent(BaseModel):
    """Single event recorded in JSONL execution traces."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent: str
    event_type: str  # e.g., "start", "thought", "tool_call", "tool_result", "output", "error", "timeout"
    payload: Dict[str, Any] = Field(default_factory=dict)
