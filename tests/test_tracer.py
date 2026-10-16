"""Tests for JSONL execution tracing and secret redaction."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from faultline.tracer import ExecutionTracer


def test_tracer_records_valid_jsonl(sample_trace_file: Path) -> None:
    tracer = ExecutionTracer(trace_file=sample_trace_file)
    tracer.log_event(
        agent="lead_investigator",
        event_type="plan_created",
        payload={"subtask_count": 2},
    )
    tracer.log_event(
        agent="timeline_specialist",
        event_type="tool_call",
        payload={"tool": "read_file_safe", "path": "logs/app.log"},
    )

    assert sample_trace_file.exists()
    lines = sample_trace_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    event1 = json.loads(lines[0])
    assert event1["agent"] == "lead_investigator"
    assert event1["payload"]["subtask_count"] == 2

    event2 = json.loads(lines[1])
    assert event2["agent"] == "timeline_specialist"
    assert event2["event_type"] == "tool_call"


def test_tracer_redacts_sensitive_keys(sample_trace_file: Path) -> None:
    tracer = ExecutionTracer(trace_file=sample_trace_file)
    tracer.log_event(
        agent="system",
        event_type="config_check",
        payload={
            "groq_api_key": "gsk_secret_1234567890",
            "auth_token": "super_secret_token",
            "public_param": "visible",
        },
    )

    lines = sample_trace_file.read_text(encoding="utf-8").strip().splitlines()
    event = json.loads(lines[0])
    assert event["payload"]["groq_api_key"] == "[REDACTED]"
    assert event["payload"]["auth_token"] == "[REDACTED]"
    assert event["payload"]["public_param"] == "visible"


def test_offline_trace_replay_without_model_client(sample_trace_file: Path) -> None:
    """Verify that TraceInspector/TraceReplayer audits execution history deterministically offline without invoking any model client."""
    from unittest.mock import patch
    from faultline.tracer import TraceInspector, TraceReplayer, inspect_trace, replay_trace

    tracer = ExecutionTracer(trace_file=sample_trace_file)
    tracer.log_event("orchestrator", "investigation_start", {"title": "Replay Test"})
    tracer.log_event("timeline_specialist", "tool_call", {"tool": "read_file_safe", "path": "logs/app.log"})
    tracer.log_event("timeline_specialist", "tool_result", {"tool": "read_file_safe", "bytes_read": 1024})
    tracer.log_event("timeline_specialist", "llm_call_success", {"prompt_tokens": 100, "completion_tokens": 50})
    tracer.log_event("checker_agent", "verify_citation_check", {"file_path": "logs/app.log", "line_number": 1, "snippet": "Error"})
    tracer.log_event("checker_agent", "verify_citation_result", {"file_path": "logs/app.log", "verified": True})
    tracer.log_event("orchestrator", "revision_attempt", {"critique_count": 1})
    tracer.log_event("orchestrator", "recheck_outcome", {"reviews": []})
    tracer.log_event("orchestrator", "investigation_end", {"status": "completed"})

    # Assert that during offline trace inspection, AsyncGroq is NEVER called
    with patch("groq.AsyncGroq") as mock_groq:
        summary = inspect_trace(sample_trace_file)
        assert mock_groq.call_count == 0

    assert summary["inspected"] is True
    assert summary["replayed_without_model"] is True
    assert summary["total_events"] == 9
    assert summary["agents"] == ["orchestrator", "timeline_specialist", "checker_agent"]
    assert summary["tool_calls_count"] == 1
    assert summary["citation_checks_count"] == 2
    assert summary["llm_calls_count"] == 1
    assert summary["token_summary"]["total_tokens"] == 150
    assert len(summary["revisions"]) == 2

    # Verify explicitly documented limitations regarding full pipeline replay
    assert summary["deterministic_pipeline_replay_supported"] is False
    assert "raw_model_completion_responses" in summary["missing_data_for_replay"]
    assert "full_prompt_payloads" in summary["missing_data_for_replay"]

    # Non-existent trace file handling
    with pytest.raises(FileNotFoundError):
        TraceInspector(sample_trace_file.parent / "non_existent.jsonl").inspect()

