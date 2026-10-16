"""JSONL execution tracing module for recording agent steps, tool calls, and outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from faultline.config import settings
from faultline.models import TraceEvent


class ExecutionTracer:
    """Thread-safe and async-friendly JSONL execution tracer."""

    def __init__(self, trace_file: Optional[Path] = None):
        if trace_file is None:
            traces_dir = Path("traces")
            traces_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.trace_file = traces_dir / f"trace_{timestamp}.jsonl"
        else:
            self.trace_file = trace_file
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(self, data: Any) -> Any:
        """Recursively redact any known secrets or API keys from logging, preserving token counts."""
        if isinstance(data, dict):
            sanitized = {}
            for k, v in data.items():
                k_lower = k.lower()
                # Preserve actual token counts reported by provider
                is_token_count = k_lower.endswith("_tokens") or k_lower in (
                    "token_usage", "prompt_tokens", "completion_tokens", "total_tokens", "max_tokens"
                )
                if not is_token_count and any(
                    secret_term in k_lower for secret_term in ["api_key", "secret", "token", "auth", "password"]
                ):
                    sanitized[k] = "[REDACTED]"
                else:
                    sanitized[k] = self._sanitize(v)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize(item) for item in data]
        elif isinstance(data, str):
            # Redact if it matches settings.groq_api_key when key is configured
            if settings.groq_api_key and settings.groq_api_key in data:
                return data.replace(settings.groq_api_key, "[REDACTED_API_KEY]")
            return data
        return data

    def log_event(self, agent: str, event_type: str, payload: Dict[str, Any]) -> None:
        """Write a sanitized event line into the JSONL trace."""
        sanitized_payload = self._sanitize(payload)
        event = TraceEvent(
            agent=agent,
            event_type=event_type,
            payload=sanitized_payload,
        )
        line = json.dumps(event.model_dump(), default=str)
        with open(self.trace_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class TraceInspector:
    """Offline deterministic inspector and auditor of recorded JSONL execution traces.

    LIMITATION & SCOPE:
    The Faultline trace schema records execution audit events and operational metrics:
    tool calls, tool results, citation check verdicts, token consumption, latency, and
    truncated prompt previews.
    
    However, traces do NOT record raw model response choices/JSON payloads or full prompt texts.
    Consequently, true deterministic replay of the agent pipeline (feeding recorded LLM responses
    back into the orchestrator/agents without live model calls) cannot be performed from existing
    traces without fabricating missing model responses.
    
    This component provides deterministic offline event inspection, transition validation, and
    execution auditing without invoking live model APIs.
    """

    def __init__(self, trace_file: Path):
        self.trace_file = Path(trace_file)

    def load_events(self) -> List[TraceEvent]:
        """Load and parse all recorded events from the JSONL trace."""
        if not self.trace_file.exists():
            raise FileNotFoundError(f"Trace file not found: {self.trace_file}")
        events: List[TraceEvent] = []
        with open(self.trace_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    events.append(TraceEvent.model_validate(data))
                except Exception as exc:
                    raise ValueError(f"Corrupt trace event on line {line_num}: {exc}") from exc
        return events

    def inspect(self) -> Dict[str, Any]:
        """Deterministically inspect and summarize recorded events from the trace offline."""
        events = self.load_events()
        agents: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []
        citation_checks: List[Dict[str, Any]] = []
        llm_calls: int = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        has_unknown_tokens = False
        revisions: List[Dict[str, Any]] = []

        for event in events:
            if event.agent not in agents:
                agents.append(event.agent)
            if event.event_type == "tool_call":
                tool_calls.append(event.payload)
            elif event.event_type == "tool_result":
                tool_results.append(event.payload)
            elif event.event_type in ("verify_citation_check", "verify_citation_result"):
                citation_checks.append({"type": event.event_type, **event.payload})
            elif event.event_type == "llm_call_success":
                llm_calls += 1
                pt = event.payload.get("prompt_tokens")
                ct = event.payload.get("completion_tokens")
                if isinstance(pt, int) and isinstance(ct, int):
                    total_prompt_tokens += pt
                    total_completion_tokens += ct
                else:
                    has_unknown_tokens = True
            elif event.event_type in ("revision_attempt", "recheck_outcome", "revision_blocked"):
                revisions.append({"event": event.event_type, **event.payload})

        return {
            "trace_file": str(self.trace_file),
            "total_events": len(events),
            "agents": agents,
            "tool_calls_count": len(tool_calls),
            "tool_calls": tool_calls,
            "citation_checks_count": len(citation_checks),
            "llm_calls_count": llm_calls,
            "inspected": True,
            "replayed_without_model": True,
            "deterministic_pipeline_replay_supported": False,
            "missing_data_for_replay": [
                "raw_model_completion_responses",
                "full_prompt_payloads",
            ],
            "revisions": revisions,
            "token_summary": {
                "prompt_tokens": total_prompt_tokens if not has_unknown_tokens else "unknown",
                "completion_tokens": total_completion_tokens if not has_unknown_tokens else "unknown",
                "total_tokens": (total_prompt_tokens + total_completion_tokens) if not has_unknown_tokens else "unknown",
            },
        }

    def replay(self) -> Dict[str, Any]:
        """Backward-compatible alias for inspect()."""
        return self.inspect()


# Backward-compatible aliases
TraceReplayer = TraceInspector


def inspect_trace(trace_file: Path) -> Dict[str, Any]:
    """Convenience function for deterministic offline trace inspection."""
    inspector = TraceInspector(trace_file)
    return inspector.inspect()


replay_trace = inspect_trace
