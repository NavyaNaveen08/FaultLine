"""Command-line interface (CLI) for Faultline using standard argparse."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from faultline.config import settings
from faultline.orchestrator import IncidentOrchestrator
from faultline.tracer import ExecutionTracer


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the Faultline harness."""
    parser = argparse.ArgumentParser(
        prog="faultline",
        description="Faultline: Fault-tolerant multi-agent incident investigation harness.",
    )
    parser.add_argument(
        "--inspect-trace",
        type=Path,
        default=None,
        help="Path to an existing JSONL execution trace file to inspect and audit offline.",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--incident",
        type=str,
        help="Textual description of the incident.",
    )
    group.add_argument(
        "--incident-file",
        type=Path,
        help="Path to a text file containing the incident description.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing synthetic logs, error traces, and config files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the final investigation report (JSON).",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=None,
        help="Optional path for JSONL execution trace output.",
    )

    parsed = parser.parse_args(args)
    if not parsed.inspect_trace:
        if not parsed.incident and not parsed.incident_file:
            parser.error("one of the arguments --incident --incident-file is required (unless using --inspect-trace)")
        if not parsed.data_dir:
            parser.error("the following arguments are required: --data-dir (unless using --inspect-trace)")
    return parsed


async def _run(args: argparse.Namespace, incident_description: str) -> int:
    tracer = ExecutionTracer(trace_file=args.trace_file) if args.trace_file else ExecutionTracer()
    orchestrator = IncidentOrchestrator(
        incident_description=incident_description,
        evidence_dir=args.data_dir,
        incident_title="CLI Incident Investigation",
        tracer=tracer,
    )

    print("Investigating incident with Faultline multi-agent harness...")
    try:
        report = await orchestrator.run_investigation()
    except Exception as exc:
        sys.stderr.write(f"Investigation execution failed: {exc}\n")
        return 1

    report_json = json.dumps(report.model_dump(), indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json, encoding="utf-8")
        print(f"Report written to: {args.output.resolve()}")
    else:
        print("\n=== INVESTIGATION REPORT ===")
        print(report_json)

    print(f"\nTrace log saved to: {tracer.trace_file.resolve()}")
    return 0


def main() -> None:
    """Main CLI entry point."""
    args = parse_args()

    # Offline trace inspection mode
    if args.inspect_trace:
        from faultline.tracer import inspect_trace
        if not args.inspect_trace.exists():
            sys.stderr.write(f"Error: trace file not found: {args.inspect_trace}\n")
            sys.exit(1)
        try:
            summary = inspect_trace(args.inspect_trace)
            print("\n=== TRACE AUDIT SUMMARY ===")
            print(json.dumps(summary, indent=2, default=str))
            sys.exit(0)
        except Exception as exc:
            sys.stderr.write(f"Error inspecting trace: {exc}\n")
            sys.exit(1)

    # Read incident description
    if args.incident_file:
        if not args.incident_file.exists():
            sys.stderr.write(f"Error: incident file not found: {args.incident_file}\n")
            sys.exit(1)
        incident_description = args.incident_file.read_text(encoding="utf-8")
    else:
        incident_description = args.incident

    # Validate evidence directory
    if not args.data_dir.exists() or not args.data_dir.is_dir():
        sys.stderr.write(f"Error: evidence directory not found: {args.data_dir}\n")
        sys.exit(1)

    # Validate API key
    try:
        settings.validate_api_key()
    except ValueError as e:
        sys.stderr.write(f"Configuration Error: {e}\n")
        sys.exit(1)

    exit_code = asyncio.run(_run(args, incident_description))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
