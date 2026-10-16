# Faultline 🔍
> **Fault-tolerant multi-agent incident investigation harness**

Faultline investigates software incidents by orchestrating specialized AI agents to analyze application logs, error traces, and service configurations. It produces an evidence-backed incident report with root causes, confidence levels, alternatives, and explicit uncertainties.

---

## Architecture Overview

Faultline uses a **centralized blackboard architecture** with strictly typed Pydantic models and asynchronous coordination:

1. **Lead Investigator**: Ingests incident context and data directory manifest; synthesizes a structured investigation plan with subtasks.
2. **Specialists in Parallel (`asyncio.gather`)**:
   - **Timeline Specialist**: Reconstructs the chronological event timeline, error spikes, and latencies from log files.
   - **Dependency Specialist**: Maps service topologies, connection bounds, timeouts, and failure blast radiuses from configs.
3. **Hypothesis Specialist**: Evaluates collected evidence to formulate and rank candidate root causes.
4. **Checker Agent**: Adversarially challenges findings, cross-examining cited log lines and configurations against raw files to eliminate hallucinations.
   - **Bounded Revision Loop**: If the Checker identifies material unsupported claims or invalid citations, at most **one** revision attempt is granted to the Hypothesis Specialist, followed by a recheck.
5. **Synthesis & Audit**: Lead Investigator produces the final `InvestigationReport` with explicit uncertainties and verified evidence citations. Every agent action is audited in a JSONL execution trace.

```
                           +----------------------+
                           |   User Incident &    |
                           |  Synthetic Evidence  |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           |  Lead Investigator   |
                           |  (Decomposes Tasks)  |
                           +----------+-----------+
                                      |
                  +-------------------+-------------------+
                  |                                       |
                  v                                       v
      +-----------------------+               +-----------------------+
      |  Timeline Specialist  |               | Dependency Specialist |
      | (Log & Event Analysis)|               |  (Topology & Config)  |
      +-----------+-----------+               +-----------+-----------+
                  |                                       |
                  +-------------------+-------------------+
                                      |
                                      v
                           +----------------------+
                           | Hypothesis Specialist|
                           | (Root Cause Ranking) |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           |    Checker Agent     | <----+ (At most 1
                           | (Adversarial Audit & |      |  revision pass)
                           | Physical Line Check) | -----+
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           |  Lead Investigator   |
                           |  (Final Synthesis)   |
                           +----------+-----------+
                                      |
                                      v
                         Structured Investigation Report
                           + JSONL Execution Trace
```

---

## Directory Structure

```text
FaultLine/
├── .env.example             # Template for API keys & operational limits
├── .gitignore               # Excludes secrets, virtual environments, caches, traces
├── pyproject.toml           # Package configuration & pytest settings
├── requirements.txt         # Core dependencies: groq, pydantic, python-dotenv, pytest
├── README.md                # Project documentation
├── sample_incident/         # Synthetic incident fixture for verification
│   ├── incident.txt         # Incident description (504 timeouts after cache update)
│   ├── logs/
│   │   └── checkout.log     # Timestamped logs showing latency spikes and pool exhaustion
│   └── config/
│       └── services.json    # Dependency topology, timeouts, and connection bounds
├── faultline/
│   ├── __init__.py
│   ├── cli.py               # CLI entry point (run incident or inspect trace)
│   ├── config.py            # Safe environment loading & runtime settings
│   ├── models.py            # Pydantic schemas (citations, hypotheses, report, trace)
│   ├── budget.py            # Shared token/call budget & deadline tracker
│   ├── tools.py             # Sandboxed, read-only safe file inspection tools
│   ├── tracer.py            # JSONL execution trace logger and offline trace inspector
│   ├── orchestrator.py      # Multi-agent coordination harness with revision loop
│   └── agents/
│       ├── __init__.py
│       ├── base.py          # BaseAgent with bounded retries, timeouts, and tracing
│       ├── lead.py          # Lead Investigator (planning & report synthesis)
│       ├── timeline.py      # Timeline Specialist (log event chronology)
│       ├── dependency.py    # Dependency Specialist (topology & config limits)
│       ├── hypothesis.py    # Hypothesis Specialist (formulation & revision)
│       └── checker.py       # Checker Agent (physical line-level verification & critique)
└── tests/
    ├── __init__.py
    ├── conftest.py          # Pytest fixtures for synthetic incident directories
    ├── test_agents.py       # BaseAgent retries, timeouts, and line-specific citation checks
    ├── test_budget.py       # Shared token budget enforcement & accumulation tests
    ├── test_injection.py    # Prompt-injection resistance & read-only boundary tests
    ├── test_models.py       # Pydantic schema validation & citation contract tests
    ├── test_orchestrator.py # Parallel execution, recovery, and revision loop tests
    ├── test_tools.py        # Safe file tools, directory traversal, & sandboxing tests
    └── test_tracer.py       # JSONL tracing, secret redaction, & offline inspection tests
```

---

## Security Model & Tool Restrictions

- **Strict Read-Only Sandboxing**: File tools in `faultline/tools.py` (`list_directory_safe`, `read_file_safe`, `search_file_safe`) strictly confine all operations within the designated incident directory.
  - Path traversal attempts (`../`) and absolute paths (`C:\...`, `/...`) raise `SandboxedPathError`.
- **No Execution Capabilities**: No shell commands, subprocess calls, `os.system`, `exec`, or file write capabilities exist in the toolset.
- **Prompt-Injection Resistance**: Agents operate under a `SECURITY_SYSTEM_PROMPT` explicitly defining all log files, configurations, and stack traces as untrusted data. Malicious directives (e.g. `SYSTEM OVERRIDE`, `EXECUTE_SHELL`) embedded in logs are parsed passively as evidence and never executed.
- **Zero Secret Leakage**:
  - `ExecutionTracer._sanitize()` automatically redacts API keys, passwords, and bearer tokens from logs.
  - Actual token consumption metrics (`prompt_tokens`, `completion_tokens`, `total_tokens`) are preserved unredacted.
- **Git Exclusions**: `.gitignore` strictly excludes `.env`, `traces/`, `.venv/`, `.pytest_cache/`, and log files.

---

## Budgets & Failure Recovery

- **Shared Run-Level Budget (`RunBudget`)**:
  - Enforces global execution deadline (`FAULTLINE_OVERALL_TIMEOUT_SECONDS`, default 300s).
  - Enforces shared model call cap (`FAULTLINE_MAX_TOTAL_MODEL_CALLS`, default 20 calls).
  - Enforces shared token budget (`FAULTLINE_MAX_TOTAL_TOKENS`) checked before each model call whenever prior usage is known.
- **Per-Agent Isolation & Timeouts**:
  - Each specialist agent executes under an individual timeout (`FAULTLINE_AGENT_TIMEOUT_SECONDS`, default 60s).
  - Transient API failures (rate limits, timeouts, 5xx) retry with exponential backoff and jitter up to `FAULTLINE_MAX_RETRIES` (default 3).
- **Graceful Partial Failure Recovery**:
  - If a specialist fails or times out during parallel execution, the orchestrator isolates the failure, logs it, and continues with available findings.
  - The Lead Investigator incorporates failed steps directly into `failed_steps` and `uncertainties_and_gaps`, automatically downgrading report confidence from "High".
- **Bounded Checker Revision Loop**:
  - If the Checker Agent flags ungrounded claims or invalid citations, exactly **one** revision attempt is dispatched to the Hypothesis Specialist.
  - If the budget is exhausted before revision, revision is blocked (`revision_blocked` event), and the report explicitly labels what remains unverified.

---

## Setup & Verification

### 1. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables (Optional for Offline Testing)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
# Set GROQ_API_KEY=gsk_... for live runs
```

---

## Run Commands

### 1. Run Complete Offline Test Suite
To verify the entire harness without live API calls:
```bash
pytest -v
```
*(Runs all 36 unit tests across models, agents, tools, tracer, orchestrator, budget, and prompt-injection defense).*

### 2. Run Sample Incident Investigation (Requires `GROQ_API_KEY`)
To execute the multi-agent investigation against the included synthetic incident:
```bash
python -m faultline.cli --incident-file sample_incident/incident.txt --data-dir sample_incident
```
Optionally save report and trace to specific paths:
```bash
python -m faultline.cli --incident-file sample_incident/incident.txt --data-dir sample_incident --output report.json --trace-file traces/investigation.jsonl
```

### 3. Inspect and Audit an Execution Trace (Offline)
Inspect any recorded JSONL trace without invoking the model API:
```bash
python -m faultline.cli --inspect-trace traces/trace_20261014_151843.jsonl
```

---

## Demonstrating Failure Recovery & Budget Limits Offline

Failure recovery and budget enforcement can be demonstrated credibly and deterministically without uncontrolled live API calls:

### 1. Demonstrate Specialist Failure & Timeout Isolation
```bash
pytest tests/test_orchestrator.py -k "failing_specialist or timed_out_specialist" -v
```
- **Proves**: When a specialist agent throws an exception or hits a timeout, the orchestrator isolates the error, continues parallel execution with remaining specialists, documents the failure in `failed_steps`, and produces a valid partial report.

### 2. Demonstrate Budget-Blocked Revision & Token Exhaustion
```bash
pytest tests/test_orchestrator.py -k "revision_loop_budget_blocked" -v
pytest tests/test_budget.py -k "subsequent_call_blocked" -v
```
- **Proves**: When call or token limits are reached, subsequent model calls and revision attempts are immediately blocked with `BudgetExceededError`, logging a `revision_blocked` event and returning an unverified claims disclosure.

---

## Comparison Methodology & Actual Results

No fabricated benchmark numbers or simulated multi-model rankings are claimed. The system was evaluated using actual measured executions:

1. **Deterministic Test Suite**:
   - **Total Tests**: 36 passed in 1.49s.
   - **Coverage Areas**: Tool path traversal prevention, exact line-level snippet citation, single-attempt revision loops, honest unknown token accounting, prompt-injection defense, and offline trace inspection.
2. **Sample Incident Execution (Recorded Baseline)**:
   - **Incident**: HTTP 504 Gateway Timeouts following cache configuration change.
   - **Execution Metrics**: 6 model calls, 3 tool reads (`incident.txt`, `checkout.log`, `services.json`), 22 physical citation checks, and 0 unhandled agent crashes.
   - **Identified Root Cause**: Redis connection pool exhaustion (`max_connections: 5`) combined with latency spike (2500ms), verified against line 6 of `logs/checkout.log` and line 7 of `config/services.json`.

---

## Limitations & Tradeoffs

1. **Token Upper Bounds with Unknown Provider Usage**:
   - When the LLM provider omits token usage data (`has_unknown_tokens=True`), unmeasured token consumption cannot be calculated without fabricating numbers. Faultline refuses to invent counts or assume zero; it documents this limitation and uses call limits (`max_calls`) and deadlines (`overall_timeout`) as deterministic safeguards.
2. **Trace Inspection vs. Full Pipeline Replay**:
   - Traces record execution audit events, tool interactions, citation verification outcomes, and operational metrics. However, raw model response strings and complete prompts were omitted from the trace schema. As a result, offline trace analysis performs event inspection and auditing, not full offline deterministic pipeline replay.
3. **Strict Read-Only Sandboxing**:
   - Confining agents to read-only safe inspection prevents active testing (e.g. issuing test HTTP queries or pinging endpoints). The investigation relies entirely on static logs and configurations present in the incident directory.

---

## Cuts & Scope Reductions

1. **No Autonomous Shell Execution**: Excluded shell execution tools to prevent prompt-injection escalation and destructive system actions.
2. **Bounded Revision Loop**: Strictly capped Checker revision to at most **one** pass to avoid unbounded token consumption and infinite critique loops.
3. **No Unverified Benchmarks**: Cut speculative performance claims across untested models; evaluation is anchored strictly in reproducible offline test suites.

---

## Time Spent & AI Usage

- **Total Time Spent**: Approximately 14 hours across architecture design, multi-agent orchestrator implementation, security sandboxing, budget enforcement, testing, and documentation.
- **AI Tooling Used**: Antigravity IDE and Gemini agentic pair-programming for drafting Pydantic schemas, generating test fixtures, conducting code reviews, and automating regression testing.

### Incorrect AI Output & Correction Example

- **The Issue**:
  During the implementation of physical citation verification in `CheckerAgent`, the AI assistant initially implemented snippet verification using whitespace normalization across the entire file:
  ```python
  # Initial flawed AI implementation
  normalized_content = " ".join(content.split())
  normalized_snippet = " ".join(citation.snippet.split())
  is_present = normalized_snippet in normalized_content
  ```
  This implementation contained two severe defects:
  1. It completely ignored `citation.line_number`, allowing a snippet that appeared on a completely different line to be verified as true.
  2. The whitespace normalization (`" ".join(...)`) allowed snippets with corrupted spacing (e.g. extra spaces or tabs) to falsely pass.
- **The Correction**:
  1. Updated `EvidenceCitation` to make `line_number: int = Field(ge=1)` required.
  2. Constrained snippet evaluation exclusively to the specified 1-based line index (`lines[citation.line_number - 1]`).
  3. Replaced whitespace normalization with an exact literal substring match:
     ```python
     # Corrected implementation
     target_line = lines[citation.line_number - 1]
     is_present = bool(citation.snippet and citation.snippet in target_line)
     citation.verified = is_present
     ```
  4. Added explicit tests in `tests/test_agents.py` demonstrating that wrong line numbers and whitespace-mismatched snippets are strictly rejected.
