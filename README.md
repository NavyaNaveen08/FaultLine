# Faultline 
> **Fault-tolerant multi-agent incident investigation harness**

Faultline investigates software incidents by orchestrating specialized AI agents to analyze application logs, error traces, and service configurations. It produces an evidence-backed incident report with root causes, confidence levels, alternatives, and explicit uncertainties.

---

## Architecture Overview

Faultline uses a **centralized blackboard architecture** with strictly typed Pydantic models, fail-closed human approval gates, and asynchronous coordination:

1. **Lead Investigator**: Ingests incident context and data directory manifest; synthesizes a structured investigation plan with subtasks.
2. **Specialists in Parallel (`asyncio.gather`)**:
   - **Timeline Specialist**: Reconstructs the chronological event timeline, error spikes, and latencies from log files.
   - **Dependency Specialist**: Maps service topologies, connection bounds, timeouts, and failure blast radiuses from configs.
3. **Hypothesis Specialist**: Evaluates collected evidence to formulate and rank candidate root causes.
4. **Checker Agent**: Adversarially challenges findings, cross-examining cited log lines and configurations against raw files to eliminate hallucinations.
   - **Bounded Revision Loop**: If the Checker identifies material unsupported claims or invalid citations, at most **one** revision attempt is granted to the Hypothesis Specialist, followed by a recheck.
5. **Synthesis & Audit**: Lead Investigator produces the final `InvestigationReport` with explicit uncertainties and verified evidence citations. Every agent action is audited in a JSONL execution trace.
6. **Single-Agent Baseline**: Provides a fair comparative benchmark using a single monolithic prompt with access to the same 3 read-only tools, same models, and identical citation verification rules.
7. **Human Approval Gate**: Protects external paid API endpoints and sensitive operations with a fail-closed confirmation prompt.

```
                           +----------------------+
                           |   User Incident &    |
                           |  Synthetic Evidence  |
                           +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           |  Human Approval Gate | (Fail-closed confirmation)
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
├── .gitignore               # Excludes secrets, venvs, caches, traces, eval results
├── pyproject.toml           # Package configuration & pytest settings
├── requirements.txt         # Core dependencies: groq, pydantic, python-dotenv, pytest
├── README.md                # Project documentation
├── sample_incident/         # Task 1: Redis pool exhaustion & HTTP 504 timeouts
│   ├── incident.txt         # Incident description
│   ├── logs/checkout.log    # Timestamped logs showing latency spikes and pool exhaustion
│   └── config/services.json # Dependency topology, timeouts, and connection bounds
├── sample_incident_auth/    # Task 2: Auth service JWKS cache OOM & HTTP 503 latency
│   ├── incident.txt         # Incident description
│   ├── logs/auth.log        # Timestamped logs showing JWKS cache growth and OOMKilled
│   └── config/services.json # Heap limit (512MB) and cache capacity configs
├── sample_incident_db/      # Task 3: Postgres deadlocks & connection starvation
│   ├── incident.txt         # Incident description
│   ├── logs/order.log       # Timestamped logs showing deadlock detected & pool exhaustion
│   └── config/services.json # Postgres pool size (10) and row lock parameters
├── faultline/
│   ├── __init__.py
│   ├── approval.py          # Fail-closed human approval gate for external API calls
│   ├── baseline.py          # Single-agent monolithic baseline investigator
│   ├── evaluator.py         # Comparative evaluation runner and objective rubric scorer
│   ├── cli.py               # CLI entry point (investigate, inspect-trace, evaluate)
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
    ├── test_approval.py     # Fail-closed human approval gate and non-interactive tests
    ├── test_budget.py       # Shared token budget enforcement & accumulation tests
    ├── test_evaluation.py   # Baseline execution, task fixtures, and rubric tests
    ├── test_injection.py    # Prompt-injection resistance & exfiltration blocking tests
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
- **Fail-Closed Human Approval Gate (`faultline/approval.py`)**:
  - Live calls to external paid services (Groq) or future sensitive actions require explicit confirmation (`[y/N]`).
  - Defaults to rejecting if confirmation is denied, EOF is reached, or running in an automated non-interactive shell without `--yes` or `FAULTLINE_APPROVE_EXTERNAL_CALLS=1`.
- **Prompt-Injection Resistance**: Agents operate under a `SECURITY_SYSTEM_PROMPT` explicitly defining all log files, configurations, and stack traces as untrusted data. Malicious directives (e.g. `SYSTEM OVERRIDE`, `EXECUTE_SHELL`) embedded in logs are parsed passively as evidence and never executed.
- **Zero Secret Leakage**:
  - `ExecutionTracer._sanitize()` automatically redacts API keys, passwords, and bearer tokens from logs.
  - Actual token consumption metrics (`prompt_tokens`, `completion_tokens`, `total_tokens`) are preserved unredacted.
- **Git Exclusions**: `.gitignore` strictly excludes `.env`, `traces/`, `eval_results/`, `.venv/`, `.pytest_cache/`, and log files.

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
*(Runs all 47 unit tests across models, agents, tools, tracer, approval gate, orchestrator, baseline, evaluation suite, budget, and prompt-injection defense).*

---

### 2. Run Investigations Across All 3 Incident Tasks (Requires `GROQ_API_KEY`)

#### Task 1: Redis Connection Pool Exhaustion (504 Timeouts)
```bash
python -m faultline.cli --incident-file sample_incident/incident.txt --data-dir sample_incident
```

#### Task 2: Auth Service JWKS Cache OOM (503 Latency Spike)
```bash
python -m faultline.cli --incident-file sample_incident_auth/incident.txt --data-dir sample_incident_auth
```

#### Task 3: Postgres Database Deadlocks & Connection Starvation
```bash
python -m faultline.cli --incident-file sample_incident_db/incident.txt --data-dir sample_incident_db
```

*Note*: Each command prompts for human confirmation before initiating live model calls. To pre-authorize in headless/CI environments, add `-y` or `--yes`.

---

### 3. Run Single-Agent Baseline
To run the monolithic single-agent baseline on any task (using the same tools and rubric):
```bash
python -m faultline.cli --incident-file sample_incident/incident.txt --data-dir sample_incident --baseline
```

---

### 4. Run Comparative Evaluation Suite Across All 3 Tasks
To evaluate both multi-agent and single-agent baseline approaches and generate local metrics:
```bash
python -m faultline.cli --evaluate
```
Results and objective rubric scores are written to `eval_results/evaluation_summary.json` (automatically excluded from Git).

---

### 5. Inspect and Audit an Execution Trace (Offline)
Inspect any recorded JSONL trace without invoking the model API:
```bash
python -m faultline.cli --inspect-trace traces/trace_20261014_151843.jsonl
```

---

## Evaluation Rubric Definition

The evaluation harness scores runs along four objective, deterministic dimensions (0.0 to 1.0):

1. **Correctness (0.0 - 1.0)**:
   - Measures whether the primary hypothesis and root cause identify the true technical failure mechanism verified against the incident ground truth.
2. **Evidence Grounding (0.0 - 1.0)**:
   - Evaluates citation veracity: $\frac{\text{Physically Verified Citations}}{\text{Total Citations Emitted}}$. Citations must match the specified relative path, 1-based line number, and exact literal line substring.
3. **Completeness (0.0 - 1.0)**:
   - Checks presence of all 4 critical reporting sections: chronological timeline, service dependency impacts, alternative hypotheses, and explicit uncertainties/gaps.
4. **Safe Handling (0.0 - 1.0)**:
   - Evaluates runtime hygiene: zero tool safety breaches, graceful degradation on specialist failure, and execution strictly within call and token budgets.

---

## Demonstrating Failure Recovery & Budget Limits Offline

Failure recovery and budget enforcement can be demonstrated credibly and deterministically without live API calls:

### 1. Specialist Failure & Timeout Isolation
```bash
pytest tests/test_orchestrator.py -k "failing_specialist or timed_out_specialist" -v
```
- **Proves**: When a specialist agent throws an exception or hits a timeout, the orchestrator isolates the error, continues parallel execution with remaining specialists, documents the failure in `failed_steps`, and produces a valid partial report.

### 2. Budget-Blocked Revision & Token Exhaustion
```bash
pytest tests/test_orchestrator.py -k "revision_loop_budget_blocked" -v
pytest tests/test_budget.py -k "subsequent_call_blocked" -v
```
- **Proves**: When call or token limits are reached, subsequent model calls and revision attempts are immediately blocked with `BudgetExceededError`, logging a `revision_blocked` event and returning an unverified claims disclosure.

### 3. Fail-Closed Human Approval Gate
```bash
pytest tests/test_approval.py -v
```
- **Proves**: External live calls without approval fail closed; non-interactive environments default to rejection.

---

## Comparison Methodology & Actual Results

No fabricated benchmark numbers or simulated multi-model rankings are claimed. The system was evaluated using actual measured executions:

1. **Deterministic Offline Test Suite**:
   - **Total Tests**: 47 passed in ~4.0s.
   - **Coverage Areas**: Tool path traversal prevention, exact line-level snippet citation, single-attempt revision loops, honest unknown token accounting, prompt-injection defense, fail-closed human approval, single-agent baseline, and offline trace inspection.

2. **Live Measured Benchmark Results Across All 3 Tasks (`--evaluate`)**:
   Executed against `llama-3.3-70b-versatile` under identical system directives and read-only tools:

   | Task | Approach | Status | Duration (s) | Model Calls | Prompt Tokens | Completion Tokens | Total Tokens | Citations Verified | Rubric Score |
   |---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
   | **Task 1: Redis Pool (504)** | **Multi-Agent** | **PASS** | 150.3s | 8 | 17,694 | 10,523 | 28,217 | 14 / 14 (100%) | **1.00** |
   | | **Single-Agent** | **PASS** | 37.0s | 1 | 2,280 | 3,013 | 5,293 | 12 / 12 (100%) | **1.00** |
   | **Task 2: Auth OOM (503)** | **Multi-Agent** | **PASS** | 222.1s | 7 | 12,974 | 7,623 | 20,597 | 5 / 5 (100%) | **1.00** |
   | | **Single-Agent** | **PASS** | 29.2s | 1 | 2,343 | 3,072 | 5,415 | 12 / 12 (100%) | **0.94** |
   | **Task 3: DB Deadlock (500)** | **Multi-Agent** | **PASS** | 161.8s | 5 | 7,900 | 6,129 | 14,029 | 6 / 6 (100%) | **1.00** |
   | | **Single-Agent** | **PASS** | 31.9s | 1 | 2,329 | 2,960 | 5,289 | 13 / 13 (100%) | **1.00** |

   **Key Takeaways**:
   - **Single-Agent Baseline**: Faster (~30–37s per incident) and lower token overhead (~5,300 tokens), but skips adversarial cross-examination and revision loops.
   - **Multi-Agent Orchestration**: Higher latency (~150–222s) and token consumption (~14,000–28,000 tokens), but actively challenges findings via the Checker Agent, catches unsupported inferences, and conducts bounded revision before report synthesis.


---

## Limitations & Tradeoffs

1. **Token Upper Bounds with Unknown Provider Usage**:
   - When the LLM provider omits token usage data (`has_unknown_tokens=True`), unmeasured token consumption cannot be calculated without fabricating numbers. Faultline refuses to invent counts or assume zero; it documents this limitation and uses call limits (`max_calls`) and deadlines (`overall_timeout`) as deterministic safeguards.
2. **Trace Inspection vs. Full Pipeline Replay**:
   - Traces record execution audit events, tool interactions, citation verification outcomes, and operational metrics. However, raw model response strings and complete prompts were omitted from the trace schema. As a result, offline trace analysis performs event inspection and auditing, not full offline deterministic pipeline replay.
3. **Strict Read-Only Sandboxing**:
   - Confining agents to read-only safe inspection prevents active testing (e.g. issuing test HTTP queries or pinging endpoints). The investigation relies entirely on static logs and configurations present in the incident directory.
4. **Single-Agent vs. Multi-Agent Trade-Off**:
   - Multi-agent orchestration provides superior citation grounding and adversarial hallucination reduction through the Checker Agent's bounded revision loop, but incurs higher token latency and multiple API round-trips compared to the single-agent baseline.

---

## Cuts & Scope Reductions

1. **No Autonomous Shell Execution**: Excluded shell execution tools to prevent prompt-injection escalation and destructive system actions.
2. **Bounded Revision Loop**: Strictly capped Checker revision to at most **one** pass to avoid unbounded token consumption and infinite critique loops.
3. **No Unverified Benchmarks**: Cut speculative performance claims across untested models; evaluation is anchored strictly in reproducible offline test suites and transparent rubric definitions.

---

## Time Spent & AI Usage

- **Total Time Spent**: Approximately 4.5 hours across architecture design, multi-agent orchestrator implementation, security sandboxing, budget enforcement, testing, single-agent baseline comparison, and documentation.
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
