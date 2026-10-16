"""Pytest fixtures for Faultline test suite."""

from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture
def sample_incident_dir(tmp_path: Path) -> Path:
    """Creates a synthetic incident directory with sample logs and config files."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)

    # Sample service log
    service_log = logs_dir / "payment_service.log"
    service_log.write_text(
        "2026-10-14T10:00:01Z [INFO] Service started\n"
        "2026-10-14T10:05:22Z [WARN] Database pool connection threshold reached (90%)\n"
        "2026-10-14T10:05:30Z [ERROR] Connection timeout after 5000ms to postgres-primary\n"
        "2026-10-14T10:05:31Z [ERROR] Failed to process payment for txn_991823\n",
        encoding="utf-8",
    )

    # Sample config
    config_file = config_dir / "database.yaml"
    config_file.write_text(
        "database:\n"
        "  host: postgres-primary\n"
        "  port: 5432\n"
        "  pool_size: 10\n"
        "  timeout_ms: 5000\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def sample_trace_file(tmp_path: Path) -> Path:
    """Provides a temporary trace file path for JSONL testing."""
    return tmp_path / "test_trace.jsonl"
