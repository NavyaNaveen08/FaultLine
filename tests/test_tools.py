"""Tests for sandboxed safe file inspection tools."""

from __future__ import annotations

from pathlib import Path
import pytest

from faultline.tools import (
    SandboxedPathError,
    list_directory_safe,
    read_file_safe,
    search_file_safe,
)


def test_list_directory_safe(sample_incident_dir: Path) -> None:
    entries = list_directory_safe(sample_incident_dir)
    names = [e["name"] for e in entries]
    assert "logs" in names
    assert "config" in names


def test_read_file_safe(sample_incident_dir: Path) -> None:
    content = read_file_safe(sample_incident_dir, "logs/payment_service.log")
    assert "Connection timeout" in content
    assert "postgres-primary" in content


def test_read_file_safe_truncation(sample_incident_dir: Path) -> None:
    # Read with tight byte limit
    content = read_file_safe(sample_incident_dir, "logs/payment_service.log", max_bytes=30)
    assert len(content) >= 30
    assert "[... Truncated by Faultline Safe File Tool ...]" in content


def test_search_file_safe(sample_incident_dir: Path) -> None:
    matches = search_file_safe(sample_incident_dir, "logs/payment_service.log", "ERROR")
    assert len(matches) == 2
    assert matches[0]["line_number"] == 3
    assert "Connection timeout" in matches[0]["content"]


def test_path_traversal_prevention_parent_dir(sample_incident_dir: Path) -> None:
    with pytest.raises(SandboxedPathError):
        read_file_safe(sample_incident_dir, "../outside.txt")


def test_path_traversal_prevention_absolute_path(sample_incident_dir: Path) -> None:
    with pytest.raises(SandboxedPathError):
        read_file_safe(sample_incident_dir, "C:/Windows/System32/drivers/etc/hosts")


def test_file_not_found(sample_incident_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_file_safe(sample_incident_dir, "logs/non_existent.log")
