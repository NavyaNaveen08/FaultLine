"""Tests for the fail-closed human approval mechanism."""

from __future__ import annotations

import sys
from unittest.mock import patch
import pytest

from faultline.approval import (
    ApprovalDeniedError,
    default_cli_approval_prompt,
    get_approval_callback,
    request_approval,
    set_approval_callback,
)


@pytest.fixture(autouse=True)
def reset_approval_callback():
    """Ensure global callback is clean before and after each test."""
    set_approval_callback(None)
    yield
    set_approval_callback(None)


def test_approval_granted_with_custom_callback():
    """Verify that an approving callback allows the action to proceed."""
    mock_cb = lambda action, details: True
    approved = request_approval("test_action", {"item": 1}, callback=mock_cb)
    assert approved is True


def test_approval_denied_raises_approval_denied_error():
    """Verify that rejecting the action raises ApprovalDeniedError under fail_fast=True."""
    mock_cb = lambda action, details: False
    with pytest.raises(ApprovalDeniedError) as exc_info:
        request_approval("test_risky_action", {"danger": "high"}, callback=mock_cb)
    assert "rejected by the human approval gate" in str(exc_info.value)


def test_approval_denied_returns_false_when_fail_fast_disabled():
    """Verify that fail_fast=False returns False instead of raising an error."""
    mock_cb = lambda action, details: False
    result = request_approval("test_action", callback=mock_cb, fail_fast=False)
    assert result is False


def test_global_approval_callback_injection():
    """Verify that setting the global callback intercepts request_approval."""
    set_approval_callback(lambda action, details: action == "allowed_action")

    assert request_approval("allowed_action") is True
    with pytest.raises(ApprovalDeniedError):
        request_approval("forbidden_action")


def test_default_cli_prompt_non_interactive_defaults_to_reject():
    """Verify that non-interactive sessions without TTY fail closed by default."""
    with patch.object(sys.stdin, "isatty", return_value=False):
        # Without env override
        with patch.dict("os.environ", {}, clear=True):
            assert default_cli_approval_prompt("action", {}) is False

        # With explicit env override
        with patch.dict("os.environ", {"FAULTLINE_APPROVE_EXTERNAL_CALLS": "1"}):
            assert default_cli_approval_prompt("action", {}) is True


def test_default_cli_prompt_interactive_input():
    """Verify interactive CLI confirmation prompt responses."""
    with patch.object(sys.stdin, "isatty", return_value=True):
        # User types 'y'
        with patch.object(sys.stdin, "readline", return_value="y\n"):
            assert default_cli_approval_prompt("action", {"foo": "bar"}) is True

        # User types 'yes'
        with patch.object(sys.stdin, "readline", return_value="yes\n"):
            assert default_cli_approval_prompt("action", {}) is True

        # User types 'n' or anything else
        with patch.object(sys.stdin, "readline", return_value="n\n"):
            assert default_cli_approval_prompt("action", {}) is False

        # User presses Enter (empty string / default No)
        with patch.object(sys.stdin, "readline", return_value="\n"):
            assert default_cli_approval_prompt("action", {}) is False
