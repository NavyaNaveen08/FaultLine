"""Fail-closed human approval mechanism for live external API calls and risky operations.

Requires explicit confirmation before issuing external paid API requests (e.g. Groq)
or performing sensitive operations. Defaults to rejecting (fail-closed) if approval
is not explicitly granted or if running in a non-interactive environment without an
explicit authorization flag.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, Optional


class ApprovalDeniedError(Exception):
    """Raised when an operation is rejected by the human approval gate."""


ApprovalCallback = Callable[[str, Dict[str, Any]], bool]

# Global callback override (used for unit tests and headless automation)
_GLOBAL_APPROVAL_CALLBACK: Optional[ApprovalCallback] = None


def set_approval_callback(callback: Optional[ApprovalCallback]) -> None:
    """Set a global approval callback (primarily used for unit testing)."""
    global _GLOBAL_APPROVAL_CALLBACK
    _GLOBAL_APPROVAL_CALLBACK = callback


def get_approval_callback() -> Optional[ApprovalCallback]:
    """Return the currently configured approval callback."""
    return _GLOBAL_APPROVAL_CALLBACK


def default_cli_approval_prompt(action: str, details: Dict[str, Any]) -> bool:
    """Interactive CLI confirmation prompt.
    
    Fail-closed behavior:
    - If stdin is not a TTY (non-interactive, automated pipe, background process),
      returns False unless FAULTLINE_APPROVE_EXTERNAL_CALLS=1 is set.
    - If user enters anything other than 'y' or 'yes', returns False.
    - On EOF or keyboard interrupt, returns False.
    """
    env_override = os.environ.get("FAULTLINE_APPROVE_EXTERNAL_CALLS", "").strip().lower()
    if env_override in ("1", "true", "yes"):
        return True

    if not sys.stdin.isatty():
        # Fail-closed in non-interactive sessions
        return False

    detail_str = ""
    if details:
        safe_items = [
            f"{k}={v}" for k, v in details.items()
            if not any(s in str(k).lower() for s in ["key", "token", "secret", "password"])
        ]
        if safe_items:
            detail_str = f" ({', '.join(safe_items)})"

    try:
        sys.stdout.write(f"\n[APPROVAL GATE] External operation requested: '{action}'{detail_str}\n")
        sys.stdout.write("Explicit confirmation required to proceed. Authorize? [y/N]: ")
        sys.stdout.flush()
        line = sys.stdin.readline()
        if not line:
            return False
        return line.strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def request_approval(
    action: str,
    details: Optional[Dict[str, Any]] = None,
    callback: Optional[ApprovalCallback] = None,
    fail_fast: bool = True,
) -> bool:
    """Request human confirmation for an action.
    
    Parameters:
        action: Identifier for the sensitive action (e.g. 'live_groq_api_call').
        details: Safe context dictionary describing the action.
        callback: Optional custom approval function (defaults to configured or CLI prompt).
        fail_fast: If True, raises ApprovalDeniedError on rejection; if False, returns False.
    
    Returns:
        True if approved.
    
    Raises:
        ApprovalDeniedError: If approval is rejected and fail_fast is True.
    """
    effective_cb = callback or _GLOBAL_APPROVAL_CALLBACK or default_cli_approval_prompt
    details_dict = details or {}

    approved = bool(effective_cb(action, details_dict))
    if not approved and fail_fast:
        raise ApprovalDeniedError(
            f"Operation '{action}' was rejected by the human approval gate (fail-closed)."
        )
    return approved
