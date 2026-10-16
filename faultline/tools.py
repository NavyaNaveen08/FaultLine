"""Safe, strictly read-only file tools for agent evidence inspection.

Enforces sandboxing, path traversal prevention, and size/line bounds.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from faultline.config import settings


class SandboxedPathError(Exception):
    """Raised when an agent attempts to access a path outside the designated sandbox."""


def _resolve_and_validate_path(sandbox_dir: Path, rel_path: str) -> Path:
    """Resolve a relative path against the sandbox and assert confinement.

    Raises SandboxedPathError if the path is absolute or attempts directory traversal.
    """
    if Path(rel_path).is_absolute():
        raise SandboxedPathError(
            f"Security violation: path '{rel_path}' is an absolute path. Only relative paths within the incident directory are allowed."
        )

    sandbox_resolved = sandbox_dir.resolve()
    target_path = (sandbox_resolved / rel_path).resolve()

    try:
        target_path.relative_to(sandbox_resolved)
    except ValueError as exc:
        raise SandboxedPathError(
            f"Security violation: path '{rel_path}' resolves outside the allowed incident directory."
        ) from exc

    return target_path


def list_directory_safe(sandbox_dir: Path, rel_path: str = "") -> List[Dict[str, Any]]:
    """Safely list entries within a subdirectory of the incident folder."""
    target = _resolve_and_validate_path(sandbox_dir, rel_path)
    if not target.exists():
        raise FileNotFoundError(f"Directory not found: {rel_path}")
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {rel_path}")

    entries = []
    for item in sorted(target.iterdir()):
        entries.append(
            {
                "name": item.name,
                "relative_path": str(item.relative_to(sandbox_dir.resolve())),
                "is_dir": item.is_dir(),
                "size_bytes": item.stat().st_size if item.is_file() else 0,
            }
        )
    return entries


def read_file_safe(
    sandbox_dir: Path,
    rel_path: str,
    max_bytes: Optional[int] = None,
    max_lines: Optional[int] = None,
) -> str:
    """Safely read content from a file, bounded by byte size and line count."""
    target = _resolve_and_validate_path(sandbox_dir, rel_path)
    if not target.exists():
        raise FileNotFoundError(f"File not found: {rel_path}")
    if not target.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {rel_path}")

    limit = max_bytes or settings.max_file_read_bytes
    with open(target, "r", encoding="utf-8", errors="replace") as f:
        content = f.read(limit + 1)
        truncated = len(content) > limit
        if truncated:
            content = content[:limit]

    lines = content.splitlines()
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
        content = "\n".join(lines)

    if truncated:
        content += "\n\n[... Truncated by Faultline Safe File Tool ...]"

    return content


def search_file_safe(
    sandbox_dir: Path,
    rel_path: str,
    pattern: str,
    max_matches: int = 50,
) -> List[Dict[str, Any]]:
    """Safely search lines in a file matching a regex or literal query."""
    target = _resolve_and_validate_path(sandbox_dir, rel_path)
    if not target.exists():
        raise FileNotFoundError(f"File not found: {rel_path}")
    if not target.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {rel_path}")

    regex = re.compile(pattern, re.IGNORECASE)
    matches = []

    with open(target, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, start=1):
            if regex.search(line):
                matches.append({"line_number": line_num, "content": line.rstrip("\r\n")})
                if len(matches) >= max_matches:
                    break

    return matches
