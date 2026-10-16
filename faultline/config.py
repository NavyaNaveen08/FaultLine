"""Configuration module for Faultline.

Loads environment variables safely using python-dotenv and provides
immutable or validated runtime settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from workspace root if present
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the Faultline harness."""

    # Groq API configuration
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    fallback_model: str = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")

    # Timeouts (seconds)
    agent_timeout_seconds: float = float(
        os.getenv("FAULTLINE_AGENT_TIMEOUT_SECONDS", "60.0")
    )
    overall_timeout_seconds: float = float(
        os.getenv("FAULTLINE_OVERALL_TIMEOUT_SECONDS", "300.0")
    )

    # Retry and limit bounds
    max_retries: int = int(os.getenv("FAULTLINE_MAX_RETRIES", "3"))
    max_tool_calls_per_agent: int = int(
        os.getenv("FAULTLINE_MAX_TOOL_CALLS_PER_AGENT", "10")
    )
    max_file_read_bytes: int = int(
        os.getenv("FAULTLINE_MAX_FILE_READ_BYTES", "65536")
    )  # 64 KB cap
    max_total_model_calls: int = int(
        os.getenv("FAULTLINE_MAX_TOTAL_MODEL_CALLS", "20")
    )
    max_total_tokens: Optional[int] = (
        int(os.getenv("FAULTLINE_MAX_TOTAL_TOKENS"))
        if os.getenv("FAULTLINE_MAX_TOTAL_TOKENS")
        else None
    )

    def validate_api_key(self) -> None:
        """Validate presence of Groq API key without exposing its value."""
        if not self.groq_api_key or self.groq_api_key.startswith("your_groq_api_key"):
            raise ValueError(
                "GROQ_API_KEY is not set or contains default placeholder. "
                "Set it in your process environment or in a local .env file."
            )


settings = Settings()
