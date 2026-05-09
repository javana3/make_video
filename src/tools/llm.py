"""LLM client factory.

Both SDKs are pointed at the 火山方舟 Coding Plan endpoint:
  - anthropic SDK   → Anthropic-compatible endpoint (/api/coding)
  - openai SDK      → OpenAI-compatible endpoint    (/api/coding/v3)

Auto-instrumented via openinference (configured in observability.tracer).
"""
from __future__ import annotations

import os
from typing import Optional

from anthropic import Anthropic
from openai import OpenAI


def _default_key() -> str:
    return (
        os.environ.get("ARK_KEY_1")
        or os.environ.get("ANTHROPIC_API_KEY")
        or ""
    )


def anthropic_client(api_key: Optional[str] = None) -> Anthropic:
    base = os.environ.get("ARK_BASE_URL_ANTHROPIC") or os.environ.get("ANTHROPIC_BASE_URL")
    return Anthropic(api_key=api_key or _default_key(), base_url=base)


def openai_client(api_key: Optional[str] = None) -> OpenAI:
    base = os.environ.get("ARK_BASE_URL_OPENAI")
    return OpenAI(api_key=api_key or _default_key(), base_url=base)


def model_for(role: str) -> str:
    """Resolve a logical role to a concrete model name (overridable via env)."""
    r = role.lower()
    if r in {"reasoning", "agent", "deep"}:
        return os.environ.get("LLM_REASONING", "claude-sonnet-4-20250514")
    if r in {"fast", "routing", "ping"}:
        return os.environ.get("LLM_FAST", "deepseek-v3.2")
    if r in {"opus", "best"}:
        return os.environ.get("LLM_DEEP", "claude-opus-4-20250514")
    raise ValueError(f"unknown role: {role!r}")
