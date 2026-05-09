"""Phoenix tracing setup.

Call `setup()` once at pipeline entry. After that, any Anthropic SDK call is
automatically traced into Phoenix at http://localhost:6006.
"""
from __future__ import annotations

import os
from typing import Optional

import phoenix as px
from phoenix.otel import register
from opentelemetry import trace

_INITIALIZED = False
_PHOENIX_SESSION = None


def setup(project_name: str = "video-workflow",
          launch_ui: bool = True,
          port: int = 6006) -> None:
    """Idempotent: configure OTEL + auto-instrument Anthropic SDK + (optionally) launch UI."""
    global _INITIALIZED, _PHOENIX_SESSION
    if _INITIALIZED:
        return

    if launch_ui:
        os.environ.setdefault("PHOENIX_PORT", str(port))
        _PHOENIX_SESSION = px.launch_app()
        register(project_name=project_name, auto_instrument=False)

        try:
            from openinference.instrumentation.anthropic import AnthropicInstrumentor
            AnthropicInstrumentor().instrument()
        except ImportError:
            pass

        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor
            OpenAIInstrumentor().instrument()
        except ImportError:
            pass
    # When launch_ui=False: no register() → OTel uses no-op tracer; spans become
    # no-ops, instrumentation skipped, no OTLP retry noise.

    _INITIALIZED = True


def get_tracer(name: str = "video-workflow"):
    """Return an OTEL tracer. `setup()` must have been called first."""
    return trace.get_tracer(name)


def phoenix_url() -> Optional[str]:
    if _PHOENIX_SESSION is not None:
        return _PHOENIX_SESSION.url
    return "http://localhost:6006"
