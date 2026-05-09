"""Langfuse tracing setup (OTLP HTTP exporter).

Call `setup()` once at pipeline entry. After that, any span created via
`get_tracer()` and any auto-instrumented Anthropic / OpenAI SDK call is
exported to the self-hosted Langfuse instance running at
http://localhost:3000 by default.

Auth uses Basic <base64(public_key:secret_key)>; the keys come from
the docker .env (LANGFUSE_INIT_PROJECT_PUBLIC_KEY / _SECRET_KEY).

Override via env:
  LANGFUSE_HOST          default http://localhost:3000
  LANGFUSE_PUBLIC_KEY    default pk-lf-local-a54eab0493a6786e
  LANGFUSE_SECRET_KEY    default sk-lf-local-f1c0e48b0a8ca49c17330a89cd116d45
"""
from __future__ import annotations

import base64
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)

_INITIALIZED = False
_LANGFUSE_HOST: Optional[str] = None


_DEFAULT_HOST = "http://localhost:3000"
_DEFAULT_PUBLIC_KEY = "pk-lf-local-a54eab0493a6786e"
_DEFAULT_SECRET_KEY = "sk-lf-local-f1c0e48b0a8ca49c17330a89cd116d45"


def setup(project_name: str = "video-workflow",
          launch_ui: bool = True,
          port: int = 3000) -> None:
    """Idempotent: configure OTEL exporter → Langfuse + auto-instrument LLM SDKs.

    `launch_ui` / `port` are accepted for callsite compatibility but are
    now no-ops: Langfuse is hosted externally via docker compose (this
    setup() only configures the OTLP HTTP exporter that ships spans to it).
    """
    global _INITIALIZED, _LANGFUSE_HOST
    if _INITIALIZED:
        return

    host = os.environ.get("LANGFUSE_HOST", _DEFAULT_HOST).rstrip("/")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", _DEFAULT_PUBLIC_KEY)
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", _DEFAULT_SECRET_KEY)
    _LANGFUSE_HOST = host

    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

    provider = TracerProvider(resource=Resource.create({
        "service.name": project_name,
    }))
    exporter = OTLPSpanExporter(
        endpoint=f"{host}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {auth}"},
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

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

    _INITIALIZED = True


def get_tracer(name: str = "video-workflow"):
    """Return an OTEL tracer. `setup()` must have been called first."""
    return trace.get_tracer(name)


def langfuse_url() -> Optional[str]:
    return _LANGFUSE_HOST or _DEFAULT_HOST
