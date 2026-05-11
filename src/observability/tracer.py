"""Phoenix tracing setup (OTLP HTTP exporter).

Call `setup()` once per process. After that, any span created via
`get_tracer()` and any auto-instrumented Anthropic / OpenAI SDK call is
exported to the self-hosted Phoenix instance running at
http://localhost:6006 by default.

Phoenix is fully open-source (Elastic License 2.0, no feature gates on
self-hosted). Unlike Langfuse, Phoenix routes spans to per-project views
using the OpenInference `openinference.project.name` attribute — so each
pipeline run automatically gets its own Phoenix project (just emit spans
with that attribute set to `<project>-<run_id>`; Phoenix auto-creates the
project if it doesn't exist).

No auth header is required for a local Phoenix server (auth is off by
default; enable via `PHOENIX_ENABLE_AUTH=true` if needed).

Override via env:
  PHOENIX_HOST           default http://localhost:6006
"""
from __future__ import annotations

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
_PHOENIX_HOST: Optional[str] = None


_DEFAULT_HOST = "http://localhost:6006"


def _build_provider(project_name: str) -> TracerProvider:
    """Construct a fresh TracerProvider routing spans to `project_name`.

    Phoenix routes spans to projects by the RESOURCE attribute
    `openinference.project.name` (set at provider creation time), not by
    per-span attributes. So per-run project isolation requires building a
    fresh provider per run — see `rotate_project()`.
    """
    host = os.environ.get("PHOENIX_HOST", _DEFAULT_HOST).rstrip("/")
    headers: dict[str, str] = {}
    api_key = os.environ.get("PHOENIX_API_KEY")
    if api_key:
        headers["api-key"] = api_key

    provider = TracerProvider(resource=Resource.create({
        # Both keys: service.name is the OTEL standard; openinference.project.name
        # is what Phoenix actually reads.
        "service.name": project_name,
        "openinference.project.name": project_name,
    }))
    exporter = OTLPSpanExporter(
        endpoint=f"{host}/v1/traces",
        headers=headers or None,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def setup(project_name: str = "video-workflow",
          launch_ui: bool = True,
          port: int = 6006) -> None:
    """Idempotent ONE-TIME wiring of instrumentation + initial provider.

    `launch_ui` / `port` are kept for callsite compatibility — Phoenix is
    hosted externally (start it with `python -m phoenix.server.main serve`
    or `phoenix serve`); this setup() configures the OTLP HTTP exporter
    that ships spans to it.

    Per-run project routing is done via `rotate_project()` — call that
    each time a new pipeline run starts so its spans land in their own
    Phoenix project (auto-created on first span).
    """
    global _INITIALIZED, _PHOENIX_HOST
    if _INITIALIZED:
        return

    host = os.environ.get("PHOENIX_HOST", _DEFAULT_HOST).rstrip("/")
    _PHOENIX_HOST = host

    trace.set_tracer_provider(_build_provider(project_name))

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


def rotate_project(project_name: str) -> None:
    """Switch the global TracerProvider so subsequent spans route to a
    different Phoenix project. Flushes the current provider first so the
    previous run's tail spans aren't lost.

    OTEL's public `trace.set_tracer_provider()` is **set-once** — it logs
    "Overriding of current TracerProvider is not allowed" and silently
    drops the new provider. Reach past that guard via the API package's
    private `_TRACER_PROVIDER` / `_TRACER_PROVIDER_SET_ONCE` globals.

    This is the only way to get per-run Phoenix project routing inside a
    long-lived server process short of forking a subprocess per run.
    NOT concurrency-safe — REGISTRY enforces single-run-at-a-time.
    """
    old = trace.get_tracer_provider()
    if hasattr(old, "force_flush"):
        try:
            old.force_flush(timeout_millis=5000)  # type: ignore[attr-defined]
        except Exception:
            pass

    new_provider = _build_provider(project_name)
    try:
        # Bypass the set-once guard by writing the internals directly.
        # Confirmed attribute names against opentelemetry-api 1.x.
        import opentelemetry.trace as _t
        _t._TRACER_PROVIDER = new_provider  # type: ignore[attr-defined]
        once = getattr(_t, "_TRACER_PROVIDER_SET_ONCE", None)
        if once is not None and hasattr(once, "_done"):
            once._done = False  # type: ignore[attr-defined]
    except Exception:
        # Fallback to the public path; it will warn but at least the new
        # provider is built and the old one was flushed.
        trace.set_tracer_provider(new_provider)


def get_tracer(name: str = "video-workflow"):
    """Return an OTEL tracer. `setup()` must have been called first."""
    return trace.get_tracer(name)


def phoenix_url() -> Optional[str]:
    return _PHOENIX_HOST or _DEFAULT_HOST


# Back-compat alias for callers still importing langfuse_url. Returns the
# Phoenix URL; if anything still routes UI links there it will land on the
# new Phoenix UI rather than 404.
langfuse_url = phoenix_url
