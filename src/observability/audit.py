"""@traced_agent decorator.

Wraps an Agent's execution in a single OTEL span and emits agent_start /
agent_done events. Every Agent entry point in the workflow MUST use this.
"""
from __future__ import annotations

import functools
import traceback
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Optional

from opentelemetry.trace import Status, StatusCode

from .events import EventBus
from .tracer import get_tracer


_RUN_CONTEXT: ContextVar[dict] = ContextVar("run_context", default={})


def set_run_context(run_id: str, event_bus: EventBus, run_dir: Path) -> None:
    _RUN_CONTEXT.set({
        "run_id": run_id,
        "event_bus": event_bus,
        "run_dir": run_dir,
    })


def get_run_context() -> dict:
    return _RUN_CONTEXT.get()


def traced_agent(name: str, phase: int) -> Callable:
    """Decorator: wraps an Agent function in a parent span + lifecycle events."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = get_run_context()
            run_id = ctx.get("run_id", "unknown")
            bus: Optional[EventBus] = ctx.get("event_bus")

            tracer = get_tracer("video-workflow")
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("agent.name", name)
                span.set_attribute("agent.phase", phase)
                span.set_attribute("run_id", run_id)

                if bus is not None:
                    bus.emit("agent_start", agent=name, phase=phase)

                try:
                    result = fn(*args, **kwargs)
                    span.set_attribute("agent.result", str(result)[:500])
                    if bus is not None:
                        bus.emit("agent_done", agent=name, phase=phase,
                                 result=str(result)[:500])
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.add_event("exception", attributes={
                        "exception.type": type(e).__name__,
                        "exception.message": str(e),
                        "exception.stacktrace": traceback.format_exc()[:4000],
                    })
                    if bus is not None:
                        bus.emit("agent_done", agent=name, phase=phase,
                                 error=str(e), error_type=type(e).__name__)
                    raise

        wrapper.__traced_agent__ = True  # type: ignore[attr-defined]
        return wrapper

    return decorator
