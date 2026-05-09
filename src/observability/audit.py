"""@traced_agent decorator + traced_step context manager.

Both produce OTEL spans visible in Langfuse UI. `@traced_agent` wraps an Agent
entry point; `traced_step(...)` is for instrumenting an inner step (e.g.
"MusicGen.from_pretrained") so its duration shows nested under the agent span.
"""
from __future__ import annotations

import functools
import inspect
import time
import traceback
from contextlib import contextmanager
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


def _summarize(value: Any, max_len: int = 1500) -> str:
    """Render any value as a debuggable string under max_len chars."""
    try:
        if isinstance(value, Path):
            try:
                if value.exists() and value.is_file():
                    return f"{value.name} ({value.stat().st_size} B)"
                return str(value)
            except Exception:
                return str(value)
        if isinstance(value, (dict, list)):
            import json as _json
            s = _json.dumps(value, ensure_ascii=False, default=str)
            return s if len(s) <= max_len else s[:max_len - 20] + "...<truncated>"
        s = str(value)
        return s if len(s) <= max_len else s[:max_len - 20] + "...<truncated>"
    except Exception:
        return repr(value)[:max_len]


def traced_agent(name: str, phase: int) -> Callable:
    """Decorator: wraps an Agent function in a parent span + lifecycle events.

    Span attributes recorded:
      - agent.name / agent.phase / run_id
      - agent.input.<param_name> = summarized value of each arg/kwarg
      - agent.result = summarized return value
      - agent.duration_s = wall time
    """
    def decorator(fn: Callable) -> Callable:
        try:
            sig = inspect.signature(fn)
            param_names = list(sig.parameters.keys())
        except (TypeError, ValueError):
            param_names = []

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

                # Record positional args
                for i, val in enumerate(args):
                    pname = param_names[i] if i < len(param_names) else f"arg{i}"
                    if pname in ("self", "cls"):
                        continue
                    span.set_attribute(f"agent.input.{pname}", _summarize(val))
                # Record kwargs
                for k, v in kwargs.items():
                    span.set_attribute(f"agent.input.{k}", _summarize(v))

                if bus is not None:
                    bus.emit("agent_start", agent=name, phase=phase)

                t0 = time.monotonic()
                try:
                    result = fn(*args, **kwargs)
                    elapsed = time.monotonic() - t0
                    span.set_attribute("agent.duration_s", round(elapsed, 3))
                    span.set_attribute("agent.result", _summarize(result, 2000))
                    if bus is not None:
                        bus.emit("agent_done", agent=name, phase=phase,
                                 result=_summarize(result, 500))
                    return result
                except Exception as e:
                    elapsed = time.monotonic() - t0
                    span.set_attribute("agent.duration_s", round(elapsed, 3))
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


@contextmanager
def traced_step(name: str, **static_attrs: Any):
    """Lightweight context manager for an inner step within an Agent.

    Usage:
        with traced_step("MusicGen.from_pretrained", model=model_name):
            model = ModelCls.from_pretrained(...)

    Records:
      - step.<key> for each kwarg
      - step.duration_s on completion
      - exception event + ERROR status on raise
    """
    tracer = get_tracer("video-workflow")
    with tracer.start_as_current_span(name) as span:
        for k, v in static_attrs.items():
            span.set_attribute(f"step.{k}", _summarize(v, 500))
        t0 = time.monotonic()
        try:
            yield span
            span.set_attribute("step.duration_s", round(time.monotonic() - t0, 3))
        except Exception as e:
            span.set_attribute("step.duration_s", round(time.monotonic() - t0, 3))
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.add_event("exception", attributes={
                "exception.type": type(e).__name__,
                "exception.message": str(e),
            })
            raise
