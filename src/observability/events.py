"""Pipeline event bus.

Every state transition / artifact verification / user input must emit a
PipelineEvent here. Events are appended to events.jsonl AND attached as
events on the current OTEL span (visible in Langfuse UI).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from opentelemetry import trace

EventType = Literal[
    "agent_start", "agent_done",
    "gate_enter", "gate_pass",
    "asset_verified", "asset_failed",
    "user_input",
]


@dataclass
class PipelineEvent:
    ts: str
    run_id: str
    event: EventType
    agent: Optional[str] = None
    payload: dict = field(default_factory=dict)


class EventBus:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.events_file = run_dir / "events.jsonl"
        self.events_file.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: EventType, agent: Optional[str] = None, **payload: Any) -> PipelineEvent:
        evt = PipelineEvent(
            ts=datetime.now(timezone.utc).isoformat(),
            run_id=self.run_id,
            event=event,
            agent=agent,
            payload=payload,
        )

        with self.events_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(evt), ensure_ascii=False) + "\n")

        span = trace.get_current_span()
        if span is not None and span.is_recording():
            attrs = {"agent": agent or ""}
            for k, v in payload.items():
                attrs[f"payload.{k}"] = str(v)[:500]
            span.add_event(name=event, attributes=attrs)

        return evt
