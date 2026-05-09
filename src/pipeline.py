"""Pipeline state machine.

Owns: run_dir, observability stack, persisted state.json, asset manifest.
Linear flow per WORKFLOW.md §10:
    Phase 1 → Phase 2 (HTML ∥ recording) → Phase 3 → Phase 4 → Phase 5 → done
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .types import Gate, Phase, PipelineState
from .observability.audit import set_run_context
from .observability.events import EventBus
from .observability.logger import agent_logger, setup_logging
from .observability.tracer import setup as setup_tracing


class Pipeline:
    """Pipeline runner: state + observability + asset manifest."""

    def __init__(self,
                 project: str,
                 run_id: Optional[str] = None,
                 workspace_root: Optional[Path] = None,
                 launch_observability_ui: bool = True):
        self.project = project
        self.run_id = run_id or uuid.uuid4().hex[:8]
        root = workspace_root or Path.cwd() / "workspace"
        self.run_dir = root / project / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.run_dir / "state.json"

        self.state = self._load_or_init()

        setup_tracing(project_name=f"{project}-{self.run_id}",
                      launch_ui=launch_observability_ui)
        setup_logging(self.run_dir)
        self.bus = EventBus(self.run_dir, self.run_id)
        set_run_context(self.run_id, self.bus, self.run_dir)
        self.log = agent_logger("pipeline")
        self.log.info(f"pipeline init project={project} run_id={self.run_id}")

    # ─── state persistence ──
    def _load_or_init(self) -> PipelineState:
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            return PipelineState(**data)
        return PipelineState(run_id=self.run_id, project=self.project)

    def save(self) -> None:
        self.state_file.write_text(
            json.dumps(asdict(self.state), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # ─── transitions ──
    def transition(self, *,
                   phase: Optional[Phase] = None,
                   gate: Optional[Gate] = None) -> None:
        prev_gate = self.state.gate
        prev_phase = self.state.phase
        if phase is not None:
            self.state.phase = phase
        if gate is not None:
            self.state.gate = gate
            self.bus.emit("gate_enter", agent="pipeline",
                          gate=gate, prev_gate=prev_gate)
        self.save()
        self.log.info(
            f"transition: phase {prev_phase}→{self.state.phase}, "
            f"gate {prev_gate}→{self.state.gate}"
        )

    def gate_pass(self, gate: Gate, **payload) -> None:
        self.bus.emit("gate_pass", agent="pipeline", gate=gate, **payload)
        self.log.info(f"gate_pass: {gate}")

    # ─── asset manifest ──
    def record_asset(self, name: str, path: Path, verified: bool = False, **meta) -> None:
        self.state.manifest[name] = {
            "path": str(path),
            "verified": verified,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **meta,
        }
        self.save()
        evt = "asset_verified" if verified else "asset_failed"
        self.bus.emit(evt, agent="pipeline", name=name, path=str(path))
