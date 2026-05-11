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
from .observability.tracer import setup as setup_tracing, rotate_project


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

        # First call wires instrumentation; subsequent calls are no-ops.
        setup_tracing(project_name=f"{project}-{self.run_id}",
                      launch_ui=launch_observability_ui)
        # Phoenix routes by resource attribute set at provider creation,
        # so per-run isolation requires actually swapping the provider here.
        # This makes every span emitted during this pipeline land in a
        # Phoenix project named `<project>-<run_id>` (auto-created).
        rotate_project(f"{project}-{self.run_id}")
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

    # ─── error state ──
    def record_error(self, *, phase: Phase, agent: str,
                       error_type: str, error_text: str) -> None:
        """Mark the run as failed; UI surfaces a banner + retry button.

        Does NOT decide for the user — just persists the fact so the UI can
        offer retry/edit-prompt/error-agent. User-driven recovery only.
        """
        self.state.last_error = {
            "phase": phase,
            "agent": agent,
            "error_type": error_type,
            "error_text": error_text[:2000],
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self.state.gate = "failed"
        self.save()
        self.bus.emit("phase_failed", agent=agent, phase=phase,
                       error=error_text[:500], error_type=error_type)
        self.log.error(f"phase_failed: phase={phase} agent={agent} {error_type}: {error_text[:200]}")

    def clear_error(self) -> None:
        self.state.last_error = None
        self.save()

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
