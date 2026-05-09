"""Execute an approved setup_plan (no LLM).

Sequence:
  1. install_commands (sync, must succeed)
  2. seed_commands (sync, must succeed)
  3. services (async background, then health check each)

Writes a structured execution log to `setup_exec.json` so the UI can show
per-step status. Errors don't auto-recover — the user is shown the failure
and can iterate the plan.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger
from .service_manager import ServiceManager, ServiceRecord
from .shell import run as shell_run

StepStatus = Literal["pending", "running", "ok", "failed", "skipped"]


@dataclass
class StepResult:
    label: str
    command: str
    cwd: str
    status: StepStatus = "pending"
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_s: Optional[float] = None
    exit_code: Optional[int] = None
    stdout_tail: Optional[str] = None
    stderr_tail: Optional[str] = None


@dataclass
class ExecutionState:
    started_at: str
    status: Literal["running", "ok", "failed"] = "running"
    install_steps: list[StepResult] = field(default_factory=list)
    seed_steps: list[StepResult] = field(default_factory=list)
    service_records: list[dict] = field(default_factory=list)
    last_update: Optional[str] = None
    error: Optional[str] = None


def _persist(state: ExecutionState, path: Path) -> None:
    state.last_update = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_step(repo_dir: Path, step: dict, label: str,
              install_timeout: int = 600) -> StepResult:
    import os
    cmd = step.get("command", "")
    cwd_rel = step.get("cwd", ".")
    cwd_abs = (repo_dir / cwd_rel).resolve()
    res = StepResult(label=label, command=cmd, cwd=cwd_rel,
                     status="running",
                     started_at=datetime.now(timezone.utc).isoformat())
    t0 = time.monotonic()
    try:
        # Inject UTF-8 mode so child Python processes don't choke on UTF-8 files
        # when Windows default codepage is GBK/cp936 (target repo's read_text()
        # without explicit encoding= would otherwise blow up).
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        proc = shell_run(
            ["cmd", "/c", cmd] if _is_windows() else ["bash", "-lc", cmd],
            cwd=cwd_abs, timeout=install_timeout, env=env,
        )
        res.exit_code = proc.exit_code
        res.duration_s = round(time.monotonic() - t0, 1)
        res.stdout_tail = (proc.stdout or "")[-4000:]
        res.stderr_tail = (proc.stderr or "")[-4000:]
        res.status = "ok" if proc.exit_code == 0 else "failed"
    except Exception as e:
        res.duration_s = round(time.monotonic() - t0, 1)
        res.status = "failed"
        res.stderr_tail = f"{type(e).__name__}: {e}"
        res.exit_code = -1
    res.ended_at = datetime.now(timezone.utc).isoformat()
    return res


def _is_windows() -> bool:
    import os
    return os.name == "nt"


@traced_agent("Agent 2 SetupRunner · exec", phase=2)
def execute_plan(plan: dict, repo_dir: Path, state_path: Path,
                 services_dir: Path,
                 install_timeout: int = 600,
                 service_health_wait: float = 60.0) -> ExecutionState:
    """Run the plan synchronously. Returns final state."""
    log = agent_logger("agent2_setup")
    log.info(f"execute_plan  install={len(plan.get('install_commands') or [])} "
             f"seed={len(plan.get('seed_commands') or [])} "
             f"services={len(plan.get('services') or [])}")
    state = ExecutionState(started_at=datetime.now(timezone.utc).isoformat())
    _persist(state, state_path)

    # 1. install
    for i, step in enumerate(plan.get("install_commands") or []):
        res = _run_step(repo_dir, step, f"install[{i}] {step.get('purpose','')}",
                        install_timeout=install_timeout)
        state.install_steps.append(res)
        _persist(state, state_path)
        if res.status != "ok":
            state.status = "failed"
            state.error = (f"install step {i} failed (exit {res.exit_code}): "
                           f"{res.stderr_tail or ''}")
            _persist(state, state_path)
            return state

    # 2. seed
    for i, step in enumerate(plan.get("seed_commands") or []):
        res = _run_step(repo_dir, step, f"seed[{i}] {step.get('purpose','')}",
                        install_timeout=install_timeout)
        state.seed_steps.append(res)
        _persist(state, state_path)
        if res.status != "ok":
            state.status = "failed"
            state.error = f"seed step {i} failed (exit {res.exit_code})"
            _persist(state, state_path)
            return state

    # 3. services
    mgr = ServiceManager(services_dir)
    for s in plan.get("services") or []:
        cwd_abs = (repo_dir / s["cwd"]).resolve()
        try:
            mgr.start(name=s["name"], command=s["command"], cwd=cwd_abs,
                      port=int(s["port"]), health_url=s["health_url"])
        except Exception as e:
            state.status = "failed"
            state.error = f"failed to spawn service {s['name']}: {e}"
            _persist(state, state_path)
            return state
        state.service_records = [asdict(r) for r in mgr.list()]
        _persist(state, state_path)

    # 4. health check each
    all_healthy = True
    for s in plan.get("services") or []:
        rec = mgr.health_check(s["name"], max_wait_s=service_health_wait)
        if rec.status != "healthy":
            all_healthy = False
        state.service_records = [asdict(r) for r in mgr.list()]
        _persist(state, state_path)

    state.status = "ok" if all_healthy else "failed"
    if not all_healthy:
        state.error = "one or more services failed health check; see service_records"
    _persist(state, state_path)

    # Emit a lifecycle event when every service passed its health check.
    if all_healthy:
        bus = get_run_context().get("event_bus")
        if bus is not None:
            bus.emit("asset_verified", agent="agent2_setup",
                     name="services_healthy", path=str(state_path),
                     n_services=len(plan.get("services") or []))

    # Auto-open the primary frontend service in the user's default browser so
    # the project window is visible — Phase 2b will then auto-detect it.
    if all_healthy:
        _try_open_project_urls(plan)

    return state


def _try_open_project_urls(plan: dict) -> None:
    """Open frontend service URLs in the default browser (best effort)."""
    import webbrowser
    services = plan.get("services") or []
    # Prefer the service whose name suggests "frontend" / "ui" / "web"
    front = next((s for s in services if any(
        h in (s.get("name", "") + " " + s.get("purpose", "")).lower()
        for h in ("frontend", "ui", "web", "static"))), None)
    if not front and services:
        front = services[0]
    if not front:
        return
    url = front.get("health_url") or ""
    # health_url may end with /health → drop it for the user-facing page
    if url.endswith("/health"):
        url = url[:-len("/health")] + "/"
    try:
        webbrowser.open(url)
    except Exception:
        pass
