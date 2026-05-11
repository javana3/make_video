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
    status: Literal["running", "ok", "failed", "waiting_user_secrets"] = "running"
    config_writes: list[dict] = field(default_factory=list)
    install_steps: list[StepResult] = field(default_factory=list)
    seed_steps: list[StepResult] = field(default_factory=list)
    service_records: list[dict] = field(default_factory=list)
    waiting_for_secrets: list[dict] = field(default_factory=list)
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


_PLACEHOLDER_RE = __import__("re").compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _render_template(template: str, *, allowed: set[str]) -> tuple[str, list[str]]:
    """Substitute ${VAR} from os.environ. Returns (rendered, missing).

    Variables outside `allowed` are NOT substituted (left as literal ${VAR});
    this catches cases where the validator-side whitelist drifts from runtime.
    """
    import os
    missing: list[str] = []
    def repl(m):
        v = m.group(1)
        if v not in allowed:
            return m.group(0)
        val = os.environ.get(v)
        if val is None or val == "":
            missing.append(v)
            return m.group(0)
        return val
    return _PLACEHOLDER_RE.sub(repl, template), missing


def _load_user_secrets(run_dir: Path) -> dict[str, str]:
    """Read user-provided secrets from `<run_dir>/user_secrets.json`.

    The web UI's "provide secrets" form writes here; executor reads to render
    config_writes templates with ${USER_PROVIDED_VAR} substitutions.
    """
    p = run_dir / "user_secrets.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _run_config_writes(plan: dict, repo_dir: Path,
                       user_secrets: dict[str, str]) -> tuple[list[dict], Optional[str]]:
    """Render each config_writes entry and write to disk.

    Placeholder resolution source order:
      1. user-provided secrets (from <run_dir>/user_secrets.json)
      2. parent pipeline env (PARENT_CREDENTIALS via os.environ)

    Returns (results, error). On any error, error is non-None and execution
    should halt before install runs.
    """
    log = agent_logger("agent2_setup")
    # Late import to avoid circular dep (agents/setup_runner imports tools.llm
    # which imports things that pull plan_executor in some paths).
    from ..agents.setup_runner import PARENT_CREDENTIALS
    parent_allowed = set(PARENT_CREDENTIALS)
    user_allowed = set(user_secrets)
    results: list[dict] = []
    for i, w in enumerate(plan.get("config_writes") or []):
        path_rel = w.get("path") or ""
        tmpl = w.get("content_template") or ""
        purpose = w.get("purpose") or ""
        target = (repo_dir / path_rel).resolve()
        if repo_dir.resolve() not in target.parents and target != repo_dir.resolve():
            return results, f"config_writes[{i}] path escapes repo: {path_rel}"
        try:
            rendered, missing = _render_template_with_user_secrets(
                tmpl, parent_allowed=parent_allowed, user_secrets=user_secrets,
            )
        except Exception as e:
            return results, f"config_writes[{i}] template render failed: {e}"
        if missing:
            return results, (f"config_writes[{i}] references unresolved vars: {missing}. "
                              f"Either parent env missing or user_secrets.json incomplete.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
        size = target.stat().st_size
        log.info(f"config_writes[{i}] wrote {target} ({size}B)  purpose={purpose}")
        results.append({
            "path": path_rel,
            "purpose": purpose,
            "size_bytes": size,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
    return results, None


def _render_template_with_user_secrets(template: str, *, parent_allowed: set[str],
                                         user_secrets: dict[str, str]) -> tuple[str, list[str]]:
    """Substitute ${VAR} from user_secrets first, then os.environ for parent vars."""
    import os
    missing: list[str] = []
    def repl(m):
        v = m.group(1)
        if v in user_secrets:
            val = user_secrets[v]
            if val == "":
                missing.append(v)
                return m.group(0)
            return val
        if v in parent_allowed:
            val = os.environ.get(v)
            if val is None or val == "":
                missing.append(v)
                return m.group(0)
            return val
        # Unknown var — should have been caught by validator, but be defensive.
        missing.append(v)
        return m.group(0)
    return _PLACEHOLDER_RE.sub(repl, template), missing


def required_user_secrets(plan: dict, run_dir: Path) -> list[dict]:
    """Return the user_secrets_needed entries that don't yet have a value.

    Empty list = no user input needed; non-empty = executor must pause and
    surface a form to the user before running config_writes.
    """
    needed = plan.get("user_secrets_needed") or []
    provided = _load_user_secrets(run_dir)
    out = []
    for entry in needed:
        if not isinstance(entry, dict):
            continue
        var = entry.get("var_name")
        if not var:
            continue
        if not provided.get(var):  # missing or empty
            out.append(entry)
    return out


@traced_agent("Agent 2 SetupRunner · exec", phase=2)
def execute_plan(plan: dict, repo_dir: Path, state_path: Path,
                 services_dir: Path,
                 install_timeout: int = 600,
                 service_health_wait: float = 60.0) -> ExecutionState:
    """Run the plan synchronously. Returns final state.

    If `plan.user_secrets_needed` declares secrets the user hasn't yet filled
    (via the /provide_secrets endpoint), execute_plan EXITS EARLY with status
    "waiting_user_secrets" and the executor caller is expected to surface a
    form to the user. Re-invoke after secrets are provided.
    """
    log = agent_logger("agent2_setup")
    run_dir = state_path.parent
    log.info(f"execute_plan  user_secrets_needed={len(plan.get('user_secrets_needed') or [])} "
             f"config_writes={len(plan.get('config_writes') or [])} "
             f"install={len(plan.get('install_commands') or [])} "
             f"seed={len(plan.get('seed_commands') or [])} "
             f"services={len(plan.get('services') or [])}")
    state = ExecutionState(started_at=datetime.now(timezone.utc).isoformat())
    _persist(state, state_path)

    # Gate: any user_secrets_needed still missing? Pause and let user fill.
    waiting = required_user_secrets(plan, run_dir)
    if waiting:
        state.status = "waiting_user_secrets"
        state.waiting_for_secrets = waiting
        _persist(state, state_path)
        log.info(f"pausing: {len(waiting)} user secret(s) needed: "
                   f"{[w.get('var_name') for w in waiting]}")
        return state

    user_secrets = _load_user_secrets(run_dir)

    # 0. config_writes — render templates with parent env + user secrets
    cw_results, cw_err = _run_config_writes(plan, repo_dir, user_secrets)
    state.config_writes = cw_results
    _persist(state, state_path)
    if cw_err:
        state.status = "failed"
        state.error = cw_err
        _persist(state, state_path)
        return state

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
