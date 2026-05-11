"""Service log readers — give the Demo Driver visibility into backend logs.

Without these, the agent can only see what the UI surfaces; backend Python
tracebacks, FastAPI 500 reasons, ComfyUI workflow failures, etc. live in
`workspace/<project>/runs/<run_id>/service_logs/<name>.{stdout,stderr}.log`
and the agent has no way to read them. These tools close that gap.

Tools provided here are SAFE READ-ONLY:
- list_services: parses setup_exec.json to list current services + log paths
- tail_service_log: reads last N lines of a service's stdout/stderr/both

Designed for the Demo Driver but reusable by any future debug agent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


def _service_records(run_dir: Path) -> list[dict]:
    exec_path = run_dir / "setup_exec.json"
    if not exec_path.exists():
        return []
    try:
        d = json.loads(exec_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return d.get("service_records") or []


def list_services(run_dir: Path) -> list[dict]:
    """Return a compact list of services running in this run.

    Each entry: name / port / status / health_url / pid / stderr_path /
    stdout_path / last_error. The agent uses this to discover what to
    tail (or to check whether a service is actually up).
    """
    out = []
    for rec in _service_records(run_dir):
        out.append({
            "name": rec.get("name"),
            "port": rec.get("port"),
            "status": rec.get("status"),
            "health_url": rec.get("health_url"),
            "pid": rec.get("pid"),
            "started_at": rec.get("started_at"),
            "last_check": rec.get("last_check"),
            "last_error": rec.get("last_error"),
            "stdout_path": rec.get("stdout_path"),
            "stderr_path": rec.get("stderr_path"),
        })
    return out


def tail_service_log(run_dir: Path, name: str,
                     kind: Literal["stderr", "stdout", "both"] = "stderr",
                     lines: int = 100,
                     max_chars: int = 16000) -> str:
    """Read last `lines` lines of a service's log.

    `kind`:
        - 'stderr' (default — where Python tracebacks land)
        - 'stdout'
        - 'both' — concatenated with `--- STDOUT ---` and `--- STDERR ---` markers
    """
    records = _service_records(run_dir)
    rec = next((r for r in records if r.get("name") == name), None)
    if rec is None:
        names = [r.get("name") for r in records]
        return f"ERROR: no service named {name!r}. Available: {names}"

    def _tail(path_str: str | None, n: int) -> str:
        if not path_str:
            return "<no log path>"
        p = Path(path_str)
        if not p.exists():
            return f"<log file does not exist: {p}>"
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"<read failed: {type(e).__name__}: {e}>"
        log_lines = text.splitlines()
        return "\n".join(log_lines[-n:])

    parts = []
    if kind in ("stderr", "both"):
        body = _tail(rec.get("stderr_path"), lines)
        if kind == "both":
            parts.append("--- STDERR ---")
        parts.append(body)
    if kind in ("stdout", "both"):
        body = _tail(rec.get("stdout_path"), lines)
        if kind == "both":
            parts.append("--- STDOUT ---")
        parts.append(body)
    out = "\n".join(parts)
    if len(out) > max_chars:
        out = out[: max_chars - 60] + f"\n[...truncated to {max_chars} chars]"
    return out
