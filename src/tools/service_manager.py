"""Background service lifecycle: spawn, health-check, kill, status.

Each service runs as a child of the web-server process. State is persisted to
`<run_dir>/services.json` so requests across the FastAPI process can see
current PIDs / status.

The host (NOT the LLM agent) calls into this module to execute approved plans.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from urllib.error import URLError
from urllib.request import urlopen

from loguru import logger

from ..observability.logger import agent_logger

ServiceStatus = Literal["pending", "starting", "healthy", "dead", "stopped", "timeout"]


@dataclass
class ServiceRecord:
    name: str
    command: str
    cwd: str
    port: int
    health_url: str
    pid: Optional[int] = None
    status: ServiceStatus = "pending"
    started_at: Optional[str] = None
    last_check: Optional[str] = None
    last_error: Optional[str] = None
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None


class ServiceManager:
    """Owns running subprocesses for a single run. Thread-safe."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = run_dir / "services.json"
        self.log_dir = run_dir / "service_logs"
        self.log_dir.mkdir(exist_ok=True)
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._records: dict[str, ServiceRecord] = self._load_records()

    def _load_records(self) -> dict[str, ServiceRecord]:
        if not self.state_file.exists():
            return {}
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            return {name: ServiceRecord(**rec) for name, rec in data.items()}
        except Exception:
            return {}

    def _save(self) -> None:
        data = {name: asdict(rec) for name, rec in self._records.items()}
        self.state_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list(self) -> list[ServiceRecord]:
        with self._lock:
            return list(self._records.values())

    def get(self, name: str) -> Optional[ServiceRecord]:
        return self._records.get(name)

    def start(self, name: str, command: str, cwd: Path, port: int,
              health_url: str, env: Optional[dict] = None) -> ServiceRecord:
        with self._lock:
            if name in self._procs and self._procs[name].poll() is None:
                agent_logger("agent2_setup").warning(
                    f"start: '{name}' already running PID {self._procs[name].pid}; skipping"
                )
                return self._records[name]

            stdout_path = self.log_dir / f"{name}.stdout.log"
            stderr_path = self.log_dir / f"{name}.stderr.log"

            full_env = os.environ.copy()
            if env:
                full_env.update(env)
            # PYTHONUNBUFFERED so live console reflects server activity
            full_env.setdefault("PYTHONUNBUFFERED", "1")
            # Force UTF-8 mode so target repo's read_text() etc. don't fail on
            # Windows-GBK default codepage (a common third-party-repo pitfall).
            full_env.setdefault("PYTHONUTF8", "1")
            full_env.setdefault("PYTHONIOENCODING", "utf-8")

            log = agent_logger("agent2_setup")
            log.info(f"start '{name}' → {command}  (cwd={cwd}, port={port})")

            try:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=str(cwd),
                    env=full_env,
                    stdout=stdout_path.open("ab"),
                    stderr=stderr_path.open("ab"),
                    creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                                   if os.name == "nt" else 0),
                )
            except Exception as e:
                rec = ServiceRecord(
                    name=name, command=command, cwd=str(cwd),
                    port=port, health_url=health_url,
                    status="dead", last_error=f"spawn failed: {e}",
                    last_check=datetime.now(timezone.utc).isoformat(),
                    stdout_path=str(stdout_path), stderr_path=str(stderr_path),
                )
                self._records[name] = rec
                self._save()
                raise

            rec = ServiceRecord(
                name=name, command=command, cwd=str(cwd),
                port=port, health_url=health_url,
                pid=proc.pid, status="starting",
                started_at=datetime.now(timezone.utc).isoformat(),
                stdout_path=str(stdout_path), stderr_path=str(stderr_path),
            )
            self._procs[name] = proc
            self._records[name] = rec
            self._save()
            return rec

    def health_check(self, name: str, max_wait_s: float = 60.0,
                     interval_s: float = 1.5) -> ServiceRecord:
        """Poll health_url until 200 / max_wait. Updates record in place."""
        rec = self._records.get(name)
        if rec is None:
            raise KeyError(name)
        log = agent_logger("agent2_setup")

        deadline = time.time() + max_wait_s
        last_err: Optional[str] = None
        while time.time() < deadline:
            proc = self._procs.get(name)
            if proc and proc.poll() is not None:
                exit_code = proc.returncode
                rec.status = "dead"
                rec.last_error = f"process exited with code {exit_code}"
                rec.last_check = datetime.now(timezone.utc).isoformat()
                self._save()
                log.error(f"health_check '{name}': process died (exit {exit_code})")
                return rec
            try:
                with urlopen(rec.health_url, timeout=2.0) as r:
                    if 200 <= r.status < 400:
                        rec.status = "healthy"
                        rec.last_error = None
                        rec.last_check = datetime.now(timezone.utc).isoformat()
                        self._save()
                        log.info(f"health_check '{name}': healthy ({rec.health_url})")
                        return rec
                    last_err = f"HTTP {r.status}"
            except URLError as e:
                last_err = f"{type(e).__name__}: {e}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(interval_s)

        rec.status = "timeout"
        rec.last_error = last_err or "no response"
        rec.last_check = datetime.now(timezone.utc).isoformat()
        self._save()
        log.warning(f"health_check '{name}': timeout after {max_wait_s}s ({last_err})")
        return rec

    def stop(self, name: str) -> None:
        with self._lock:
            proc = self._procs.get(name)
            log = agent_logger("agent2_setup")
            if proc is None:
                rec = self._records.get(name)
                if rec:
                    rec.status = "stopped"
                    self._save()
                return
            if proc.poll() is None:
                log.info(f"stop '{name}' (PID {proc.pid})")
                try:
                    if os.name == "nt":
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                        try:
                            proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            proc.terminate()
                            try:
                                proc.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                    else:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                except Exception as e:
                    log.warning(f"stop '{name}': {type(e).__name__}: {e}")
            self._procs.pop(name, None)
            rec = self._records.get(name)
            if rec:
                rec.status = "stopped"
                rec.last_check = datetime.now(timezone.utc).isoformat()
                self._save()

    def stop_all(self) -> None:
        for name in list(self._records.keys()):
            self.stop(name)

    def refresh_status(self) -> None:
        """Update each record by polling health_url + checking process."""
        for name, rec in self._records.items():
            if rec.status in ("stopped", "dead"):
                continue
            proc = self._procs.get(name)
            if proc and proc.poll() is not None:
                rec.status = "dead"
                rec.last_error = f"process exited (code {proc.returncode})"
                rec.last_check = datetime.now(timezone.utc).isoformat()
                continue
            try:
                with urlopen(rec.health_url, timeout=1.5) as r:
                    if 200 <= r.status < 400:
                        rec.status = "healthy"
                        rec.last_error = None
                    else:
                        rec.status = "starting"
                        rec.last_error = f"HTTP {r.status}"
            except Exception as e:
                if rec.status == "healthy":
                    rec.status = "starting"
                rec.last_error = f"{type(e).__name__}: {e}"
            rec.last_check = datetime.now(timezone.utc).isoformat()
        self._save()
