"""OpenDesign daemon lifecycle helper.

Discovers a running daemon by reading its stdout log; if none, spawns
`pnpm tools-dev run web` and waits for the "Open Design dev server ready"
banner. Returns OpenDesignEndpoint with both URLs.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from loguru import logger

from .opendesign_client import OpenDesignEndpoint, health_check


_DEFAULT_REPO = Path(r"C:\Users\dfgfd\Downloads\open_design_index")
_DEFAULT_LOG = Path(tempfile.gettempdir()) / "opendesign_dev.log"
_BANNER = re.compile(r"Open Design dev server ready", re.IGNORECASE)


def _read_endpoint_from_log(log_path: Path) -> OpenDesignEndpoint | None:
    if not log_path.exists():
        return None
    try:
        return OpenDesignEndpoint.from_dev_log(log_path.read_text(encoding="utf-8"))
    except RuntimeError:
        return None


def _alive(endpoint: OpenDesignEndpoint) -> bool:
    try:
        return bool(health_check(endpoint.daemon_url, timeout=2.0).get("ok"))
    except Exception:
        return False


def ensure_daemon(repo_dir: Path = _DEFAULT_REPO,
                  log_path: Path = _DEFAULT_LOG,
                  startup_timeout_s: float = 90.0) -> OpenDesignEndpoint:
    """Return a live OpenDesignEndpoint. Starts the dev server if needed."""
    from ..observability.logger import agent_logger
    log = agent_logger("agent6_opendesigner")

    existing = _read_endpoint_from_log(log_path)
    if existing and _alive(existing):
        log.info(f"reusing existing daemon: {existing.daemon_url}")
        return existing

    log.info(f"starting OpenDesign dev server in {repo_dir}")
    if not repo_dir.exists():
        raise RuntimeError(f"OpenDesign repo not found at {repo_dir}")

    # Truncate the log so we can detect the new banner unambiguously
    log_path.write_text("", encoding="utf-8")

    env = os.environ.copy()
    home_corepack = Path.home() / "corepack-bin"
    if home_corepack.exists():
        env["PATH"] = f"{home_corepack};{env.get('PATH','')}"

    log_fp = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        ["pnpm", "tools-dev", "run", "web"],
        cwd=str(repo_dir), env=env,
        stdout=log_fp, stderr=subprocess.STDOUT,
        shell=True,
    )

    deadline = time.monotonic() + startup_timeout_s
    while time.monotonic() < deadline:
        time.sleep(1.0)
        # Detect "Open Design dev server ready"
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if _BANNER.search(text):
            try:
                ep = OpenDesignEndpoint.from_dev_log(text)
            except RuntimeError:
                continue
            if _alive(ep):
                log.info(f"daemon ready: {ep.daemon_url}")
                return ep
        if proc.poll() is not None:
            raise RuntimeError(
                f"pnpm tools-dev run web exited prematurely (code={proc.returncode}); "
                f"see {log_path}"
            )

    raise RuntimeError(
        f"OpenDesign dev server did not announce ready within {startup_timeout_s}s; "
        f"see {log_path}"
    )
