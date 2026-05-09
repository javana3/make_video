"""Shell command runner.

Every shell call in the workflow MUST go through `run()`. Direct
`subprocess.run` is forbidden — it would bypass logging and tracing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


@dataclass
class ShellResult:
    cmd: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def run(cmd: list[str],
        cwd: Optional[Union[str, Path]] = None,
        timeout: Optional[float] = None,
        check: bool = False,
        env: Optional[dict] = None) -> ShellResult:
    """Run a shell command, log it, return structured result.

    Args:
        cmd: argv list (avoid shell=True style strings)
        cwd: working directory
        timeout: seconds; raises subprocess.TimeoutExpired on overshoot
        check: if True, raise CalledProcessError on non-zero exit
        env: full environment dict (None = inherit)
    """
    tracer = trace.get_tracer("video-workflow.shell")
    log = logger.bind(agent="shell")

    # On Windows, subprocess does not honor PATHEXT for unqualified names
    # (e.g. "npm" → "npm.cmd"). Resolve via shutil.which so .cmd/.bat work.
    if os.name == "nt" and cmd:
        first = str(cmd[0])
        if not Path(first).is_absolute() and not Path(first).exists():
            resolved = shutil.which(first)
            if resolved:
                cmd = [resolved, *cmd[1:]]

    cmd_str = " ".join(str(c) for c in cmd)

    with tracer.start_as_current_span("shell.run") as span:
        span.set_attribute("shell.cmd", cmd_str[:1000])
        if cwd:
            span.set_attribute("shell.cwd", str(cwd))
        if timeout:
            span.set_attribute("shell.timeout", timeout)

        log.info(f"$ {cmd_str}")
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                timeout=timeout,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            duration = time.time() - t0
            log.error(f"timeout after {duration:.1f}s: {cmd_str}")
            span.set_status(Status(StatusCode.ERROR, "timeout"))
            raise

        duration = time.time() - t0
        result = ShellResult(
            cmd=list(map(str, cmd)),
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration=duration,
        )

        span.set_attribute("shell.exit_code", proc.returncode)
        span.set_attribute("shell.duration_s", duration)
        span.set_attribute("shell.stdout_len", len(proc.stdout))
        if proc.stderr:
            span.set_attribute("shell.stderr_tail", proc.stderr[-500:])
        if proc.returncode != 0:
            span.set_status(Status(StatusCode.ERROR, f"exit {proc.returncode}"))

        log.bind(
            exit_code=proc.returncode,
            duration=round(duration, 3),
            stdout_len=len(proc.stdout),
            stderr_tail=proc.stderr[-500:] if proc.stderr else "",
        ).info(f"exit={proc.returncode} ({duration:.2f}s)")

        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, proc.stdout, proc.stderr
            )
        return result
