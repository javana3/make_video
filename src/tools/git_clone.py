"""Git clone with streaming progress.

`shell.run` buffers stdout/stderr, so we can't surface clone progress live
to the UI. This module runs `git clone --progress`, reads its output
line-by-line, parses the standard git progress format, and writes a JSON
snapshot to `progress_path` after every line so the UI can poll it.

Progress lines git emits (with `--progress`):
  remote: Enumerating objects: 234, done.
  remote: Counting objects: 100% (12/12), done.
  remote: Compressing objects:  85% (110/130)
  Receiving objects:  47% (110/234), 12.34 MiB | 1.23 MiB/s
  Resolving deltas: 100% (45/45), done.
"""
from __future__ import annotations
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Optional


# git's progress lines come in several shapes; capture what matters for UX.
_RECV_RE = re.compile(
    r"Receiving objects:\s+(\d+)%\s+\((\d+)/(\d+)\)"
    r"(?:,\s+([\d.]+)\s+([KMG]i?B))?"
    r"(?:\s*\|\s+([\d.]+)\s+([KMG]i?B/s))?"
)
_RESOLVE_RE = re.compile(r"Resolving deltas:\s+(\d+)%")
_COMPRESS_RE = re.compile(r"remote:\s+Compressing objects:\s+(\d+)%")
_ENUM_RE = re.compile(r"remote:\s+(?:Counting|Enumerating) objects:\s+(\d+)")


def _write(path: Path, **kv) -> None:
    kv["ts"] = time.time()
    try:
        path.write_text(json.dumps(kv, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # Best-effort: never crash the clone over progress write failure
        pass


def _parse_line(line: str) -> dict:
    """Return a partial dict of fields parsed from one git stderr line."""
    out: dict = {"last_line": line}
    m = _RECV_RE.search(line)
    if m:
        out["phase"] = "receiving"
        out["pct"] = int(m.group(1))
        out["objects_done"] = int(m.group(2))
        out["objects_total"] = int(m.group(3))
        if m.group(4) and m.group(5):
            out["size_text"] = f"{m.group(4)} {m.group(5)}"
        if m.group(6) and m.group(7):
            out["speed_text"] = f"{m.group(6)} {m.group(7)}"
        return out
    m = _RESOLVE_RE.search(line)
    if m:
        out["phase"] = "resolving"
        out["pct"] = int(m.group(1))
        return out
    m = _COMPRESS_RE.search(line)
    if m:
        out["phase"] = "compressing"
        out["pct"] = int(m.group(1))
        return out
    if _ENUM_RE.search(line):
        out["phase"] = "enumerating"
        return out
    if line.lower().startswith("cloning into"):
        out["phase"] = "starting"
    return out


def clone_with_progress(url: str,
                          dest: Path,
                          progress_path: Path,
                          timeout: float = 600,
                          on_line: Optional[callable] = None) -> tuple[int, str]:
    """Run `git clone --depth=1 --progress` with live progress writes.

    Returns (exit_code, tail_text). tail_text contains the last 4 KB of
    combined stderr/stdout for the error banner if exit != 0.
    """
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    _write(progress_path, phase="starting", url=url, elapsed=0)

    proc = subprocess.Popen(
        ["git", "clone", "--depth=1", "--progress", url, str(dest)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        bufsize=1,
    )

    tail: list[str] = []
    state: dict = {"phase": "starting"}

    try:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, ""):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            tail.append(line)
            if len(tail) > 200:
                tail = tail[-150:]

            parsed = _parse_line(line)
            state.update(parsed)
            state["elapsed"] = round(time.time() - started, 1)
            _write(progress_path, **state)

            if on_line:
                try:
                    on_line(line, state)
                except Exception:
                    pass

            if timeout and time.time() - started > timeout:
                proc.kill()
                state["phase"] = "timeout"
                _write(progress_path, **state)
                return -1, "\n".join(tail[-50:])
    finally:
        proc.wait()

    code = proc.returncode
    state["phase"] = "done" if code == 0 else "failed"
    state["exit_code"] = code
    state["elapsed"] = round(time.time() - started, 1)
    _write(progress_path, **state)

    return code, "\n".join(tail[-50:])
