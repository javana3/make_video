"""Run npm install + npx remotion render for a generated Remotion project."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from loguru import logger

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger
from .shell import run as shell_run


def _node_env() -> dict:
    """Env tweaks: avoid GPU-related failures in headless render."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def npm_install(remotion_dir: Path, timeout: int = 600) -> None:
    log = agent_logger("agent3_remotion")
    if (remotion_dir / "node_modules").exists():
        log.info("node_modules exists; skipping npm install")
        return
    log.info(f"npm install in {remotion_dir} (this may take 30s-3min) ...")
    proc = shell_run(
        ["cmd", "/c", "npm install --no-fund --no-audit"],
        cwd=remotion_dir, timeout=timeout, env=_node_env(),
    )
    if proc.exit_code != 0:
        raise RuntimeError(f"npm install failed: {proc.stderr[-2000:]}")
    log.info("npm install ok")


def _find_chrome() -> Optional[str]:
    """Locate a Chromium binary already on the box (Playwright / system Chrome)
    so Remotion doesn't have to download from storage.googleapis.com (blocked
    in some networks)."""
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright" / "chromium-1217" / "chrome-win64" / "chrome.exe",
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    # Glob for any playwright chromium (version may differ)
    pw_root = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    if pw_root.exists():
        for d in pw_root.glob("chromium-*"):
            for ex in d.rglob("chrome.exe"):
                candidates.insert(0, ex)
                break
    for c in candidates:
        if c.exists():
            return str(c)
    return None


@traced_agent("Agent 3 RemotionComposer · render", phase=3)
def render(remotion_dir: Path,
           output_path: Path,
           composition_id: str = "MyVideo",
           timeout: int = 1200,
           gl: str = "swiftshader") -> dict:
    """Run `npx remotion render` to produce mp4. Returns summary dict."""
    log = agent_logger("agent3_remotion")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    chrome_path = _find_chrome()
    # No quotes — Remotion CLI takes the value verbatim and compares to fs path.
    # Path has no spaces on standard Windows install paths we look at.
    chrome_arg = f'--browser-executable={chrome_path} ' if chrome_path else ""
    if chrome_path:
        log.info(f"reusing chrome: {chrome_path}")

    # Don't wrap output_path in quotes — cmd.exe + npx pass them through
    # literally, breaking Remotion's filename suffix check.
    out_str = str(output_path.resolve())
    cmd = (
        f'npx remotion render src/index.ts {composition_id} '
        f'{out_str} '
        f'--bundle-cache=false '
        f'{chrome_arg}--gl={gl} --concurrency=2 --log=info'
    )
    log.info(f"render: {cmd}  (cwd={remotion_dir})")
    proc = shell_run(["cmd", "/c", cmd], cwd=remotion_dir,
                     timeout=timeout, env=_node_env())
    if proc.exit_code != 0:
        tail = (proc.stderr or "")[-3000:] + "\n--- stdout ---\n" + (proc.stdout or "")[-2000:]
        raise RuntimeError(f"remotion render failed (exit {proc.exit_code}):\n{tail}")

    if not output_path.exists():
        raise RuntimeError(f"render reported success but no file at {output_path}")
    size = output_path.stat().st_size
    log.info(f"render done: {output_path} ({size}B)")
    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="agent3_remotion",
                 name="v1_video", path=str(output_path),
                 size_bytes=size, composition_id=composition_id)
    return {"output_path": str(output_path), "size_bytes": size}
