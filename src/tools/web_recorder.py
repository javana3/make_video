"""Browser-internal video recording via Playwright.

Records the page render directly via Chromium's DevTools Protocol — bypasses
screen capture entirely. Result: clean video of just the page, regardless of
what other windows / apps the user is using during recording.

Use for any URL-driven web project. For desktop/native apps fall back to
`recorder.record_window` (gdigrab).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from loguru import logger

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger
from .ffbin import ffmpeg, ffprobe
from .shell import run as shell_run

WebRecStatus = Literal["pending", "recording", "done", "failed"]


@dataclass
class WebRecState:
    output_path: str
    url: str
    duration_s: float
    width: int
    height: int
    status: WebRecStatus = "pending"
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    elapsed_s: float = 0.0
    file_size_bytes: int = 0
    error: Optional[str] = None
    ffprobe: Optional[dict] = None


def _persist(state: WebRecState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2),
                    encoding="utf-8")


@traced_agent("Agent 2 SetupRunner · web-record", phase=2)
def record_url(url: str,
               duration_s: float,
               output_path: Path,
               state_path: Path,
               width: int = 1280,
               height: int = 800,
               headless: bool = True,
               wait_load_s: float = 2.0) -> WebRecState:
    """Open `url` in Chromium, record `duration_s` seconds, save to `output_path`.

    Playwright records as .webm; if `output_path` ends with `.mp4` we transcode
    via ffmpeg (also lets us add faststart for streaming).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = WebRecState(
        output_path=str(output_path),
        url=url,
        duration_s=duration_s,
        width=width,
        height=height,
    )
    log = agent_logger("agent2_setup")
    log.info(f"record_url  url={url}  duration={duration_s}s  size={width}x{height}  headless={headless}")

    record_dir = output_path.parent / "_pw_temp"
    record_dir.mkdir(exist_ok=True)
    # Wipe stale webm files so we pick up the new one cleanly
    for f in record_dir.glob("*.webm"):
        try:
            f.unlink()
        except Exception:
            pass

    state.status = "recording"
    state.started_at = datetime.now(timezone.utc).isoformat()
    _persist(state, state_path)

    raw_webm: Optional[Path] = None
    t0 = time.monotonic()

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            ctx = browser.new_context(
                viewport={"width": width, "height": height},
                record_video_dir=str(record_dir),
                record_video_size={"width": width, "height": height},
            )
            page = ctx.new_page()
            log.info(f"page.goto({url}) ...")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
            except Exception as e:
                log.warning(f"goto reported {e}; continuing anyway")
            log.info(f"page loaded; waiting {wait_load_s}s for JS bootstrap")
            page.wait_for_timeout(int(wait_load_s * 1000))

            # Update progress while waiting
            tick = 0.5
            n = max(1, int(duration_s / tick))
            for i in range(n):
                page.wait_for_timeout(int(tick * 1000))
                state.elapsed_s = round(time.monotonic() - t0, 1)
                if i % 4 == 0:  # persist every 2s, not every tick
                    _persist(state, state_path)

            log.info("closing context (this finalizes the webm) ...")
            ctx.close()
            browser.close()
    except Exception as e:
        state.status = "failed"
        state.error = f"{type(e).__name__}: {e}"
        state.ended_at = datetime.now(timezone.utc).isoformat()
        _persist(state, state_path)
        log.exception("record_url failed")
        return state

    # Find the produced webm
    webm_candidates = sorted(record_dir.glob("*.webm"),
                             key=lambda f: f.stat().st_mtime, reverse=True)
    if not webm_candidates:
        state.status = "failed"
        state.error = "no .webm produced by Playwright"
        state.ended_at = datetime.now(timezone.utc).isoformat()
        _persist(state, state_path)
        return state
    raw_webm = webm_candidates[0]
    log.info(f"raw webm: {raw_webm} ({raw_webm.stat().st_size}B)")

    # Transcode to mp4 if requested
    if output_path.suffix.lower() == ".mp4":
        log.info(f"transcoding webm → mp4 ...")
        try:
            shell_run([
                ffmpeg(), "-y", "-i", str(raw_webm),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(output_path),
            ], check=True, timeout=120)
        except Exception as e:
            state.status = "failed"
            state.error = f"transcode failed: {e}"
            state.ended_at = datetime.now(timezone.utc).isoformat()
            _persist(state, state_path)
            return state
    else:
        # rename / move webm to output
        if output_path.exists():
            output_path.unlink()
        raw_webm.replace(output_path)

    # Cleanup temp dir
    try:
        for f in record_dir.glob("*"):
            f.unlink()
        record_dir.rmdir()
    except Exception:
        pass

    state.ended_at = datetime.now(timezone.utc).isoformat()
    state.status = "done"
    state.file_size_bytes = output_path.stat().st_size
    state.elapsed_s = round(time.monotonic() - t0, 1)

    # ffprobe
    try:
        probe = shell_run([
            ffprobe(), "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", str(output_path),
        ], check=True)
        data = json.loads(probe.stdout)
        fmt = data.get("format", {})
        v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
        state.ffprobe = {
            "duration": float(fmt.get("duration", 0)),
            "size_bytes": int(fmt.get("size", 0)),
            "video_codec": v.get("codec_name") if v else None,
            "width": int(v.get("width", 0)) if v else 0,
            "height": int(v.get("height", 0)) if v else 0,
            "fps": v.get("r_frame_rate") if v else None,
        }
    except Exception as e:
        log.warning(f"ffprobe failed: {e}")

    _persist(state, state_path)
    log.info(f"done: {output_path} ({state.file_size_bytes}B)")

    bus = get_run_context().get("event_bus")
    if bus is not None:
        stem = output_path.stem.lower()
        asset_name = "test_recording" if "test" in stem else "final_recording"
        bus.emit("asset_verified", agent="agent2_setup",
                 name=asset_name, path=str(output_path),
                 url=url, duration_s=duration_s,
                 size_bytes=state.file_size_bytes,
                 width=width, height=height)
    return state
