"""Phase 2B-record · Execute demo_script while recording.

Reads demo_script.json (from `src/agents/demo_planner.py`), opens the URL
in headless Chromium with screen recording on, then steps through each
action — clicking, typing, scrolling, etc. — while capturing real
wall-clock timestamps for each step. The output `demo_timings.json` is
the source of truth for caption timing in `src/tools/captions.py`.

Errors per step are captured but DO NOT abort the recording; the row
gets `status="error"` and we move on, so the recording is still usable
even if one selector goes stale.
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger
from .ffbin import ffmpeg, ffprobe
from .shell import run as shell_run


def _exec_step(page, step: dict, log) -> dict:
    """Run one step; return {status, error?} dict."""
    action = step.get("action")
    target = step.get("target")
    try:
        if action == "navigate":
            url = step.get("url") or target
            page.goto(url, wait_until="domcontentloaded", timeout=20_000)
        elif action == "click":
            page.locator(target).first.click(timeout=8_000)
        elif action == "type":
            page.locator(target).first.fill(step.get("text", ""), timeout=8_000)
        elif action == "press_key":
            page.keyboard.press(step.get("key", "Enter"))
        elif action == "hover":
            page.locator(target).first.hover(timeout=8_000)
        elif action == "scroll":
            y = float(step.get("x", 0))  # historical: x means scroll-y in script
            y = float(step.get("y", y))
            if target:
                page.locator(target).first.evaluate(f"el => el.scrollBy(0, {y})")
            else:
                page.evaluate(f"() => window.scrollBy(0, {y})")
        elif action == "wait":
            # pure dwell, handled by wait_after_s after this returns
            pass
        else:
            return {"status": "error", "error": f"unknown action {action!r}"}
        return {"status": "ok"}
    except Exception as e:
        log.warning(f"step {step.get('id','?')} {action} failed: {type(e).__name__}: {e}")
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


@traced_agent("Phase 2B · Demo Executor", phase=2)
def execute_demo(demo_script_path: Path,
                 output_video: Path,
                 timings_path: Path,
                 width: int = 1920,
                 height: int = 1080,
                 headless: bool = True) -> dict:
    """Run demo_script while recording. Writes mp4 + timings JSON."""
    log = agent_logger("demo_executor")
    plan = json.loads(demo_script_path.read_text(encoding="utf-8"))
    steps = plan.get("steps", [])
    if not steps:
        raise ValueError("demo_script has no steps")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    record_dir = output_video.parent / "_pw_temp"
    record_dir.mkdir(exist_ok=True)
    for f in record_dir.glob("*.webm"):
        try: f.unlink()
        except Exception: pass

    log.info(f"executing {len(steps)} steps  viewport={width}x{height}  headless={headless}")
    timings: list[dict] = []
    raw_webm: Optional[Path] = None

    started_iso = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless,
                                     args=["--no-proxy-server"])
        ctx = browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(record_dir),
            record_video_size={"width": width, "height": height},
        )
        page = ctx.new_page()

        for step in steps:
            t_start = time.monotonic() - t0
            log.info(f"  [{t_start:5.1f}s] {step.get('id','?')} {step.get('action')} → {step.get('target', step.get('url',''))[:80]}")
            res = _exec_step(page, step, log)
            wait_after = float(step.get("wait_after_s", 1.5))
            page.wait_for_timeout(int(wait_after * 1000))
            t_end = time.monotonic() - t0
            timings.append({
                "id": step.get("id"),
                "action": step.get("action"),
                "target": step.get("target"),
                "t_start": round(t_start, 3),
                "t_end": round(t_end, 3),
                "duration_s": round(t_end - t_start, 3),
                "wait_after_s": wait_after,
                "caption_zh": step.get("caption_zh", ""),
                "caption_en": step.get("caption_en", ""),
                "status": res["status"],
                "error": res.get("error"),
            })

        ctx.close()
        # webm finalized; pick most recent
        webms = sorted(record_dir.glob("*.webm"), key=lambda p: p.stat().st_mtime)
        if webms:
            raw_webm = webms[-1]
        browser.close()

    if raw_webm is None or not raw_webm.exists():
        raise RuntimeError("Playwright produced no webm")

    log.info(f"transcoding {raw_webm.name} → {output_video.name}")
    cmd = [
        ffmpeg(), "-y",
        "-i", str(raw_webm),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_video),
    ]
    shell_run(cmd, check=True, timeout=120)

    # Probe final mp4
    probe = shell_run([
        ffprobe(), "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(output_video),
    ], check=True)
    data = json.loads(probe.stdout)
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    fmt = data.get("format", {})

    finished_iso = datetime.now(timezone.utc).isoformat()
    elapsed = time.monotonic() - t0

    timings_payload = {
        "demo_script_path": str(demo_script_path),
        "video_path": str(output_video),
        "viewport": {"width": width, "height": height},
        "started_at": started_iso,
        "finished_at": finished_iso,
        "elapsed_s": round(elapsed, 1),
        "video_duration_s": float(fmt.get("duration", 0)),
        "video_size_bytes": int(fmt.get("size", 0)),
        "video_codec": v.get("codec_name"),
        "n_steps": len(timings),
        "n_errors": sum(1 for t in timings if t["status"] == "error"),
        "steps": timings,
    }
    timings_path.parent.mkdir(parents=True, exist_ok=True)
    timings_path.write_text(json.dumps(timings_payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    # Cleanup raw webm cache
    try:
        shutil.rmtree(record_dir)
    except Exception:
        pass

    log.info(f"  ✓ {output_video.name}  dur={timings_payload['video_duration_s']:.1f}s  "
             f"errors={timings_payload['n_errors']}/{len(timings)}")

    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="demo_executor",
                 name="demo_recording", path=str(output_video),
                 video_duration_s=timings_payload["video_duration_s"],
                 n_steps=len(timings),
                 n_errors=timings_payload["n_errors"])

    return timings_payload
