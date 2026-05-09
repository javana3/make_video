"""Screen-record a specific window via `ffmpeg -f gdigrab`.

Output: H.264 / yuv420p MP4 in `<run_dir>/recordings/`. Browser-playable; the
UI can embed via <video> directly.

Progress:
- `<run_dir>/recording_state.json` updated each second with status / elapsed_s /
  remaining_s. Polled by the web UI.
"""
from __future__ import annotations

import json
import subprocess
import threading
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
from .window_enum import get_window_rect, set_topmost

RecordingStatus = Literal["pending", "recording", "done", "failed", "cancelled"]


@dataclass
class RecordingState:
    output_path: str
    window_title: str
    duration_s: float
    framerate: int
    status: RecordingStatus = "pending"
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    elapsed_s: float = 0.0
    remaining_s: float = 0.0
    file_size_bytes: int = 0
    error: Optional[str] = None
    ffprobe: Optional[dict] = None


def _persist(state: RecordingState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@traced_agent("Agent 2 SetupRunner · record", phase=2)
def record_window(window_title: str,
                  duration_s: float,
                  output_path: Path,
                  state_path: Path,
                  framerate: int = 30,
                  preset: str = "veryfast",
                  crf: int = 22) -> RecordingState:
    """Synchronous: blocks for ~duration_s, returns RecordingState."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state = RecordingState(
        output_path=str(output_path),
        window_title=window_title,
        duration_s=duration_s,
        framerate=framerate,
    )

    log = agent_logger("agent2_setup")
    log.info(f"record_window  title={window_title!r}  duration={duration_s}s  → {output_path.name}")

    # Use rect-based capture to avoid ffmpeg gdigrab's ANSI title= bug
    # (which can't match titles with em dash / unicode on cp936 hosts).
    rect = get_window_rect(window_title)
    if rect is None:
        state.status = "failed"
        state.error = (f"Cannot locate window {window_title!r} via FindWindowW + "
                       f"GetWindowRect. Window may be minimized / closed / title "
                       f"changed. Refresh window list and try again.")
        state.ended_at = datetime.now(timezone.utc).isoformat()
        _persist(state, state_path)
        log.error(state.error)
        return state

    left, top, width, height = rect
    log.info(f"window rect: ({left},{top}) {width}x{height}")

    cmd = [
        ffmpeg(),
        "-y",
        "-f", "gdigrab",
        "-framerate", str(framerate),
        "-offset_x", str(left),
        "-offset_y", str(top),
        "-video_size", f"{width}x{height}",
        "-i", "desktop",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-t", str(duration_s),
        str(output_path),
    ]

    state.status = "recording"
    state.started_at = datetime.now(timezone.utc).isoformat()
    _persist(state, state_path)

    # Pin the target window as always-on-top so other windows can't occlude
    # the capture during the recording window. Unset on exit.
    set_topmost(window_title, on=True)
    log.info(f"set_topmost({window_title!r}) → on")

    proc: Optional[subprocess.Popen] = None
    stop_progress = threading.Event()

    def progress_loop():
        t0 = time.monotonic()
        while not stop_progress.is_set():
            elapsed = time.monotonic() - t0
            state.elapsed_s = round(elapsed, 1)
            state.remaining_s = max(0.0, round(duration_s - elapsed, 1))
            if output_path.exists():
                try:
                    state.file_size_bytes = output_path.stat().st_size
                except Exception:
                    pass
            _persist(state, state_path)
            if elapsed >= duration_s + 5:
                break
            time.sleep(1.0)

    progress_thread = threading.Thread(target=progress_loop, daemon=True)
    progress_thread.start()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        # Wait up to duration + 30s slack
        try:
            stdout, stderr = proc.communicate(timeout=duration_s + 30)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            state.status = "failed"
            state.error = f"ffmpeg timed out after {duration_s + 30:.0f}s"
            stop_progress.set()
            _persist(state, state_path)
            return state

        if proc.returncode != 0:
            state.status = "failed"
            tail = (stderr or "")[-2000:]
            state.error = f"ffmpeg exit {proc.returncode}\n{tail}"
            log.error(f"record_window failed: {tail[-400:]}")
            stop_progress.set()
            _persist(state, state_path)
            return state

    except Exception as e:
        state.status = "failed"
        state.error = f"{type(e).__name__}: {e}"
        log.exception("record_window exception")
        stop_progress.set()
        _persist(state, state_path)
        return state
    finally:
        stop_progress.set()
        progress_thread.join(timeout=2)
        # Unpin (best effort)
        set_topmost(window_title, on=False)

    state.ended_at = datetime.now(timezone.utc).isoformat()
    state.status = "done"
    state.elapsed_s = duration_s
    state.remaining_s = 0
    if output_path.exists():
        state.file_size_bytes = output_path.stat().st_size

    # ffprobe verification
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
        log.warning(f"ffprobe failed (non-fatal): {e}")

    _persist(state, state_path)
    log.info(f"record_window done: {output_path} ({state.file_size_bytes} bytes)")

    # Emit a lifecycle event keyed by the recording role inferred from the
    # filename stem (test → test_recording; anything else → final_recording).
    bus = get_run_context().get("event_bus")
    if bus is not None:
        stem = output_path.stem.lower()
        asset_name = "test_recording" if "test" in stem else "final_recording"
        bus.emit("asset_verified", agent="agent2_setup",
                 name=asset_name, path=str(output_path),
                 duration_s=duration_s, framerate=framerate,
                 size_bytes=state.file_size_bytes,
                 window_title=window_title)
    return state
