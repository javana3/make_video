"""Phase 2B-cli · Generic CLI / TUI terminal recorder.

For projects that don't have a web service to point a browser at:
  1. Spawn the project's command (`python main.py` etc.) via subprocess.
  2. Stream stdout+stderr in a background thread, timestamping every chunk.
  3. Feed the bytes through `pyte` to emulate an 80×30 terminal (handles
     ANSI escapes, cursor moves, scrolling, colours).
  4. Snapshot the terminal grid at fixed FPS, render each snapshot with
     PIL (dark background + JetBrains Mono + ANSI palette).
  5. ffmpeg image sequence → mp4 (h264 + faststart).

This produces a real "watching the program run" video, suitable as the
recording.test.mp4 input to Phase 3 cutting plan.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

import pyte
from PIL import Image, ImageDraw, ImageFont

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger
from .ffbin import ffmpeg, ffprobe
from .shell import run as shell_run


# 8-color ANSI palette (foreground)
_ANSI_FG = {
    "default":  "#e2e8f0",
    "black":    "#1e293b",
    "red":      "#ef4444",
    "green":    "#22c55e",
    "yellow":   "#eab308",
    "blue":     "#3b82f6",
    "magenta":  "#a855f7",
    "cyan":     "#06b6d4",
    "white":    "#f1f5f9",
    "brightblack":   "#475569",
    "brightred":     "#f87171",
    "brightgreen":   "#4ade80",
    "brightyellow":  "#facc15",
    "brightblue":    "#60a5fa",
    "brightmagenta": "#c084fc",
    "brightcyan":    "#22d3ee",
    "brightwhite":   "#ffffff",
}
_BG_COLOR = "#0a0f1a"


def _find_mono_font() -> Optional[str]:
    """Prefer a CJK-capable monospace font so Chinese chars render properly.

    PIL's ImageFont doesn't support font fallback chains, so we need a single
    font that covers both ASCII and CJK. Consolas/Courier render Chinese as
    tofu boxes; SimSun/SimHei (Windows system fonts) cover both with the
    standard CJK 2× ASCII width that pyte already emits.
    """
    candidates = [
        # Windows: CJK-capable fonts (preferred for projects with Chinese output)
        r"C:\Windows\Fonts\simsun.ttc",     # SimSun 宋体 — CJK + ASCII, true monowidth
        r"C:\Windows\Fonts\simhei.ttf",     # SimHei 黑体 — CJK + ASCII
        # Windows: ASCII-only monospace fallback
        r"C:\Windows\Fonts\consola.ttf",    # Consolas
        r"C:\Windows\Fonts\cour.ttf",       # Courier New
        # Linux / macOS
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/Library/Fonts/Menlo.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def _spawn_and_capture(command: list[str], cwd: Path, env: Optional[dict],
                       duration_s: float, log) -> list[tuple[float, bytes]]:
    """Run command for `duration_s`, return list of (rel_ts_seconds, bytes_chunk)."""
    log.info(f"spawn: {command}  cwd={cwd}")
    chunks: list[tuple[float, bytes]] = []
    t0 = time.monotonic()

    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    def reader():
        try:
            while True:
                chunk = proc.stdout.read(4096) if proc.stdout else b""
                if not chunk:
                    break
                chunks.append((time.monotonic() - t0, chunk))
        except Exception as e:
            log.warning(f"reader err: {e}")

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    deadline = t0 + duration_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.05)

    # Stop the process if still running
    if proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        except Exception:
            pass
    th.join(timeout=2)

    log.info(f"captured {len(chunks)} chunks  exit={proc.returncode}")
    return chunks


def _replay_to_grids(chunks: list[tuple[float, bytes]],
                     fps: int, total_duration_s: float,
                     cols: int = 100, rows: int = 30) -> list[list[list[pyte.screens.Char]]]:
    """Replay chunks through a pyte screen + snapshot at each frame.

    Returns list of grid snapshots, length == fps * total_duration_s.
    """
    screen = pyte.Screen(cols, rows)
    stream = pyte.Stream(screen)
    frames: list[list[list[pyte.screens.Char]]] = []
    n_frames = int(fps * total_duration_s)
    chunk_idx = 0
    for f in range(n_frames):
        frame_t = (f + 1) / fps
        # Feed all chunks whose timestamp <= frame_t
        while chunk_idx < len(chunks) and chunks[chunk_idx][0] <= frame_t:
            try:
                stream.feed(chunks[chunk_idx][1].decode("utf-8", errors="replace"))
            except Exception:
                pass
            chunk_idx += 1
        # Snapshot current grid
        snapshot = []
        for row in screen.buffer.values():
            row_chars = []
            for col in range(cols):
                ch = row.get(col)
                if ch is None:
                    row_chars.append(pyte.screens.Char(" "))
                else:
                    row_chars.append(ch)
            snapshot.append(row_chars)
        frames.append(snapshot)
    return frames


def _render_frame(grid, font, char_w, char_h, width, height, padding=12) -> Image.Image:
    img = Image.new("RGB", (width, height), _BG_COLOR)
    d = ImageDraw.Draw(img)
    for r, row in enumerate(grid):
        if r >= 30:
            break
        # Combine consecutive chars with same fg color into spans for fewer draw calls
        col_idx = 0
        while col_idx < len(row):
            char = row[col_idx]
            fg = _ANSI_FG.get(char.fg, _ANSI_FG["default"]) if char.fg else _ANSI_FG["default"]
            # Find run of same fg
            text = char.data
            j = col_idx + 1
            while j < len(row):
                nxt = row[j]
                nfg = _ANSI_FG.get(nxt.fg, _ANSI_FG["default"]) if nxt.fg else _ANSI_FG["default"]
                if nfg != fg:
                    break
                text += nxt.data
                j += 1
            x = padding + col_idx * char_w
            y = padding + r * char_h
            d.text((x, y), text, fill=fg, font=font)
            col_idx = j
    return img


@traced_agent("Phase 2B · CLI Recorder", phase=2)
def record_cli(command: list[str],
                cwd: Path,
                output_path: Path,
                duration_s: float = 30.0,
                fps: int = 15,
                width: int = 1920,
                height: int = 1080,
                env: Optional[dict] = None,
                font_size: int = 22,
                cols: int = 100,
                rows: int = 30) -> dict:
    """Record a CLI/TUI program's stdout into an mp4 by emulating a terminal.

    Returns metadata dict (output_path, duration_s, n_chunks, exit_code, ...).
    """
    log = agent_logger("cli_recorder")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Run + capture
    log.info(f"M2b-cli record: {' '.join(command)} for ~{duration_s}s")
    chunks = _spawn_and_capture(command, cwd, env, duration_s, log)
    n_bytes = sum(len(c) for _, c in chunks)
    log.info(f"  captured {n_bytes} bytes in {len(chunks)} chunks")

    # 2. Replay through pyte → grid snapshots
    grids = _replay_to_grids(chunks, fps=fps, total_duration_s=duration_s,
                              cols=cols, rows=rows)
    log.info(f"  {len(grids)} grid frames @ {fps}fps × {cols}×{rows}")

    # 3. Render each grid → PNG
    font_path = _find_mono_font()
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    # Measure char width
    bbox = font.getbbox("M")
    char_w = bbox[2] - bbox[0]
    char_h = max(font_size + 6, bbox[3] - bbox[1] + 6)

    frame_dir = output_path.parent / "_cli_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir()

    log.info(f"  rendering {len(grids)} PNG frames in {frame_dir}")
    for i, grid in enumerate(grids):
        img = _render_frame(grid, font, char_w, char_h, width, height)
        img.save(frame_dir / f"f_{i:05d}.png", optimize=False)

    # 4. ffmpeg image seq → mp4
    log.info(f"  ffmpeg compose → {output_path.name}")
    cmd = [
        ffmpeg(), "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "f_%05d.png"),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    shell_run(cmd, check=True, timeout=180)

    # 5. Probe + cleanup
    probe = shell_run([ffprobe(), "-v", "quiet", "-print_format", "json",
                        "-show_streams", "-show_format", str(output_path)], check=True)
    pdata = json.loads(probe.stdout)
    v = next((s for s in pdata.get("streams", []) if s.get("codec_type") == "video"), {})
    fmt = pdata.get("format", {})

    try:
        shutil.rmtree(frame_dir)
    except Exception:
        pass

    result = {
        "output_path": str(output_path),
        "duration_s": float(fmt.get("duration", 0)),
        "size_bytes": int(fmt.get("size", 0)),
        "video_codec": v.get("codec_name"),
        "width": v.get("width"),
        "height": v.get("height"),
        "n_chunks": len(chunks),
        "n_bytes": n_bytes,
        "command": command,
        "cwd": str(cwd),
        "fps": fps,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    log.info(f"  ✓ {output_path.name}  {result['size_bytes']/1024/1024:.2f}MB  dur={result['duration_s']:.1f}s")

    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="cli_recorder",
                 name="cli_recording", path=str(output_path),
                 duration_s=result["duration_s"], n_chunks=len(chunks))

    return result
