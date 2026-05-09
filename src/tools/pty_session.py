"""Long-lived terminal session for the demo-driver agent.

A `PtySession` wraps subprocess.Popen with stdin/stdout pipes, feeds bytes
through a pyte virtual terminal, and *concurrently* samples the screen at
fixed FPS into PNG frames on disk. Recording lasts the full lifetime of
the session — there is NO duration cap. The driver agent decides when to
call `stop()`; that finalises the mp4.

This replaces the old `cli_recorder.py`'s fixed-30s-then-terminate logic.
The recording is a side-effect of the session's lifetime; the agent
operates the program through `send`, `screen_text`, `wait_for`,
`read_recent`, `is_alive`, and `stop`. The session never decides to
end on its own except when the underlying process exits.

Windows note: this uses subprocess pipes, not a real ConPTY. Programs
that detect `isatty() == False` will drop colour and skip raw-mode
features. For most CLI programs (those using `input()` / line-buffered
stdout) this is fine. If a project needs a real TTY, swap in pywinpty
behind the same API.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyte
from PIL import Image, ImageDraw, ImageFont

from ..observability.audit import traced_step
from ..observability.logger import agent_logger
from .ffbin import ffmpeg, ffprobe
from .shell import run as shell_run


# ─── ANSI palette (foreground) ──────────────────────────────────────────
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
    """CJK-capable monospace; fall back to ASCII-only fonts."""
    candidates = [
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\cour.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/Library/Fonts/Menlo.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def _render_grid(grid, font, char_w: int, char_h: int,
                 width: int, height: int, padding: int = 12) -> Image.Image:
    img = Image.new("RGB", (width, height), _BG_COLOR)
    d = ImageDraw.Draw(img)
    for r, row in enumerate(grid):
        col_idx = 0
        while col_idx < len(row):
            char = row[col_idx]
            fg = _ANSI_FG.get(char.fg, _ANSI_FG["default"]) if char.fg else _ANSI_FG["default"]
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


def _grid_text(screen: pyte.Screen) -> str:
    """Snapshot pyte screen to a plain-text 2-D string."""
    rows: list[str] = []
    for r in range(screen.lines):
        row = screen.buffer.get(r) or {}
        cols: list[str] = []
        for c in range(screen.columns):
            ch = row.get(c)
            cols.append(ch.data if ch is not None else " ")
        rows.append("".join(cols).rstrip())
    return "\n".join(rows)


@dataclass
class PtySessionResult:
    output_path: str
    duration_s: float
    size_bytes: int
    n_frames: int
    n_bytes_captured: int
    exit_code: Optional[int]
    started_at: str
    stopped_at: str
    command: list[str]
    cwd: str


class PtySession:
    """Live terminal session with concurrent recording.

    Lifetime:
        s = PtySession.start(cmd, cwd, frames_dir=..., fps=15)
        s.send("input\\n")
        s.wait_for(r"Game over", timeout_s=120)
        ...
        result = s.stop(output_path=Path("recording.mp4"))

    Frames are written as PNGs to `frames_dir` while the session runs;
    `stop()` runs ffmpeg to assemble an mp4 then deletes the frame dir.
    """

    def __init__(self,
                 command: list[str],
                 cwd: Path,
                 frames_dir: Path,
                 env: Optional[dict] = None,
                 fps: int = 15,
                 cols: int = 100,
                 rows: int = 30,
                 width: int = 1920,
                 height: int = 1080,
                 font_size: int = 22):
        self.command = command
        self.cwd = cwd
        self.env = env
        self.fps = fps
        self.cols = cols
        self.rows = rows
        self.width = width
        self.height = height
        self.font_size = font_size
        self.frames_dir = frames_dir

        self._screen = pyte.Screen(cols, rows)
        self._stream = pyte.Stream(self._screen)
        self._proc: Optional[subprocess.Popen] = None
        self._t0: float = 0.0
        self._stopped_at: Optional[float] = None
        self._stop_event = threading.Event()
        self._frame_count = 0
        self._n_bytes = 0
        self._screen_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._sampler_thread: Optional[threading.Thread] = None

        font_path = _find_mono_font()
        self._font = (ImageFont.truetype(font_path, font_size)
                      if font_path else ImageFont.load_default())
        bbox = self._font.getbbox("M")
        self._char_w = bbox[2] - bbox[0]
        self._char_h = max(font_size + 6, bbox[3] - bbox[1] + 6)

        self.log = agent_logger("pty_session")
        self.started_at_iso: Optional[str] = None

    # ─── lifecycle ─────────────────────────────────────────────────────
    @classmethod
    def start(cls, command: list[str], cwd: Path, frames_dir: Path,
              **kwargs) -> "PtySession":
        s = cls(command, cwd, frames_dir, **kwargs)
        s._spawn()
        return s

    def _spawn(self) -> None:
        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)
        self.frames_dir.mkdir(parents=True)

        self.log.info(f"spawn {self.command}  cwd={self.cwd}")
        self._t0 = time.monotonic()
        self.started_at_iso = datetime.now(timezone.utc).isoformat()
        self._proc = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            env={**os.environ, **(self.env or {})},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="pty-reader", daemon=True)
        self._sampler_thread = threading.Thread(
            target=self._sampler_loop, name="pty-sampler", daemon=True)
        self._reader_thread.start()
        self._sampler_thread.start()

    # ─── background loops ──────────────────────────────────────────────
    def _reader_loop(self) -> None:
        assert self._proc is not None
        try:
            while not self._stop_event.is_set():
                chunk = self._proc.stdout.read(4096) if self._proc.stdout else b""
                if not chunk:
                    break
                self._n_bytes += len(chunk)
                try:
                    text = chunk.decode("utf-8", errors="replace")
                    with self._screen_lock:
                        self._stream.feed(text)
                except Exception as e:
                    self.log.warning(f"feed err: {e}")
        except Exception as e:
            self.log.warning(f"reader err: {e}")

    def _sampler_loop(self) -> None:
        period = 1.0 / float(self.fps)
        next_t = self._t0 + period
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(period, next_t - now))
                continue
            with self._screen_lock:
                snapshot = []
                for r in range(self.rows):
                    row = self._screen.buffer.get(r) or {}
                    snapshot.append([row.get(c) or pyte.screens.Char(" ")
                                      for c in range(self.cols)])
            try:
                img = _render_grid(snapshot, self._font, self._char_w, self._char_h,
                                    self.width, self.height)
                img.save(self.frames_dir / f"f_{self._frame_count:07d}.png",
                         optimize=False)
                self._frame_count += 1
            except Exception as e:
                self.log.warning(f"render err: {e}")
            next_t += period

    # ─── agent-facing API ──────────────────────────────────────────────
    def send(self, text: str) -> None:
        """Write to stdin. Caller decides whether to include trailing \\n."""
        if not self._proc or self._proc.stdin is None:
            raise RuntimeError("session not running")
        with traced_step("pty.send", n_bytes=len(text), text=text[:200]):
            self._proc.stdin.write(text.encode("utf-8"))
            self._proc.stdin.flush()

    def screen_text(self) -> str:
        with self._screen_lock:
            return _grid_text(self._screen)

    def read_recent(self, n_lines: int = 30) -> str:
        text = self.screen_text()
        lines = text.split("\n")
        return "\n".join(lines[-n_lines:])

    def wait_for(self, pattern: str, timeout_s: float = 30.0,
                 poll_interval_s: float = 0.2) -> bool:
        """Poll `screen_text()` for a regex; return True on hit, False on timeout."""
        compiled = re.compile(pattern, re.MULTILINE | re.DOTALL)
        deadline = time.monotonic() + timeout_s
        with traced_step("pty.wait_for", pattern=pattern, timeout_s=timeout_s) as span:
            while time.monotonic() < deadline:
                if self._stop_event.is_set():
                    span.set_attribute("step.result", "stopped")
                    return False
                if compiled.search(self.screen_text()):
                    span.set_attribute("step.result", "matched")
                    return True
                if not self.is_alive():
                    if compiled.search(self.screen_text()):
                        span.set_attribute("step.result", "matched_after_exit")
                        return True
                    span.set_attribute("step.result", "process_exited")
                    return False
                time.sleep(poll_interval_s)
            span.set_attribute("step.result", "timeout")
            return False

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def exit_code(self) -> Optional[int]:
        return self._proc.returncode if self._proc else None

    def elapsed_s(self) -> float:
        end = self._stopped_at if self._stopped_at is not None else time.monotonic()
        return end - self._t0

    def stop(self, output_path: Path,
             terminate_process: bool = True,
             keep_frames: bool = False) -> PtySessionResult:
        """Halt sampling/reading, finalise mp4, return result."""
        if self._proc is None:
            raise RuntimeError("session never started")
        with traced_step("pty.stop", output_path=str(output_path)):
            self._stop_event.set()
            self._stopped_at = time.monotonic()

            if terminate_process and self.is_alive():
                try:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        self._proc.wait(timeout=2)
                except Exception:
                    pass

            for th in (self._reader_thread, self._sampler_thread):
                if th and th.is_alive():
                    th.join(timeout=3)

            return self._render_mp4(output_path, keep_frames=keep_frames)

    # ─── ffmpeg finalisation ───────────────────────────────────────────
    def _render_mp4(self, output_path: Path, keep_frames: bool) -> PtySessionResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self._frame_count == 0:
            raise RuntimeError("no frames captured — process produced no output "
                               "or sampler never ran")

        with traced_step("pty.ffmpeg_compose",
                          n_frames=self._frame_count, fps=self.fps,
                          output=str(output_path)):
            cmd = [
                ffmpeg(), "-y",
                "-framerate", str(self.fps),
                "-i", str(self.frames_dir / "f_%07d.png"),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path),
            ]
            shell_run(cmd, check=True, timeout=900)

        probe = shell_run([ffprobe(), "-v", "quiet", "-print_format", "json",
                            "-show_streams", "-show_format", str(output_path)],
                           check=True)
        import json as _json
        pdata = _json.loads(probe.stdout)
        fmt = pdata.get("format", {})

        if not keep_frames:
            try:
                shutil.rmtree(self.frames_dir)
            except Exception:
                pass

        return PtySessionResult(
            output_path=str(output_path),
            duration_s=float(fmt.get("duration", 0)),
            size_bytes=int(fmt.get("size", 0)),
            n_frames=self._frame_count,
            n_bytes_captured=self._n_bytes,
            exit_code=self.exit_code(),
            started_at=self.started_at_iso or "",
            stopped_at=datetime.now(timezone.utc).isoformat(),
            command=list(self.command),
            cwd=str(self.cwd),
        )


@contextmanager
def pty_session(command: list[str], cwd: Path, frames_dir: Path,
                output_path: Path, **kwargs):
    """Context-manager helper: yields the session; auto-stops on exit."""
    s = PtySession.start(command, cwd, frames_dir, **kwargs)
    try:
        yield s
    finally:
        if s.is_alive() or s._stopped_at is None:
            try:
                s.stop(output_path)
            except Exception:
                pass
