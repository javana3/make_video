"""Phase 2B-captions · SRT generator + ffmpeg burn-in.

Reads `demo_timings.json` (from `src/tools/demo_executor.py`) and writes
`captions_zh.srt` + `captions_en.srt` with timecodes synced to the actual
recording. Optionally burns either track into the mp4 using ffmpeg's
`subtitles=...` filter (force_style for modern look).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from loguru import logger

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger
from .ffbin import ffmpeg
from .shell import run as shell_run


def _fmt_srt_time(seconds: float) -> str:
    """0.0 → '00:00:00,000'."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(timings: dict, srt_path: Path, lang: str = "zh") -> dict:
    """Write captions_<lang>.srt from demo_timings. Returns {n_cues, path}."""
    log = agent_logger("captions")
    cap_key = f"caption_{lang}"
    steps = timings.get("steps", [])
    cues: list[str] = []
    n_cues = 0
    for step in steps:
        cap = (step.get(cap_key) or "").strip()
        if not cap:
            continue
        # Extend caption display to span the full step (action + dwell)
        t_start = float(step["t_start"])
        t_end = float(step["t_end"])
        # Cap minimum visible duration so single-frame clicks don't flash
        if t_end - t_start < 1.0:
            t_end = t_start + 1.0
        n_cues += 1
        cues.append(f"{n_cues}")
        cues.append(f"{_fmt_srt_time(t_start)} --> {_fmt_srt_time(t_end)}")
        cues.append(cap)
        cues.append("")
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text("\n".join(cues), encoding="utf-8")
    log.info(f"  ✓ {srt_path.name}  cues={n_cues}  lang={lang}")
    return {"path": str(srt_path), "n_cues": n_cues, "lang": lang}


@traced_agent("Phase 2B · Captions Write", phase=2)
def write_caption_tracks(timings_path: Path, voice_dir: Path) -> dict:
    """Write both zh + en SRT tracks. Returns dict with paths + counts."""
    log = agent_logger("captions")
    timings = json.loads(timings_path.read_text(encoding="utf-8"))
    out: dict = {"zh": None, "en": None}
    for lang, fname in [("zh", "captions_zh.srt"), ("en", "captions_en.srt")]:
        p = voice_dir / fname
        out[lang] = write_srt(timings, p, lang=lang)

    bus = get_run_context().get("event_bus")
    if bus is not None and (out["zh"]["n_cues"] + out["en"]["n_cues"] > 0):
        bus.emit("asset_verified", agent="captions",
                 name="caption_tracks", path=str(voice_dir),
                 zh_cues=out["zh"]["n_cues"], en_cues=out["en"]["n_cues"])
    return out


# ---------------------------------------------------------------------------
# Burn-in
# ---------------------------------------------------------------------------

def _ffmpeg_path_arg(p: Path) -> str:
    """ffmpeg's subtitles= filter has POSIX-quoting nightmares on Windows.

    On Windows we pass forward-slashes and escape the colon after the drive
    letter (e.g. `C\\:/Users/.../captions.srt`) inside the filter argument.
    """
    s = str(p.resolve()).replace("\\", "/")
    # Escape drive colon for ffmpeg filter parser
    if len(s) > 1 and s[1] == ":":
        s = s[0] + r"\:" + s[2:]
    return s


@traced_agent("Phase 2B · Captions Burn", phase=2)
def burn_captions(video_in: Path,
                   srt_path: Path,
                   video_out: Path,
                   font_size: int = 28,
                   margin_v: int = 64) -> dict:
    """Hard-burn captions into mp4 (re-encodes video). Idempotent on output."""
    log = agent_logger("captions")
    if video_out.exists():
        video_out.unlink()
    video_out.parent.mkdir(parents=True, exist_ok=True)

    style = (
        f"FontName=Microsoft YaHei,FontSize={font_size},"
        f"PrimaryColour=&HFFFFFF&,OutlineColour=&H80000000&,"
        f"BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV={margin_v}"
    )
    sub_arg = f"subtitles='{_ffmpeg_path_arg(srt_path)}':force_style='{style}'"

    cmd = [
        ffmpeg(), "-y",
        "-i", str(video_in),
        "-vf", sub_arg,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(video_out),
    ]
    log.info(f"burn captions: {srt_path.name} → {video_out.name}")
    shell_run(cmd, check=True, timeout=240)

    if not video_out.exists():
        raise RuntimeError(f"burn-in produced no file at {video_out}")

    sz = video_out.stat().st_size
    log.info(f"  ✓ {video_out.name}  {sz/1024/1024:.2f} MB")

    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="captions",
                 name="captioned_recording", path=str(video_out),
                 size_bytes=sz, srt_path=str(srt_path))

    return {"output_path": str(video_out), "size_bytes": sz, "srt_used": str(srt_path)}
