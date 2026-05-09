"""Video file verification — WORKFLOW.md §7.1."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..tools.ffbin import ffprobe
from ..tools.shell import run as shell_run


@dataclass
class VideoProbe:
    path: Path
    duration: float
    size_bytes: int
    width: int
    height: int
    video_codec: Optional[str]
    audio_codec: Optional[str]
    has_video: bool
    has_audio: bool


def probe(path: Path) -> VideoProbe:
    if not path.exists():
        raise FileNotFoundError(path)

    result = shell_run([
        ffprobe(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ], check=True)

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    return VideoProbe(
        path=path,
        duration=float(fmt.get("duration", 0)),
        size_bytes=int(fmt.get("size", 0)),
        width=int(v.get("width", 0)) if v else 0,
        height=int(v.get("height", 0)) if v else 0,
        video_codec=v.get("codec_name") if v else None,
        audio_codec=a.get("codec_name") if a else None,
        has_video=v is not None,
        has_audio=a is not None,
    )


@dataclass
class VideoVerifyResult:
    ok: bool
    probe: VideoProbe
    issues: list[str] = field(default_factory=list)


def verify_video(path: Path,
                 expected_duration: Optional[float] = None,
                 duration_tolerance: float = 0.5,
                 expect_audio: bool = False,
                 expected_audio_codec: str = "aac",
                 expected_video_codec: str = "h264",
                 min_size_mb: float = 5.0) -> VideoVerifyResult:
    """WORKFLOW §7.1: check video stream / audio stream / duration / size."""
    p = probe(path)
    issues: list[str] = []

    if not p.has_video:
        issues.append("no video stream")
    elif p.video_codec != expected_video_codec:
        issues.append(f"video codec is {p.video_codec}, expected {expected_video_codec}")

    if expect_audio:
        if not p.has_audio:
            issues.append("no audio stream")
        elif p.audio_codec != expected_audio_codec:
            issues.append(f"audio codec is {p.audio_codec}, expected {expected_audio_codec}")

    if expected_duration is not None:
        if abs(p.duration - expected_duration) > duration_tolerance:
            issues.append(
                f"duration {p.duration:.2f}s outside expected "
                f"{expected_duration:.2f}±{duration_tolerance}s"
            )

    if min_size_mb is not None:
        size_mb = p.size_bytes / (1024 * 1024)
        if size_mb < min_size_mb:
            issues.append(f"size {size_mb:.2f}MB < expected min {min_size_mb}MB")

    return VideoVerifyResult(ok=len(issues) == 0, probe=p, issues=issues)
