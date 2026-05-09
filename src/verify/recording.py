"""Recording (.webm/.mp4) verification — WORKFLOW.md §7.3.

Recording duration is User-decided per WORKFLOW §2.2; if no min_duration is
provided, only sanity-check the file is non-trivial.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from opentelemetry import trace

from .video import VideoProbe, probe

tracer = trace.get_tracer("video-workflow")


@dataclass
class RecordingVerifyResult:
    ok: bool
    probe: VideoProbe
    issues: list[str] = field(default_factory=list)


def verify_recording(path: Path,
                     min_duration: Optional[float] = None,
                     min_width: int = 1920,
                     min_height: int = 1080) -> RecordingVerifyResult:
    with tracer.start_as_current_span("verify.recording") as span:
        span.set_attribute("path", str(path))
        if min_duration is not None:
            span.set_attribute("min_duration", min_duration)
        span.set_attribute("min_width", min_width)
        span.set_attribute("min_height", min_height)

        p = probe(path)
        issues: list[str] = []

        if not p.has_video:
            issues.append("no video stream")

        if min_duration is not None and p.duration < min_duration:
            issues.append(f"duration {p.duration:.1f}s < required {min_duration:.1f}s")
        elif min_duration is None and p.duration < 1.0:
            issues.append(f"recording too short ({p.duration:.2f}s)")

        if p.width < min_width or p.height < min_height:
            issues.append(
                f"resolution {p.width}x{p.height} below {min_width}x{min_height}"
            )

        result = RecordingVerifyResult(ok=len(issues) == 0, probe=p, issues=issues)
        span.set_attribute("verify.passed", result.ok)
        span.set_attribute("verify.issues", len(issues))
        span.set_attribute("probe.duration", p.duration)
        span.set_attribute("probe.width", p.width)
        span.set_attribute("probe.height", p.height)
        return result
