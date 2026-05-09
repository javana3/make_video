"""Audio file verification — WORKFLOW.md §7.2."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from opentelemetry import trace

from ..tools.ffbin import ffprobe
from ..tools.shell import run as shell_run

tracer = trace.get_tracer("video-workflow")


@dataclass
class AudioProbe:
    path: Path
    duration: float
    sample_rate: int
    channels: int
    codec: str


def probe(path: Path) -> AudioProbe:
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

    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if a is None:
        raise ValueError(f"no audio stream in {path}")

    return AudioProbe(
        path=path,
        duration=float(fmt.get("duration", 0)),
        sample_rate=int(a.get("sample_rate", 0)),
        channels=int(a.get("channels", 0)),
        codec=a.get("codec_name", ""),
    )


@dataclass
class AudioVerifyResult:
    ok: bool
    probe: AudioProbe
    issues: list[str] = field(default_factory=list)


def verify_audio(path: Path,
                 expected_duration: Optional[float] = None,
                 duration_tolerance: float = 0.1) -> AudioVerifyResult:
    """WORKFLOW §7.2: voice_full.wav must match video duration ±0.1s."""
    with tracer.start_as_current_span("verify.audio") as span:
        span.set_attribute("path", str(path))
        if expected_duration is not None:
            span.set_attribute("expected_duration", expected_duration)
            span.set_attribute("duration_tolerance", duration_tolerance)

        p = probe(path)
        issues: list[str] = []

        if expected_duration is not None:
            delta = abs(p.duration - expected_duration)
            if delta > duration_tolerance:
                issues.append(
                    f"duration {p.duration:.3f}s outside expected "
                    f"{expected_duration:.3f}±{duration_tolerance}s (delta={delta:.3f}s)"
                )

        result = AudioVerifyResult(ok=len(issues) == 0, probe=p, issues=issues)
        span.set_attribute("verify.passed", result.ok)
        span.set_attribute("verify.issues", len(issues))
        span.set_attribute("probe.duration", p.duration)
        span.set_attribute("probe.sample_rate", p.sample_rate)
        span.set_attribute("probe.channels", p.channels)
        return result
