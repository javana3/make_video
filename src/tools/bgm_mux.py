"""M4c · Mux BGM into video without re-encoding video stream.

WORKFLOW R7: each version a separate file (don't overwrite). R9: amix
normalize=0 (we don't amix in this step — just attach single audio track,
but the principle of 'not normalizing per-track' applies to mux too).
"""
from __future__ import annotations

from pathlib import Path

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger
from .ffbin import ffmpeg, ffprobe
from .shell import run as shell_run


@traced_agent("Agent 4 BGM · M4c Mux", phase=4)
def mux_bgm(video_path: Path, audio_path: Path, output_path: Path,
            audio_volume: float = 1.0,
            audio_bitrate: str = "192k") -> dict:
    log = agent_logger("agent4_bgm")
    log.info(f"M4c mux: {video_path.name} + {audio_path.name} → {output_path.name}  vol={audio_volume}")
    """Attach audio_path as the only audio stream of output, video copied."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    # -map 0:v:0 keeps video from input #0 (the silent video)
    # -map 1:a:0 takes audio from input #1 (the BGM wav)
    # -c:v copy: don't re-encode video (R7-spirit: lossless video preservation)
    # -shortest: stop when shorter input ends (handle slight wav/video mismatch)
    cmd = [
        ffmpeg(),
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-filter_complex", f"[1:a]volume={audio_volume}[a]",
        "-map", "0:v:0",
        "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-shortest",
        str(output_path),
    ]
    shell_run(cmd, check=True, timeout=120)

    if not output_path.exists():
        raise RuntimeError(f"mux produced no file at {output_path}")

    # ffprobe verify
    import json as _json
    probe = shell_run([
        ffprobe(), "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(output_path),
    ], check=True)
    data = _json.loads(probe.stdout)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    result = {
        "output_path": str(output_path),
        "duration_s": float(data.get("format", {}).get("duration", 0)),
        "size_bytes": int(data.get("format", {}).get("size", 0)),
        "video_codec": v.get("codec_name") if v else None,
        "audio_codec": a.get("codec_name") if a else None,
        "audio_bitrate": int(a.get("bit_rate", 0)) if a else 0,
    }
    log.info(f"M4c done: {output_path.name}  {result['size_bytes']}B  {result['video_codec']}+{result['audio_codec']}")
    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="agent4_bgm",
                 name="video_with_bgm", path=str(output_path),
                 duration_s=result["duration_s"], size_bytes=result["size_bytes"])
    return result
