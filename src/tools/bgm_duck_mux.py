"""M5 Step 4 · BGM Ducking + amix → final video.

WORKFLOW 5.4: take video_bgm.mp4 (already has BGM audio) + voice_full.wav,
duck BGM under voice and amix with normalize=0.

Filter graph for N voice segments:
  [0:a]volume=enable='between(t,t1s,t1e)+between(t,t2s,t2e)+...':volume=0.3[bgm_duck]
  [1:a]volume=0.7[voice]
  [bgm_duck][voice]amix=inputs=2:normalize=0[aout]

normalize=0 is critical (R9): default normalize=1 divides every track by N which
silently halves BGM after amix.
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger
from .ffbin import ffmpeg, ffprobe
from .shell import run as shell_run


@traced_agent("Agent 5 Voice · Step4 Ducking + Mux", phase=5)
def duck_and_mux(video_with_bgm: Path,
                 voice_full_wav: Path,
                 voiceover_script_path: Path,
                 output_path: Path,
                 voice_volume: float = 0.7,
                 bgm_ducked_volume: float = 0.3,
                 audio_bitrate: str = "192k") -> dict:
    """Duck BGM under voice intervals, amix, mux into final video."""
    log = agent_logger("agent5_voice")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    script = json.loads(voiceover_script_path.read_text(encoding="utf-8"))
    if not script:
        raise ValueError("voiceover_script.json is empty — nothing to duck against")

    # Build OR'd between() expression for all voice segments
    parts = [
        f"between(t,{e['t_start']},{e['t_end']})" for e in script
    ]
    enable_expr = "+".join(parts)
    log.info(f"ducking BGM during {len(parts)} voice segments")

    # filter_complex
    fc = (
        f"[0:a]volume=enable='{enable_expr}':volume={bgm_ducked_volume}[bgm_duck];"
        f"[1:a]volume={voice_volume}[voice];"
        f"[bgm_duck][voice]amix=inputs=2:normalize=0:dropout_transition=0[aout]"
    )
    cmd = [
        ffmpeg(), "-y",
        "-i", str(video_with_bgm),
        "-i", str(voice_full_wav),
        "-filter_complex", fc,
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-shortest",
        str(output_path),
    ]
    shell_run(cmd, check=True, timeout=180)

    if not output_path.exists():
        raise RuntimeError(f"duck+mux produced no file at {output_path}")

    probe = shell_run([
        ffprobe(), "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(output_path),
    ], check=True)
    data = json.loads(probe.stdout)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    result = {
        "output_path": str(output_path),
        "duration_s": float(data.get("format", {}).get("duration", 0)),
        "size_bytes": int(data.get("format", {}).get("size", 0)),
        "video_codec": v.get("codec_name") if v else None,
        "audio_codec": a.get("codec_name") if a else None,
        "n_voice_segments": len(parts),
        "bgm_ducked_volume": bgm_ducked_volume,
        "voice_volume": voice_volume,
    }
    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="agent5_voice",
                 name="final_video", path=str(output_path),
                 duration_s=result["duration_s"], size_bytes=result["size_bytes"],
                 n_voice_segments=len(parts))
    return result
