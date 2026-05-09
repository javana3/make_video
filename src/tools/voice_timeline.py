"""M5 Step 3 · Assemble per-segment voice files into voice_full.wav.

WORKFLOW 5.3: each entry's clip lands at its t_start, gaps are silent. The
output wav matches the target video's duration so it can later be amix'd
against BGM in step 4 (bgm_duck_mux.py).

Implementation: one ffmpeg invocation with filter_complex —
- anullsrc generates silent base of video_duration
- each segment mp3 gets adelay=t_start*1000ms then mixed with normalize=0
  (normalize=1 would scale BOTH base + voice down by inputs count; we want
  voice at full level over silence)
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger
from .ffbin import ffmpeg, ffprobe
from .shell import run as shell_run


def _video_duration_s(video_path: Path) -> float:
    probe = shell_run([
        ffprobe(), "-v", "quiet", "-print_format", "json",
        "-show_format", str(video_path),
    ], check=True)
    data = json.loads(probe.stdout)
    return float(data["format"]["duration"])


@traced_agent("Agent 5 Voice · Step3 Timeline", phase=5)
def assemble_timeline(synth_result: dict, video_path: Path,
                      out_wav: Path, sample_rate: int = 44100) -> dict:
    """Pad and mix per-segment mp3s onto a silent track matching video length."""
    log = agent_logger("agent5_voice")
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    duration = _video_duration_s(video_path)
    log.info(f"video duration: {duration:.2f}s — building voice_full.wav")

    entries = synth_result["entries"]
    if not entries:
        # Just emit silence of matching length
        cmd = [
            ffmpeg(), "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r={sample_rate}:cl=stereo",
            "-t", f"{duration:.3f}",
            str(out_wav),
        ]
        shell_run(cmd, check=True, timeout=60)
        return {
            "output_path": str(out_wav),
            "duration_s": duration,
            "n_segments": 0,
            "size_bytes": out_wav.stat().st_size,
        }

    # Build inputs: input 0 is the silent base; input 1..N are segment mp3s
    cmd = [ffmpeg(), "-y",
           "-f", "lavfi",
           "-i", f"anullsrc=r={sample_rate}:cl=stereo",
           "-t", f"{duration:.3f}"]
    for e in entries:
        cmd += ["-i", e["path"]]

    # Build filter_complex: delay each voice clip and mix
    n = len(entries)
    filter_parts = []
    mix_inputs = ["[0:a]"]
    for i, e in enumerate(entries, start=1):
        delay_ms = int(e["t_start"] * 1000)
        # adelay needs comma-separated per channel; "all=1" pads all channels
        filter_parts.append(
            f"[{i}:a]aresample={sample_rate},aformat=channel_layouts=stereo,"
            f"adelay={delay_ms}|{delay_ms}[v{i}]"
        )
        mix_inputs.append(f"[v{i}]")
    mix_chain = "".join(mix_inputs) + (
        f"amix=inputs={n+1}:normalize=0:dropout_transition=0[aout]"
    )
    filter_complex = ";".join(filter_parts + [mix_chain])

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-t", f"{duration:.3f}",
        "-ar", str(sample_rate),
        "-ac", "2",
        str(out_wav),
    ]
    log.info(f"ffmpeg amix on {n} segments → {out_wav.name}")
    shell_run(cmd, check=True, timeout=120)

    if not out_wav.exists():
        raise RuntimeError(f"voice timeline assembly failed: {out_wav} missing")

    actual_dur = _video_duration_s(out_wav)
    log.info(f"  ✓ voice_full.wav  {out_wav.stat().st_size}B  dur={actual_dur:.2f}s")
    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="agent5_voice",
                 name="voice_full", path=str(out_wav),
                 duration_s=actual_dur, n_segments=n)
    return {
        "output_path": str(out_wav),
        "duration_s": actual_dur,
        "n_segments": n,
        "size_bytes": out_wav.stat().st_size,
    }
