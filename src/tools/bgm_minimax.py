"""M4b alt path · MiniMax music-2.6 cloud BGM generator.

Same external contract as `bgm_musicgen.generate_bgm`:
  - inputs:  scaffold_path (unused here; included for signature symmetry),
             output_path (.wav), prompt, duration_s
  - returns: dict with output_path / model / sample_rate / duration_s / size_bytes

MiniMax returns ~88s mp3 regardless of prompt duration hint, so we trim to
`duration_s` via ffmpeg and convert to 44.1 kHz stereo WAV to match the
downstream bgm_mux expectations.

Endpoint: <MINIMAX_BASE_URL>/v1/music_generation
Model:    <MINIMAX_MUSIC_MODEL> (default music-2.6)
"""
from __future__ import annotations

import binascii
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests
from loguru import logger

from ..observability.audit import get_run_context, traced_agent, traced_step
from ..observability.logger import agent_logger
from .ffbin import ffmpeg as ffmpeg_bin


@traced_agent("Agent 4 BGM · MiniMax music-2.6", phase=4)
def generate_bgm_minimax(
    scaffold_path: Path,  # unused (kept for signature symmetry with musicgen path)
    output_path: Path,
    prompt: str,
    duration_s: float = 25.0,
) -> dict:
    log = agent_logger("agent4_bgm")
    t0 = time.monotonic()

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY not set; cannot use MiniMax BGM path")
    base = os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com").rstrip("/")
    model = os.environ.get("MINIMAX_MUSIC_MODEL", "music-2.6")

    log.info(f"MiniMax music-gen model={model} target_duration={duration_s}s prompt={prompt!r}")

    payload = {
        "model": model,
        "prompt": prompt,
        "is_instrumental": True,
        "audio_setting": {"sample_rate": 44100, "bitrate": 256000, "format": "mp3"},
    }

    with traced_step("MiniMax.music_request", model=model, prompt_len=len(prompt)):
        t_req = time.monotonic()
        r = requests.post(
            f"{base}/v1/music_generation",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=180,
        )
        gen_time = time.monotonic() - t_req

    if r.status_code != 200:
        raise RuntimeError(f"MiniMax music HTTP {r.status_code}: {r.text[:200]}")
    body = r.json()
    base_resp = body.get("base_resp") or {}
    if base_resp.get("status_code") != 0:
        raise RuntimeError(f"MiniMax music base_resp error: {base_resp}")
    audio_hex = (body.get("data") or {}).get("audio") or ""
    if not audio_hex:
        raise RuntimeError(f"MiniMax music returned empty audio; body={body!r}")

    raw = binascii.unhexlify(audio_hex)
    log.info(f"received {len(raw)}B mp3 in {gen_time:.1f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        tmp_mp3 = Path(tf.name)
        tf.write(raw)

    ff = ffmpeg_bin()
    with traced_step("MiniMax.mp3_to_wav_trim",
                       target_duration_s=duration_s, mp3_bytes=len(raw)):
        cmd = [
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(tmp_mp3),
            "-t", f"{duration_s:.2f}",
            "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
            str(output_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr[:300]}")
    try:
        tmp_mp3.unlink()
    except OSError:
        pass

    size = output_path.stat().st_size
    elapsed = time.monotonic() - t0
    log.info(f"wrote {output_path} ({size}B) in {elapsed:.1f}s total")

    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="agent4_bgm",
                 name="bgm_final", path=str(output_path),
                 duration_s=duration_s, model=f"minimax/{model}", device="cloud")

    return {
        "output_path": str(output_path),
        "model": f"minimax/{model}",
        "device": "cloud",
        "use_melody": False,
        "sample_rate": 44100,
        "duration_s": duration_s,
        "size_bytes": size,
        "gen_time_s": round(gen_time, 1),
        "total_time_s": round(elapsed, 1),
    }


def has_minimax_key() -> bool:
    return bool(os.environ.get("MINIMAX_API_KEY"))
