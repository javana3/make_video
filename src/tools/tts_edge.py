"""M5 Step 2 · edge-tts wrapper — generate per-segment voice WAVs.

WORKFLOW 5.2: each script entry in voiceover_script.json becomes a single wav
file in run_dir/voice/per_segment/. Times are not encoded into the wav itself
(timeline assembly happens in voice_timeline.py); these are raw clip files.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from loguru import logger

import edge_tts

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger


# Default voice per language. WORKFLOW 5.2 calls out en-US-EricNeural and
# zh-CN-YunxiNeural; YunjianNeural is the sports-tagged Chinese male voice
# which fits hype trap promo video style better than YunxiNeural.
_DEFAULT_VOICE = {
    "en": "en-US-EricNeural",
    "en-US": "en-US-EricNeural",
    "zh": "zh-CN-YunjianNeural",
    "zh-CN": "zh-CN-YunjianNeural",
}


@dataclass
class TTSEntry:
    id: str
    text: str
    lang: str
    t_start: float
    t_end: float
    voice: str | None = None
    rate: str = "+0%"
    volume: str = "+0%"


def _pick_voice(entry: TTSEntry) -> str:
    if entry.voice:
        return entry.voice
    return _DEFAULT_VOICE.get(entry.lang, _DEFAULT_VOICE["en"])


async def _synth_one(entry: TTSEntry, out_path: Path) -> None:
    voice = _pick_voice(entry)
    communicate = edge_tts.Communicate(
        text=entry.text, voice=voice, rate=entry.rate, volume=entry.volume,
    )
    await communicate.save(str(out_path))


@traced_agent("Agent 5 Voice · Step2 TTS", phase=5)
def synth_script(script_path: Path, out_dir: Path) -> dict:
    """Read voiceover_script.json, produce one mp3 per entry.

    Returns dict with per-entry results: {id, voice, path, size_bytes}.
    """
    log = agent_logger("agent5_voice")
    out_dir.mkdir(parents=True, exist_ok=True)
    script = json.loads(script_path.read_text(encoding="utf-8"))

    results = []
    loop = asyncio.new_event_loop()
    try:
        for raw in script:
            entry = TTSEntry(
                id=raw["id"], text=raw["text"], lang=raw.get("lang", "en"),
                t_start=raw["t_start"], t_end=raw["t_end"],
                voice=raw.get("voice"),
                rate=raw.get("rate", "+0%"),
                volume=raw.get("volume", "+0%"),
            )
            voice = _pick_voice(entry)
            # edge-tts produces mp3 by default; we keep mp3 (smaller) and let
            # ffmpeg in voice_timeline read it directly.
            out_path = out_dir / f"{entry.id}.mp3"
            log.info(f"synth {entry.id}  voice={voice}  text={entry.text[:50]!r}")
            loop.run_until_complete(_synth_one(entry, out_path))
            sz = out_path.stat().st_size
            results.append({
                "id": entry.id, "voice": voice, "path": str(out_path),
                "size_bytes": sz, "t_start": entry.t_start, "t_end": entry.t_end,
                "text": entry.text,
            })
            log.info(f"  → {out_path.name} {sz}B")
    finally:
        loop.close()

    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="agent5_voice",
                 name="tts_clips", path=str(out_dir),
                 n_clips=len(results))
    return {"out_dir": str(out_dir), "entries": results}
