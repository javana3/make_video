"""M4a · Beat-grid scaffold.

Reads cutting_plan.json, computes scene cut times, generates a BGM scaffold
WAV with kicks aligned to scene boundaries + steady hi-hat + snare on beat 2/4
+ a sub-bass tone. This becomes the melody guide for MusicGen-melody.

Pure numpy + wave — no audio library deps.
"""
from __future__ import annotations

import json
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger

SAMPLE_RATE = 44100


@dataclass
class ScaffoldConfig:
    bpm: int = 130
    duration_s: float = 25.5
    sample_rate: int = SAMPLE_RATE


def _kick(t_seconds: float, duration: float = 0.18, freq_start: float = 80,
          freq_end: float = 40, sr: int = SAMPLE_RATE) -> np.ndarray:
    n = int(duration * sr)
    t = np.arange(n) / sr
    # Pitch sweep 80→40 Hz; quick exponential decay
    freq = np.linspace(freq_start, freq_end, n)
    phase = np.cumsum(2 * np.pi * freq / sr)
    env = np.exp(-t * 22)
    return 0.95 * env * np.sin(phase)


def _snare(t_seconds: float, duration: float = 0.12, sr: int = SAMPLE_RATE) -> np.ndarray:
    n = int(duration * sr)
    t = np.arange(n) / sr
    noise = (np.random.rand(n) * 2 - 1) * 0.6
    tone = np.sin(2 * np.pi * 200 * t) * 0.3
    env = np.exp(-t * 18)
    return env * (noise + tone)


def _hat(t_seconds: float, duration: float = 0.04, sr: int = SAMPLE_RATE) -> np.ndarray:
    n = int(duration * sr)
    t = np.arange(n) / sr
    noise = (np.random.rand(n) * 2 - 1) * 0.5
    env = np.exp(-t * 80)
    return env * noise


def _sub_bass(t_seconds: float, duration: float, freq: float = 55,
              sr: int = SAMPLE_RATE) -> np.ndarray:
    n = int(duration * sr)
    t = np.arange(n) / sr
    env_attack = np.minimum(t / 0.05, 1.0)  # 50ms attack
    env_release = np.exp(-(t - duration + 0.1).clip(min=0) * 8)
    return 0.45 * env_attack * env_release * np.sin(2 * np.pi * freq * t)


def _add_at(buf: np.ndarray, t_s: float, snippet: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Add snippet to buf at time offset t_s (in place; clips at buf end)."""
    start = int(t_s * sr)
    end = min(start + len(snippet), len(buf))
    if end > start:
        buf[start:end] += snippet[: end - start]


@traced_agent("Agent 4 BGM · M4a Scaffold", phase=4)
def generate_scaffold(cutting_plan: dict, output_path: Path,
                      bpm: int = 130) -> dict:
    """Generate scaffold WAV with kicks at scene cuts + steady beat grid."""
    log = agent_logger("agent4_bgm")
    log.info(f"M4a scaffold: bpm={bpm} → {output_path.name}")
    fps = cutting_plan["fps"]
    scenes = cutting_plan["scenes"]

    # Compute scene cut times in seconds (taking 15-frame crossfade overlaps into account)
    cut_times: list[float] = [0.0]
    cur_frames = 0
    for i, s in enumerate(scenes):
        cur_frames += int(s["duration_s"] * fps)
        if i < len(scenes) - 1:
            cur_frames -= 15  # crossfade overlap
        cut_times.append(cur_frames / fps)
    total_s = cut_times[-1]

    sr = SAMPLE_RATE
    n_total = int(total_s * sr) + sr  # +1s tail
    buf = np.zeros(n_total, dtype=np.float32)

    # Beat grid
    beat_s = 60.0 / bpm
    n_beats = int(total_s / beat_s) + 1

    # 4-on-the-floor kick
    for b in range(n_beats):
        t = b * beat_s
        if t > total_s:
            break
        _add_at(buf, t, _kick(t))

    # Snare on beats 2 and 4
    for b in range(n_beats):
        if b % 4 in (1, 3):
            t = b * beat_s
            if t > total_s:
                break
            _add_at(buf, t, _snare(t))

    # Hi-hat on every 8th note
    half_beat = beat_s / 2
    n_hats = int(total_s / half_beat) + 1
    for h in range(n_hats):
        t = h * half_beat
        if t > total_s:
            break
        _add_at(buf, t, _hat(t))

    # Sub-bass note per scene (root of chord/segment)
    bass_freqs = [55, 65.4, 49, 73.4, 55, 82.4, 65.4, 55][:len(scenes)]
    for i, s in enumerate(scenes):
        start_t = cut_times[i]
        dur_t = s["duration_s"]
        if start_t + dur_t > total_s:
            dur_t = total_s - start_t
        if dur_t > 0.1:
            _add_at(buf, start_t,
                    _sub_bass(start_t, dur_t,
                              freq=bass_freqs[i % len(bass_freqs)]))

    # Emphasis kick on every scene cut (extra impact)
    for t in cut_times[:-1]:
        _add_at(buf, t, _kick(t, duration=0.25, freq_start=120, freq_end=45))

    # Trim to total duration
    buf = buf[: int(total_s * sr)]

    # Normalize to -3dB
    peak = np.abs(buf).max()
    if peak > 0:
        buf = buf * (0.707 / peak)

    # Convert to int16 PCM
    pcm = (buf * 32767).astype(np.int16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())

    log.info(f"M4a done: {output_path.stat().st_size}B  dur={total_s:.2f}s  bpm={bpm}")
    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="agent4_bgm",
                 name="bgm_scaffold", path=str(output_path),
                 duration_s=total_s, bpm=bpm)

    return {
        "output_path": str(output_path),
        "duration_s": total_s,
        "bpm": bpm,
        "n_scenes": len(scenes),
        "n_cuts": len(cut_times) - 1,
        "size_bytes": output_path.stat().st_size,
    }
