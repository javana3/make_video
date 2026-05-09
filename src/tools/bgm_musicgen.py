"""M4b · MusicGen-melody upgrade via Hugging Face transformers.

Uses transformers' built-in MusicGen support instead of audiocraft (audiocraft
fails to install on Windows due to spacy/thinc/blis C++ build chain).

Hardware paths:
- GPU fp16 ≥3 GB VRAM   → facebook/musicgen-melody  (1.5B, melody-conditioned)
- GPU fp16 1.5–3 GB VRAM → facebook/musicgen-small  (300M, text-only)
- CPU                   → facebook/musicgen-small  (~10 min for 25s)

HF_ENDPOINT is honored if set externally; otherwise the official huggingface.co
is used. Mirror endpoints (hf-mirror.com etc.) frequently rate-limit large model
files to <50 KB/s, so prefer the official HF when a VPN is available.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from ..observability.audit import get_run_context, traced_agent, traced_step
from ..observability.logger import agent_logger


def _device_and_model_choice() -> tuple[str, str, bool]:
    """Decide compute device + best-fit MusicGen model + whether melody-conditioned."""
    import torch
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        free_gb = free / (1024 ** 3)
        if free_gb >= 6.0:
            return "cuda", "facebook/musicgen-melody-large", True
        if free_gb >= 2.8:
            return "cuda", "facebook/musicgen-melody", True
        if free_gb >= 1.2:
            return "cuda", "facebook/musicgen-small", False
        return "cpu", "facebook/musicgen-small", False
    return "cpu", "facebook/musicgen-small", False


@traced_agent("Agent 4 BGM · M4b MusicGen", phase=4)
def generate_bgm(scaffold_path: Path,
                 output_path: Path,
                 prompt: str,
                 duration_s: float = 25.0,
                 model_name: Optional[str] = None,
                 device: Optional[str] = None,
                 use_melody: Optional[bool] = None) -> dict:
    """Run MusicGen with optional melody conditioning + text prompt.

    Returns dict with output_path / model / device / sample_rate / duration_s.
    """
    log = agent_logger("agent4_bgm")
    t0 = time.monotonic()

    if device is None or model_name is None or use_melody is None:
        chosen_device, chosen_model, chosen_use_melody = _device_and_model_choice()
        device = device or chosen_device
        model_name = model_name or chosen_model
        if use_melody is None:
            use_melody = chosen_use_melody

    log.info(f"loading {model_name} on {device}  melody={use_melody}")
    log.info(f"HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', '(default)')}")

    import torch
    import torchaudio
    import numpy as np
    import soundfile as sf
    from transformers import (
        MusicgenForConditionalGeneration,
        MusicgenMelodyForConditionalGeneration,
        AutoProcessor,
    )

    dtype = torch.float16 if device == "cuda" else torch.float32

    # `musicgen-melody` checkpoint is a separate architecture from plain `musicgen` —
    # transformers ≥4.40 has dedicated MusicgenMelody* classes for the melody-conditioned
    # variant. Loading it via plain MusicgenForConditionalGeneration silently drops the
    # melody-conditioning weights and warns "model of type `musicgen_melody` ... not supported".
    is_melody_ckpt = "melody" in model_name
    ModelCls = (
        MusicgenMelodyForConditionalGeneration if is_melody_ckpt
        else MusicgenForConditionalGeneration
    )

    log.info(f"loading via {ModelCls.__name__} (first run downloads ~3 GB) ...")
    with traced_step("MusicGen.load_model", model_class=ModelCls.__name__,
                       model_name=model_name, dtype=str(dtype), device=device):
        model = ModelCls.from_pretrained(model_name, torch_dtype=dtype).to(device)
    with traced_step("MusicGen.load_processor", model_name=model_name):
        processor = AutoProcessor.from_pretrained(model_name)
    elapsed_load = time.monotonic() - t0
    log.info(f"model loaded in {elapsed_load:.1f}s")

    # MusicGen tokens-per-second is fixed at the encodec frame rate (50 Hz).
    max_new_tokens = int(duration_s * 50) + 4

    log.info(f"prompt:        {prompt!r}")
    log.info(f"max_new_tokens: {max_new_tokens}  (~{duration_s}s)")

    with traced_step("MusicGen.tokenize_inputs", use_melody=use_melody,
                       prompt_len=len(prompt)):
        if use_melody:
            log.info(f"loading scaffold: {scaffold_path}")
            melody, sr = torchaudio.load(str(scaffold_path))
            # MusicGen-melody expects mono input
            if melody.shape[0] > 1:
                melody = melody.mean(dim=0, keepdim=True)
            melody_np = melody.squeeze(0).numpy()
            inputs = processor(
                audio=melody_np,
                sampling_rate=sr,
                text=[prompt],
                padding=True,
                return_tensors="pt",
            ).to(device)
        else:
            inputs = processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            ).to(device)

    log.info("generating (this is the slow part — 30s-10min depending on hardware) ...")
    t1 = time.monotonic()
    with traced_step("MusicGen.generate",
                       max_new_tokens=max_new_tokens,
                       guidance_scale=3, do_sample=True,
                       device=device, dtype=str(dtype)):
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3,
                max_new_tokens=max_new_tokens,
            )
    elapsed_gen = time.monotonic() - t1
    log.info(f"generated in {elapsed_gen:.1f}s")

    # audio_values shape: (batch, 1, samples)
    out_sr = model.config.audio_encoder.sampling_rate
    audio_np = audio_values[0, 0].cpu().float().numpy()

    with traced_step("MusicGen.write_wav", path=str(output_path),
                       samples=len(audio_np), sample_rate=out_sr):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_np, out_sr)
    log.info(f"saved {output_path} ({output_path.stat().st_size}B, {out_sr}Hz)")
    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="agent4_bgm",
                 name="bgm_final", path=str(output_path),
                 duration_s=len(audio_np) / out_sr, model=model_name, device=device)

    return {
        "output_path": str(output_path),
        "model": model_name,
        "device": device,
        "use_melody": use_melody,
        "sample_rate": out_sr,
        "duration_s": len(audio_np) / out_sr,
        "size_bytes": output_path.stat().st_size,
        "load_time_s": round(elapsed_load, 1),
        "gen_time_s": round(elapsed_gen, 1),
    }


def build_prompt_from_brief(brief: str, bpm: int = 130) -> str:
    """Hardcoded prompt builder. Agent 4 LLM can replace this later."""
    if "足球" in brief or "football" in brief.lower():
        return (
            f"wild exuberant hard-hitting hype trap, {bpm} BPM, 808 sub bass, "
            f"snare rolls, melodic lead synth, stadium chant atmosphere, "
            f"no vocals, professional mix"
        )
    return (
        f"upbeat cinematic electronic, {bpm} BPM, driving rhythm, "
        f"melodic synth lead, rich bass, no vocals, professional mix"
    )
