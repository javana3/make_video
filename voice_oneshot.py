"""M5 oneshot · run all 4 voiceover steps end-to-end.

Step 1: Agent 5 LLM proposes voiceover_script_zh-CN.json + voiceover_script_en-US.json
Step 2: edge-tts synthesizes per-segment mp3
Step 3: voice_timeline assembles voice_full.wav
Step 4: bgm_duck_mux mixes BGM + voice into final video

Picks the latest available BGM-baked video (v1_bgm_final.mp4 if MusicGen done,
else v1_bgm_scaffold.mp4).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from src.tools.dotenv import load_dotenv
load_dotenv()

from src.agents.voice_over import propose_script
from src.tools.tts_edge import synth_script
from src.tools.voice_timeline import assemble_timeline
from src.tools.bgm_duck_mux import duck_and_mux
from src.observability.logger import setup_logging


def pick_bgm_video(run_dir: Path) -> Path:
    out = run_dir / "outputs"
    candidates = [
        out / "v1_bgm_final.mp4",       # MusicGen output (preferred)
        out / "v1_bgm_scaffold.mp4",    # numpy scaffold
        out / "v1.mp4",                 # silent (last resort)
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c
    raise FileNotFoundError("no BGM video found in outputs/")


def main(language: str = "zh-CN") -> None:
    run_dir = Path("workspace/football-match-simulator/runs/49aecf4a")
    setup_logging(run_dir)

    voice_dir = run_dir / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)

    bilingual_path = voice_dir / "voiceover_script_bilingual.json"
    per_lang_path = voice_dir / f"voiceover_script_{language}.json"
    per_segment_dir = voice_dir / f"per_segment_{language}"
    voice_full_wav = voice_dir / f"voice_full_{language}.wav"

    bgm_video = pick_bgm_video(run_dir)
    final_out = run_dir / "outputs" / f"final_{language}.mp4"
    print(f"=== input video (with BGM): {bgm_video.name}")
    print(f"=== language: {language}")

    # === Step 1 ===
    if not per_lang_path.exists():
        print("\n=== Step 1: Agent 5 LLM proposes voiceover script ===")
        brief = (run_dir / "project_brief.md").read_text(encoding="utf-8")
        plan = json.loads((run_dir / "cutting_plan.json").read_text(encoding="utf-8"))
        propose_script(run_dir, brief, plan, bilingual_path)
    else:
        print(f"\n=== Step 1 SKIP: {per_lang_path.name} already exists ===")

    # === Step 2 ===
    print(f"\n=== Step 2: edge-tts synth → {per_segment_dir.name}/ ===")
    t0 = time.monotonic()
    synth_result = synth_script(per_lang_path, per_segment_dir)
    print(f"  synth done in {time.monotonic()-t0:.1f}s — {len(synth_result['entries'])} clips")

    # === Step 3 ===
    print(f"\n=== Step 3: timeline assemble → {voice_full_wav.name} ===")
    timeline_result = assemble_timeline(synth_result, bgm_video, voice_full_wav)
    print(f"  voice_full.wav  dur={timeline_result['duration_s']:.2f}s  size={timeline_result['size_bytes']}B")

    # === Step 4 ===
    print(f"\n=== Step 4: BGM ducking + mux → {final_out.name} ===")
    mux_result = duck_and_mux(
        bgm_video, voice_full_wav, per_lang_path, final_out,
        voice_volume=0.7, bgm_ducked_volume=0.3,
    )
    print(f"\n=== RESULT ===")
    print(f"output:           {final_out.resolve()}")
    print(f"duration:         {mux_result['duration_s']:.2f}s")
    print(f"size:             {mux_result['size_bytes']/1024/1024:.2f} MB")
    print(f"video codec:      {mux_result['video_codec']}")
    print(f"audio codec:      {mux_result['audio_codec']}")
    print(f"voice segments:   {mux_result['n_voice_segments']}")
    print(f"voice volume:     {mux_result['voice_volume']}")
    print(f"BGM ducked vol:   {mux_result['bgm_ducked_volume']}")


if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else "zh-CN"
    main(lang)
