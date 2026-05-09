"""Oneshot: scaffold + prompt → MusicGen → bgm_final.wav."""
import sys, time
from pathlib import Path

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from src.tools.bgm_musicgen import generate_bgm, build_prompt_from_brief
from src.observability.logger import setup_logging

run_dir = Path('workspace/football-match-simulator/runs/49aecf4a')
setup_logging(run_dir)

scaffold = run_dir / 'bgm' / 'bgm_scaffold.wav'
out = run_dir / 'bgm' / 'bgm_final.wav'
brief = (run_dir / 'project_brief.md').read_text(encoding='utf-8')
prompt = build_prompt_from_brief(brief, bpm=130)

import os
import tempfile
# In MSYS2/git-bash, /tmp maps to %TEMP% (Windows Local Temp). Python's Path
# resolves /tmp to C:\tmp which won't exist on stock Windows. Use real Windows
# temp dir to find the curl-downloaded local model copy.
LOCAL_MODEL = os.environ.get("MUSICGEN_LOCAL_DIR") or str(
    Path(tempfile.gettempdir()) / "musicgen-small-local"
)
model_name = LOCAL_MODEL if Path(LOCAL_MODEL).exists() else "facebook/musicgen-small"

print(f"scaffold:    {scaffold}  (unused — small model is text-only)")
print(f"output:      {out}")
print(f"prompt:      {prompt}")
print(f"model_path:  {model_name}")
print("starting MusicGen — facebook/musicgen-small (2.20 GB, text-only)")
t0 = time.monotonic()
result = generate_bgm(
    scaffold, out, prompt, duration_s=25.0,
    model_name=model_name,
    use_melody=False,
)
total = time.monotonic() - t0

print(f"\n=== RESULT ===")
print(f"output:        {out.resolve()}")
print(f"size:          {result['size_bytes']/1024:.0f} KB")
print(f"duration:      {result['duration_s']:.2f}s")
print(f"sample_rate:   {result['sample_rate']} Hz")
print(f"model:         {result['model']}")
print(f"device:        {result['device']}")
print(f"use_melody:    {result['use_melody']}")
print(f"load time:     {result['load_time_s']}s")
print(f"gen time:      {result['gen_time_s']}s")
print(f"total wall:    {total:.1f}s")
