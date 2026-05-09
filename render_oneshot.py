"""One-shot: cutting_plan.json → Remotion project → npm install → render → v1.mp4."""
import json, sys
from pathlib import Path

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from src.tools.dotenv import load_dotenv
from src.tools.shell import run as shell_run
from src.tools.ffbin import ffprobe
from src.tools.remotion_codegen import generate_project
from src.tools.remotion_render import npm_install, render
from src.observability.logger import setup_logging

load_dotenv()

run_dir = Path('workspace/football-match-simulator/runs/49aecf4a')
setup_logging(run_dir)

plan_path = run_dir / 'cutting_plan.json'
plan = json.loads(plan_path.read_text(encoding='utf-8'))
recording = run_dir / 'recordings' / 'test.mp4'

# Probe source recording fps for accurate startFrom calculation
probe = shell_run([ffprobe(), '-v', 'quiet', '-print_format', 'json',
                   '-show_streams', str(recording)], check=True)
streams = json.loads(probe.stdout)['streams']
v = next(s for s in streams if s['codec_type'] == 'video')
src_fps_str = v['r_frame_rate']  # like "25/1"
num, den = src_fps_str.split('/')
src_fps = round(int(num) / int(den))
print(f"source recording fps: {src_fps}")

remotion_dir = run_dir / 'remotion'
print(f"\n=== Step 1: codegen → {remotion_dir} ===")
result = generate_project(plan, remotion_dir, recording, src_fps=src_fps)
print(f"  {result['scenes']} scenes, {result['total_frames']} frames @ {result['fps']}fps = {result['duration_s']:.1f}s")

print(f"\n=== Step 2: npm install (first time slow) ===")
npm_install(remotion_dir, timeout=600)

outputs_dir = run_dir / 'outputs'
out_path = outputs_dir / 'v1.mp4'
print(f"\n=== Step 3: render → {out_path} ===")
result = render(remotion_dir, out_path, composition_id='MyVideo', timeout=600)

print(f"\n=== RESULT ===")
print(f"output:    {out_path.resolve()}")
print(f"size:      {result['size_bytes']} bytes  ({result['size_bytes']/1024/1024:.1f} MB)")

# ffprobe the output
probe2 = shell_run([ffprobe(), '-v', 'quiet', '-print_format', 'json',
                    '-show_streams', '-show_format', str(out_path)], check=True)
data = json.loads(probe2.stdout)
v2 = next(s for s in data['streams'] if s['codec_type'] == 'video')
print(f"duration:  {float(data['format']['duration']):.2f}s")
print(f"size:      {v2['width']}x{v2['height']}")
print(f"codec:     {v2['codec_name']}")
