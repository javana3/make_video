"""Oneshot: cutting_plan.json → bgm_scaffold.wav (M4a only, no MusicGen yet)."""
import json, sys
from pathlib import Path

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from src.tools.bgm_scaffold import generate_scaffold

run_dir = Path('workspace/football-match-simulator/runs/49aecf4a')
plan = json.loads((run_dir / 'cutting_plan.json').read_text(encoding='utf-8'))

out = run_dir / 'bgm' / 'bgm_scaffold.wav'
print(f"generating scaffold ({len(plan['scenes'])} scenes, BPM 130) ...")
result = generate_scaffold(plan, out, bpm=130)
print(f"\n=== RESULT ===")
print(f"output:      {out.resolve()}")
print(f"duration:    {result['duration_s']:.2f}s")
print(f"size:        {result['size_bytes']} bytes")
print(f"bpm:         {result['bpm']}")
print(f"scene cuts:  {result['n_cuts']}")
