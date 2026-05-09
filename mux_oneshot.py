"""Oneshot: mux v1.mp4 + bgm_scaffold.wav → promo_v1_bgm_scaffold.mp4.
Verifies M4c works before MusicGen completes."""
import sys
from pathlib import Path

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from src.tools.bgm_mux import mux_bgm

run_dir = Path('workspace/football-match-simulator/runs/49aecf4a')
video = run_dir / 'outputs' / 'v1.mp4'
audio = run_dir / 'bgm' / 'bgm_scaffold.wav'
out = run_dir / 'outputs' / 'v1_bgm_scaffold.mp4'

print(f"mux {video.name} + {audio.name} → {out.name}")
result = mux_bgm(video, audio, out, audio_volume=0.7)
print(f"\n=== RESULT ===")
print(f"output:        {out.resolve()}")
print(f"duration:      {result['duration_s']:.2f}s")
print(f"size:          {result['size_bytes']} bytes  ({result['size_bytes']/1024/1024:.1f} MB)")
print(f"video codec:   {result['video_codec']}")
print(f"audio codec:   {result['audio_codec']}")
print(f"audio bitrate: {result['audio_bitrate']/1000:.0f} kbps")
