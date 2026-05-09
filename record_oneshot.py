"""One-shot: record localhost:5500 via Playwright (browser-internal capture).
No screen grabbing, no occlusion possible — you can use any other app freely."""
import sys
from pathlib import Path

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from src.tools.web_recorder import record_url

URL = "http://127.0.0.1:5500/index.html"
DURATION_S = 60.0

out_path = Path('workspace/football-match-simulator/runs/49aecf4a/recordings/test.mp4')
state_path = out_path.parent / 'test_state.json'

print(f"Recording {URL} for {DURATION_S}s via Playwright (headless Chromium) ...")
print(f"You can use other apps — recording is browser-internal.")
result = record_url(
    url=URL,
    duration_s=DURATION_S,
    output_path=out_path,
    state_path=state_path,
    width=1280,
    height=800,
    headless=True,
)

print(f"\n=== RESULT ===")
print(f"status:      {result.status}")
print(f"output:      {out_path.resolve()}")
print(f"file size:   {out_path.stat().st_size if out_path.exists() else 0} bytes")
if result.ffprobe:
    p = result.ffprobe
    print(f"duration:    {p.get('duration', 0):.2f}s")
    print(f"resolution:  {p.get('width', 0)}x{p.get('height', 0)}")
    print(f"codec:       {p.get('video_codec')}")
if result.error:
    print(f"\nERROR: {result.error[:1000]}")
