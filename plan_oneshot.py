"""One-shot: run Agent 3 cutting planner against current run + recording."""
import json, sys
from pathlib import Path

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from src.tools.dotenv import load_dotenv
from src.tools.shell import run as shell_run
from src.tools.ffbin import ffprobe
from src.agents.remotion_composer import run_cutting_planner
from src.observability.logger import setup_logging

load_dotenv()
setup_logging(Path('workspace/football-match-simulator/runs/49aecf4a'))

run_dir = Path('workspace/football-match-simulator/runs/49aecf4a')
brief = (run_dir / 'project_brief.md').read_text(encoding='utf-8')
recording = run_dir / 'recordings/test.mp4'

# probe recording
probe_result = shell_run([ffprobe(), '-v', 'quiet', '-print_format', 'json',
                          '-show_streams', '-show_format', str(recording)], check=True)
data = json.loads(probe_result.stdout)
v = next(s for s in data['streams'] if s['codec_type'] == 'video')
recording_meta = {
    'source_path': 'recordings/test.mp4',
    'duration_s': float(data['format']['duration']),
    'width': int(v['width']),
    'height': int(v['height']),
    'codec': v['codec_name'],
    'fps': v['r_frame_rate'],
}
print(f"recording: {recording_meta['duration_s']:.1f}s {recording_meta['width']}x{recording_meta['height']} {recording_meta['codec']}")

out = run_dir / 'cutting_plan.json'
result = run_cutting_planner(
    run_dir=run_dir,
    project_brief=brief,
    recording_meta=recording_meta,
    output_path=out,
    progress_path=run_dir / 'progress.json',
)
print(f"\n=== cutting_plan.json ===\n{out.read_text(encoding='utf-8')}")
