"""M0 smoke test: Pipeline state machine + state.json persistence + manifest."""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import Pipeline
from src.types import PipelineState


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    workspace = repo_root / "workspace"
    project = "m0_smoke"
    run_id = uuid.uuid4().hex[:8]

    print(f"\n== M0 pipeline smoke ==")
    print(f"project = {project}")
    print(f"run_id  = {run_id}\n")

    # 1. init
    pipe = Pipeline(project=project, run_id=run_id, workspace_root=workspace,
                    launch_phoenix_ui=False)  # don't start UI in test
    assert pipe.state.phase == 1
    assert pipe.state.gate == "running"
    print(f"[1/5] init OK            run_dir = {pipe.run_dir}")

    # 2. transition phase 1 → gate waiting_brief_approval
    pipe.transition(gate="waiting_brief_approval")
    assert pipe.state.gate == "waiting_brief_approval"
    pipe.gate_pass("waiting_brief_approval", reason="user approved brief")
    pipe.transition(phase=2, gate="waiting_html")
    assert pipe.state.phase == 2
    print(f"[2/5] transition OK      phase={pipe.state.phase} gate={pipe.state.gate}")

    # 3. record asset
    fake_recording = workspace / project / "fake_recording.webm"
    fake_recording.parent.mkdir(parents=True, exist_ok=True)
    fake_recording.write_bytes(b"\x00" * 1024)
    pipe.record_asset("recording_v0", fake_recording, verified=True,
                      duration=300.5, resolution="1440p")
    assert "recording_v0" in pipe.state.manifest
    assert pipe.state.manifest["recording_v0"]["verified"] is True
    print(f"[3/5] record_asset OK    manifest has {len(pipe.state.manifest)} entry")

    # 4. state.json persisted
    assert pipe.state_file.exists()
    data = json.loads(pipe.state_file.read_text(encoding="utf-8"))
    assert data["run_id"] == run_id
    assert data["phase"] == 2
    assert data["gate"] == "waiting_html"
    assert "recording_v0" in data["manifest"]
    print(f"[4/5] state.json OK      {pipe.state_file}")

    # 5. reload from state.json
    pipe2 = Pipeline(project=project, run_id=run_id, workspace_root=workspace,
                     launch_phoenix_ui=False)
    assert pipe2.state.phase == 2
    assert pipe2.state.gate == "waiting_html"
    assert "recording_v0" in pipe2.state.manifest
    print(f"[5/5] reload OK          phase={pipe2.state.phase} gate={pipe2.state.gate}")

    # events.jsonl sanity
    events_file = pipe.run_dir / "events.jsonl"
    events = [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    types = {e["event"] for e in events}
    assert "gate_enter" in types
    assert "gate_pass" in types
    assert "asset_verified" in types

    print(f"\n=== events.jsonl ({len(events)} total) ===")
    for evt in events:
        print(f"  {evt['ts'][:19]}  {evt['event']:<16}  {evt.get('payload', {})}")

    print(f"\n[OK] M0 pipeline smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
