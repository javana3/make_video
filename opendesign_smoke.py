"""Smoke-test the OpenDesign daemon HTTP client end-to-end.

Pre-req: OpenDesign dev server is running. We read /tmp/opendesign_dev.log
to discover the daemon port.

Steps:
  1. health check
  2. pick available agent + a skill + a design system
  3. create a fresh project
  4. send a prompt, stream events
  5. once an `index.html` is produced, dump first 500 bytes
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from src.tools.opendesign_client import (
    OpenDesignEndpoint,
    create_project,
    download_archive,
    health_check,
    list_agents,
    list_design_systems,
    list_project_files,
    list_skills,
    pick_available_agent,
    read_artifact_bytes,
    send_prompt_stream,
)

import os, tempfile
dev_log_path = os.environ.get("OPENDESIGN_DEV_LOG") or str(
    Path(tempfile.gettempdir()) / "opendesign_dev.log"
)
ep = OpenDesignEndpoint.from_dev_log(Path(dev_log_path).read_text(encoding="utf-8"))
print(f"web:    {ep.web_url}")
print(f"daemon: {ep.daemon_url}")

print("\n=== health ===")
print(health_check(ep.daemon_url))

print("\n=== pick agent ===")
agent_id = pick_available_agent(ep.daemon_url, ["opencode", "claude", "cursor-agent"])
print(f"  → {agent_id}")

print("\n=== pick skill ===")
skills = list_skills(ep.daemon_url)
print(f"  total skills: {len(skills)}")
# Choose web-prototype as a sane default for a video-promo visual asset
skill_id = next(
    (s["id"] for s in skills if s.get("id") == "web-prototype"),
    skills[0]["id"] if skills else None,
)
print(f"  → {skill_id}")

print("\n=== pick design system ===")
dss = list_design_systems(ep.daemon_url)
print(f"  total DS: {len(dss)}")
ds_id = next(
    (d["id"] for d in dss if d.get("id") == "vercel"),
    dss[0]["id"] if dss else None,
)
print(f"  → {ds_id}")

print("\n=== create project ===")
project = create_project(
    ep.daemon_url,
    name="football-promo-smoke",
    skill_id=skill_id,
    design_system_id=ds_id,
)
pid = project["project_id"]
cid = project["conversation_id"]
print(f"  project_id:      {pid}")
print(f"  conversation_id: {cid}")

print("\n=== send prompt (SSE stream) ===")
prompt = (
    "Build a single-page hero for a football match simulator promo video. "
    "One headline (Chinese, max 8 chars), two subhead lines, "
    "black background with gold accents, no images needed. Keep it under 500 lines of HTML."
)
print(f"  prompt: {prompt}")
print("  --- events ---")
event_count = 0
got_artifact = False
t0 = time.monotonic()
for evt in send_prompt_stream(
    ep.daemon_url, pid, cid, prompt,
    agent_id=agent_id, skill_id=skill_id, design_system_id=ds_id,
    timeout=900,
):
    event_count += 1
    et = evt.get("event", "?")
    data = evt.get("data", {})
    elapsed = time.monotonic() - t0
    # Print event type + a tiny summary
    summary = ""
    if isinstance(data, dict):
        summary = " ".join(f"{k}={str(v)[:40]!r}" for k, v in list(data.items())[:3])
    print(f"  [{elapsed:5.1f}s] #{event_count:03d} event={et!r}  {summary[:200]}")
    # Heuristic: stop on terminal events
    if et in {"done", "complete", "run.complete", "run.done", "error"}:
        print(f"  → terminal event {et!r}, breaking")
        break
    if event_count > 200:
        print("  → too many events, breaking")
        break

print("\n=== files in project ===")
files = list_project_files(ep.daemon_url, pid)
for f in files:
    print(f"  - {f.get('name','?')}  size={f.get('size','?')}  kind={f.get('kind','?')}")

if any(f.get("name") == "index.html" for f in files):
    print("\n=== read index.html (first 500 bytes) ===")
    raw = read_artifact_bytes(ep.daemon_url, pid, "index.html")
    print(raw[:500].decode("utf-8", "replace"))
    print(f"  ... ({len(raw)} bytes total)")

    print("\n=== download archive → /tmp/test_html_asset/ ===")
    archive = download_archive(ep.daemon_url, pid, Path("/tmp/test_html_asset"))
    print(f"  extracted: {archive['n_files']} files, {archive['archive_bytes']/1024:.0f} KB")
    for f in archive["files"][:10]:
        print(f"    - {f}")
else:
    print("\n  ⚠ no index.html — agent may have failed; inspect events above")
