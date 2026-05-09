"""M-1 smoke test.

Verifies the observability stack wires up end-to-end:
  1. Langfuse OTLP exporter set up + instruments Anthropic SDK
  2. loguru sinks console + JSONL
  3. EventBus writes events.jsonl + Langfuse span events
  4. @traced_agent wraps an Agent function in a span
  5. tools.shell.run() captures output + logs

Run:
    python tests/test_observability_smoke.py [--keep-alive]
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

# Windows console (GBK) chokes on emoji in third-party prints; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.observability.tracer import setup, langfuse_url
from src.observability.logger import setup_logging, agent_logger
from src.observability.events import EventBus
from src.observability.audit import traced_agent, set_run_context
from src.tools.shell import run as shell_run


def main(keep_alive: bool = False) -> int:
    run_id = uuid.uuid4().hex[:8]
    repo_root = Path(__file__).resolve().parent.parent
    run_dir = repo_root / "workspace" / "smoke" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n== M-1 smoke test ==")
    print(f"run_id  = {run_id}")
    print(f"run_dir = {run_dir}\n")

    # 1. Tracing
    setup(project_name="smoke-test", launch_ui=True)
    print(f"[1/5] tracing OK   — Langfuse at {langfuse_url()}")

    # 2. Logging
    setup_logging(run_dir)
    log = agent_logger("smoke")
    log.info("logging configured")
    print("[2/5] logging OK")

    # 3. Event bus + run context
    bus = EventBus(run_dir, run_id)
    set_run_context(run_id, bus, run_dir)
    bus.emit("agent_start", agent="smoke-driver", note="M-1 smoke test")
    print("[3/5] event bus OK")

    # 4. Shell wrapper
    result = shell_run(["python", "--version"])
    assert result.ok, f"shell failed: {result.stderr}"
    py_version = (result.stdout or result.stderr).strip()
    print(f"[4/5] shell wrapper OK — captured: {py_version}")

    # 5. @traced_agent
    @traced_agent("DummyAgent", phase=0)
    def dummy_agent(x: str) -> str:
        log.info(f"dummy_agent called with {x!r}")
        shell_run(["python", "-c", "print('hello from inside dummy')"])
        return f"processed:{x}"

    result = dummy_agent("smoke")
    assert result == "processed:smoke"
    assert hasattr(dummy_agent, "__traced_agent__")
    print(f"[5/5] @traced_agent OK — result: {result!r}")

    bus.emit("agent_done", agent="smoke-driver", status="pass")

    # ---- Verify artifacts ----
    events_file = run_dir / "events.jsonl"
    assert events_file.exists(), "events.jsonl missing"
    events = [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_events = {"agent_start", "agent_done"}
    seen_events = {e["event"] for e in events}
    assert expected_events <= seen_events, f"missing events: {expected_events - seen_events}"

    pipeline_log = run_dir / "logs" / "pipeline.jsonl"
    assert pipeline_log.exists(), "pipeline.jsonl missing"
    assert pipeline_log.stat().st_size > 0, "pipeline.jsonl is empty"

    smoke_log = run_dir / "logs" / "smoke.jsonl"
    assert smoke_log.exists(), "smoke.jsonl missing"

    print(f"\n=== events ({len(events)} total) ===")
    for evt in events:
        ts = evt["ts"][:19]
        agent = (evt.get("agent") or "-")[:20]
        print(f"  {ts}  {evt['event']:<14}  {agent}")

    print(f"\nartifacts:")
    print(f"  events:        {events_file}  ({events_file.stat().st_size} B)")
    print(f"  pipeline log:  {pipeline_log}  ({pipeline_log.stat().st_size} B)")
    print(f"  agent log:     {smoke_log}  ({smoke_log.stat().st_size} B)")

    print(f"\n✅ M-1 smoke test passed")
    print(f"   Langfuse UI: {langfuse_url()}")

    if keep_alive:
        print(f"\nPress Enter to exit (webserver will exit)...")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
    return 0


if __name__ == "__main__":
    keep = "--keep-alive" in sys.argv
    sys.exit(main(keep_alive=keep))
