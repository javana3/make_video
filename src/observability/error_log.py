"""Per-run errors.jsonl logger — separate from events.jsonl for easy grep.

Schema (one JSON per line):
  {
    "ts": ISO-8601 UTC,
    "run_id": str,
    "agent": str          # which agent (project_analyzer / setup_runner / ...)
    "step_label": str     # what step (e.g. "step 5 LLM call")
    "attempt": int        # which retry attempt this is (1-indexed)
    "max_attempts": int   # total attempts allowed
    "error_type": str     # exception class name
    "error_text": str     # truncated str(e)
    "escalated": bool     # True if this is the final attempt that triggers ErrorAgent
    "context_hint": dict  # optional small context (model, step number, etc.)
  }
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def log_error(run_dir: Path, *,
              agent: str,
              step_label: str,
              attempt: int,
              max_attempts: int,
              error: BaseException,
              escalated: bool = False,
              context_hint: Optional[dict] = None,
              error_text_max: int = 2000) -> None:
    """Append one error entry to <run_dir>/errors.jsonl."""
    run_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "agent": agent,
        "step_label": step_label,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "error_type": type(error).__name__,
        "error_text": str(error)[:error_text_max],
        "escalated": escalated,
        "context_hint": context_hint or {},
    }
    path = run_dir / "errors.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_errors(run_dir: Path, limit: int = 200) -> list[dict]:
    """Return last `limit` error entries (newest first)."""
    path = run_dir / "errors.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict] = []
    for ln in reversed(lines):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
        if len(out) >= limit:
            break
    return out


def read_escalations(run_dir: Path) -> list[dict]:
    """Only the errors that escalated to ErrorAgent (attempt == max_attempts)."""
    return [e for e in read_errors(run_dir, limit=1000) if e.get("escalated")]
