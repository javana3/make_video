"""Local score persistence (filename kept for backwards-compat — original
implementation pushed to Langfuse Scores API, which has been removed after
the Phoenix swap).

Phoenix has no equivalent simple "attach score to most-recent trace by
name" endpoint — its annotations API requires a span_id you don't easily
have when scoring runs asynchronously after the fact. So scores are kept
only in `<run_dir>/scores.jsonl` and viewed via the project's own
`/scores` UI page. Phoenix UI still shows the per-phase traces.

Module-level helpers `push_score` and `find_latest_trace_id` are retained
as no-ops so the existing call sites in quality_judge.py keep working;
they simply log and return False.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def find_latest_trace_id(name: str, limit: int = 10) -> Optional[str]:
    """No-op after Phoenix swap. Returns None so push_score skips the POST."""
    return None


def push_score(*,
                trace_id: Optional[str],
                name: str,
                value: float,
                comment: Optional[str] = None,
                data_type: str = "NUMERIC") -> tuple[bool, str]:
    """No-op after Phoenix swap; the local scores.jsonl is the source of truth.

    Returning False so callers' logging branches stay accurate ("skipped").
    """
    return False, "skipped: external score push disabled (Phoenix swap)"


def save_local_score(run_dir: Path, record: dict) -> None:
    """Append a score record to <run_dir>/scores.jsonl."""
    record["ts"] = datetime.now(timezone.utc).isoformat()
    path = run_dir / "scores.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_local_scores(run_dir: Path) -> list[dict]:
    """Read all scores from <run_dir>/scores.jsonl (newest first)."""
    path = run_dir / "scores.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    for ln in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out
