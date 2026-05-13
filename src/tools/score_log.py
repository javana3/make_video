"""Local score persistence — one JSONL row per QualityJudge / user rating.

Lives at `<run_dir>/scores.jsonl` and is the source of truth viewed via
the project's own /scores UI page. We do NOT push scores to an external
observability service: Phoenix's annotation/evaluation APIs require a
`span_id` that isn't readily available when QualityJudge runs async
after the agent's trace has already closed, so the cost of reverse-
engineering that integration outweighed the value (the /scores page
already shows everything the user needs, and Phoenix UI still shows the
per-phase traces themselves).
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path


def save_local_score(run_dir: Path, record: dict) -> None:
    """Append one score record to <run_dir>/scores.jsonl."""
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
