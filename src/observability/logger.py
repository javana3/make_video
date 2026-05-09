"""Structured logging via loguru.

Sinks:
  - console (colored, human-readable)
  - <run_dir>/logs/pipeline.jsonl (everything, JSON serialized)
  - <run_dir>/logs/<agent>.jsonl (per-agent filtered, on demand via agent_logger)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

from loguru import logger

_CONFIGURED = False
_AGENT_SINKS: Dict[str, int] = {}
_RUN_DIR: Path | None = None


def setup_logging(run_dir: Path) -> None:
    """Configure console + pipeline.jsonl sinks. Re-runnable per run_dir."""
    global _CONFIGURED, _RUN_DIR, _AGENT_SINKS

    if _RUN_DIR == run_dir and _CONFIGURED:
        return

    _RUN_DIR = run_dir
    _AGENT_SINKS.clear()
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format=("<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <7}</level> | "
                "<cyan>{extra[agent]:<22}</cyan> | "
                "{message}"),
    )

    logger.add(
        log_dir / "pipeline.jsonl",
        level="DEBUG",
        serialize=True,
        encoding="utf-8",
    )

    logger.configure(extra={"agent": "pipeline"})
    _CONFIGURED = True


def agent_logger(agent_name: str):
    """Return a logger bound to `agent_name` with its own JSONL sink.

    If setup_logging() hasn't run yet (e.g. standalone smoke scripts), fall back
    to plain bound loguru without a per-agent JSONL sink — caller still gets a
    working logger; just no isolated file. Avoids hard-failing imports.
    """
    if _RUN_DIR is None:
        return logger.bind(agent=agent_name)

    if agent_name not in _AGENT_SINKS:
        log_dir = _RUN_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        sink_id = logger.add(
            log_dir / f"{agent_name}.jsonl",
            level="DEBUG",
            serialize=True,
            encoding="utf-8",
            filter=lambda record, name=agent_name: record["extra"].get("agent") == name,
        )
        _AGENT_SINKS[agent_name] = sink_id

    return logger.bind(agent=agent_name)
