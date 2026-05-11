"""Per-run system prompt override.

Each agent has its built-in default SYSTEM_PROMPT in its .py module. This
helper lets the web UI store an override at `<run_dir>/prompts/<key>.txt`
that the agent reads at startup time. No override = use the default.

This is NOT a state machine — it's a deterministic file lookup that lets the
USER (not us) replace agent instructions. The agent itself does not branch
on this; it just receives whatever prompt the user wrote.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional


def get_system_prompt(agent_key: str, default: str,
                      run_dir: Optional[Path] = None) -> str:
    """Return the effective system prompt for an agent in a given run.

    Resolution order:
      1. `<run_dir>/prompts/<agent_key>.txt` if it exists and is non-empty
      2. the module-level default string

    agent_key uses dotted lowercase form: 'project_analyzer', 'setup_runner',
    'demo_driver', 'remotion_composer', 'voice_over', 'opendesigner'.
    """
    if run_dir is None:
        return default
    override = run_dir / "prompts" / f"{agent_key}.txt"
    if override.exists():
        try:
            txt = override.read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:
            pass
    return default


def save_override(agent_key: str, prompt: str, run_dir: Path) -> Path:
    """Write a per-run prompt override. Empty string deletes the override."""
    override_dir = run_dir / "prompts"
    override_dir.mkdir(parents=True, exist_ok=True)
    override = override_dir / f"{agent_key}.txt"
    if not prompt.strip():
        if override.exists():
            override.unlink()
        return override
    override.write_text(prompt, encoding="utf-8")
    return override


def list_overrides(run_dir: Path) -> dict[str, bool]:
    """Return {agent_key: True if override exists else False} for known agents."""
    known_agents = [
        "project_analyzer", "setup_runner", "demo_driver",
        "remotion_composer", "voice_over", "opendesigner",
    ]
    override_dir = run_dir / "prompts"
    return {
        k: (override_dir / f"{k}.txt").exists() for k in known_agents
    }
