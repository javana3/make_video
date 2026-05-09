"""Agent 2 · SetupRunner — planner phase (M2a).

Reads a cloned repo and produces a structured setup plan: install commands,
seed commands, and services with health URLs. Plan is saved to disk and the
host executes it (with user approval gate) — see plan_executor.py.

This module ONLY plans; it does not run shell commands itself. The strict
separation lets the user review every command before anything runs.
"""
from __future__ import annotations

import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..observability.audit import traced_agent
from ..observability.logger import agent_logger
from ..tools.llm import anthropic_client, model_for


SYSTEM_PROMPT = """You are Agent 2 SetupRunner in a promo-video pipeline.

Phase 1 (project_brief) is approved. Your job in Phase 2a: figure out HOW to
launch this project locally so we can record it. You produce a structured
plan; the user reviews it; the host executes it.

Approach:
1. `list_dir` the repo root. Locate README and quickstart instructions.
2. Read the README — focus on `## Quick Start` / `## 快速启动` / `## 安装` /
   "How to run" sections. They usually list the exact commands.
3. Read `requirements.txt` / `package.json` / `pyproject.toml` / `Cargo.toml`
   to confirm the stack.
4. Identify ONE OR MORE services to start (e.g., backend API + static frontend).
5. Call `submit_plan` exactly once with the structured plan.

Plan rules (HOST ENFORCED — violation will return ERROR and you must retry):
- Every `cwd` MUST be a relative path inside the repo (no absolute paths,
  no `..`).
- Every `command` MUST start with one of: python, pip, npm, npx, node, yarn,
  pnpm, uvicorn, gunicorn, cargo, go, http-server (no `rm`, no `sudo`, no
  `curl | sh`, no destructive shell).
- Every service MUST have `port` (1024–65535) and `health_url` starting with
  `http://127.0.0.1:` or `http://localhost:` and pointing at that port.
- Use `python` (not `python3`) — host OS may not alias python3.
- If README says `python3 ...`, translate to `python ...` in your plan.

If a step is genuinely needed but cannot be expressed safely (e.g., requires
manual GUI install of a tool), put it in `manual_prereqs` instead of
`install_commands`.
"""


TOOLS = [
    {
        "name": "list_dir",
        "description": "List entries in a directory relative to the repo root.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file. Up to 64KB by default; large files truncated.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_bytes": {"type": "integer", "default": 65536},
            },
            "required": ["path"],
        },
    },
    {
        "name": "find_files",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "submit_plan",
        "description": (
            "Submit the final setup plan. The host will validate every command "
            "and reject if any rule is violated, forcing you to retry. Call "
            "exactly once when complete."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "1–2 sentence overview of the stack and what we're starting.",
                },
                "manual_prereqs": {
                    "type": "array",
                    "description": "Steps requiring human action before running this plan (e.g., 'Install Postgres'). Empty for fully-automated projects.",
                    "items": {"type": "string"},
                },
                "install_commands": {
                    "type": "array",
                    "description": "Dependency installation steps, run in order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "cwd": {"type": "string", "description": "Repo-relative directory."},
                            "purpose": {"type": "string"},
                        },
                        "required": ["command", "cwd", "purpose"],
                    },
                },
                "seed_commands": {
                    "type": "array",
                    "description": "Database/data seeding steps, run after install. Empty if none.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "cwd": {"type": "string"},
                            "purpose": {"type": "string"},
                        },
                        "required": ["command", "cwd", "purpose"],
                    },
                },
                "services": {
                    "type": "array",
                    "description": "Long-running services to start in background. ≥ 1.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "e.g. 'backend', 'frontend'."},
                            "command": {"type": "string"},
                            "cwd": {"type": "string"},
                            "port": {"type": "integer"},
                            "health_url": {"type": "string", "description": "HTTP URL on localhost that returns 2xx when the service is ready."},
                            "purpose": {"type": "string"},
                        },
                        "required": ["name", "command", "cwd", "port", "health_url", "purpose"],
                    },
                },
                "env_vars_needed": {
                    "type": "array",
                    "description": "Environment variables the user must set before running (e.g., API keys). Names only.",
                    "items": {"type": "string"},
                },
                "notes": {"type": "string"},
            },
            "required": ["summary", "install_commands", "services"],
        },
    },
]


_ALLOWED_CMD_PREFIXES = (
    "python", "pip", "npm", "npx", "node", "yarn", "pnpm",
    "uvicorn", "gunicorn", "fastapi",
    "cargo", "go", "rustc",
    "http-server", "serve",
)


def _safe_path(repo_dir: Path, rel: str) -> Path:
    rel = (rel or ".").lstrip("/").lstrip("\\") or "."
    p = (repo_dir / rel).resolve()
    repo_root = repo_dir.resolve()
    if p != repo_root and repo_root not in p.parents:
        raise ValueError(f"path escapes repo: {rel!r}")
    return p


def _tool_list_dir(repo_dir: Path, args: dict) -> str:
    p = _safe_path(repo_dir, args.get("path", "."))
    if not p.exists():
        return f"ERROR: {args['path']!r} does not exist"
    if not p.is_dir():
        return f"ERROR: {args['path']!r} is not a directory"
    entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    out = []
    for e in entries[:200]:
        if e.is_dir():
            out.append(f"DIR   {e.name}/")
        else:
            sz = e.stat().st_size
            out.append(f"FILE  {e.name}  ({sz}B)")
    return "\n".join(out) if out else "(empty)"


def _tool_read_file(repo_dir: Path, args: dict) -> str:
    p = _safe_path(repo_dir, args["path"])
    if not p.exists():
        return f"ERROR: {args['path']!r} does not exist"
    if not p.is_file():
        return f"ERROR: {args['path']!r} is not a file"
    max_bytes = int(args.get("max_bytes", 65536))
    raw = p.read_bytes()[:max_bytes]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    truncated = p.stat().st_size > max_bytes
    return text + (f"\n\n[... truncated; total {p.stat().st_size}B ...]" if truncated else "")


def _tool_find_files(repo_dir: Path, args: dict) -> str:
    matches = sorted(repo_dir.glob(args["pattern"]))[:100]
    if not matches:
        return "(no matches)"
    return "\n".join(str(m.relative_to(repo_dir)).replace("\\", "/") for m in matches)


def _validate_plan(repo_dir: Path, plan: dict) -> str:
    """Returns 'OK; ...' on pass, 'ERROR: ...' on fail."""
    if not isinstance(plan, dict):
        return "ERROR: plan is not an object"

    install = plan.get("install_commands") or []
    seed = plan.get("seed_commands") or []
    services = plan.get("services") or []

    if not isinstance(services, list) or len(services) < 1:
        return "ERROR: `services` must be a non-empty array (at least one service to start)."

    def check_cmd(label: str, c: dict, idx: int) -> Optional[str]:
        if not isinstance(c, dict):
            return f"{label}[{idx}] is not an object"
        cmd = (c.get("command") or "").strip()
        cwd = (c.get("cwd") or "").strip()
        if not cmd:
            return f"{label}[{idx}] missing 'command'"
        first = cmd.split()[0].lower()
        first_basename = first.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if not any(first_basename == p or first_basename.startswith(p) for p in _ALLOWED_CMD_PREFIXES):
            return (f"{label}[{idx}] command starts with {first!r} which is not in the allow-list "
                    f"{_ALLOWED_CMD_PREFIXES}. Use python/pip/npm/etc.")
        if "sudo " in cmd or " rm " in f" {cmd} " or "curl " in cmd and "| sh" in cmd:
            return f"{label}[{idx}] contains forbidden token (sudo/rm/curl|sh)"
        if not cwd:
            return f"{label}[{idx}] missing 'cwd'"
        if "/" in cwd and cwd.startswith("/") or re.match(r"^[A-Za-z]:[/\\\\]", cwd):
            return f"{label}[{idx}] cwd {cwd!r} is absolute; must be repo-relative"
        if ".." in Path(cwd).parts:
            return f"{label}[{idx}] cwd {cwd!r} escapes repo with '..'"
        try:
            target = _safe_path(repo_dir, cwd)
        except ValueError as e:
            return f"{label}[{idx}] cwd error: {e}"
        if not target.exists() or not target.is_dir():
            return f"{label}[{idx}] cwd {cwd!r} is not an existing directory in the repo"
        return None

    for i, c in enumerate(install):
        e = check_cmd("install_commands", c, i)
        if e:
            return f"ERROR: {e}. Fix and re-call submit_plan."
    for i, c in enumerate(seed):
        e = check_cmd("seed_commands", c, i)
        if e:
            return f"ERROR: {e}. Fix and re-call submit_plan."

    seen_ports = set()
    for i, s in enumerate(services):
        if not isinstance(s, dict):
            return f"ERROR: services[{i}] is not an object"
        for f in ("name", "command", "cwd", "port", "health_url"):
            if not s.get(f):
                return f"ERROR: services[{i}] missing '{f}'"
        e = check_cmd("services", s, i)
        if e:
            return f"ERROR: {e}. Fix and re-call submit_plan."
        port = s["port"]
        if not isinstance(port, int) or port < 1024 or port > 65535:
            return f"ERROR: services[{i}].port = {port!r} must be an integer in 1024..65535"
        if port in seen_ports:
            return f"ERROR: services[{i}].port = {port} duplicates another service"
        seen_ports.add(port)
        url = s["health_url"]
        if not (url.startswith("http://127.0.0.1:") or url.startswith("http://localhost:")):
            return f"ERROR: services[{i}].health_url = {url!r} must start with http://127.0.0.1: or http://localhost:"
        if str(port) not in url:
            return (f"ERROR: services[{i}].health_url {url!r} does not contain the service "
                    f"port {port}; they must agree.")

    return f"OK; plan validated ({len(install)} install / {len(seed)} seed / {len(services)} services)."


def _build_initial_message(repo_dir: Path, project_brief: Optional[str] = None) -> str:
    parts = [
        f"Repo path: {repo_dir}",
        f"Host OS:   {platform.system()} ({platform.platform()})",
        "",
    ]
    if project_brief:
        parts.append("Project brief (Phase 1, for context):")
        parts.append("```markdown")
        parts.append(project_brief[:3000])
        parts.append("```")
        parts.append("")
    parts.append("Begin: list the repo root, read README, identify the stack and "
                 "the commands to run. Then call `submit_plan`.")
    return "\n".join(parts)


@traced_agent("Agent 2 SetupRunner · plan", phase=2)
def run_planner(repo_dir: Path,
                output_path: Path,
                project_brief: Optional[str] = None,
                feedback: Optional[str] = None,
                progress_path: Optional[Path] = None,
                max_steps: int = 20) -> Path:
    """Run the planner agent. Writes setup_plan.json and returns its path."""
    import time as _time

    log = agent_logger("agent2_setup")
    client = anthropic_client()
    model = model_for("reasoning")
    started_at = datetime.now(timezone.utc)
    started_mono = _time.monotonic()
    log.info(f"start  repo={repo_dir.name}  model={model}")

    files_read: list[str] = []
    tool_call_count = 0

    def write_progress(step: int, last_action: str,
                       status: str = "running", error: Optional[str] = None) -> None:
        if progress_path is None:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "phase": "2a-plan",
            "status": status,
            "step": step,
            "max_steps": max_steps,
            "last_action": last_action,
            "files_read": list(files_read),
            "tool_calls": tool_call_count,
            "started_at": started_at.isoformat(),
            "last_update": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(_time.monotonic() - started_mono, 1),
            "error": error,
        }
        progress_path.write_text(json.dumps(payload, ensure_ascii=False),
                                 encoding="utf-8")

    write_progress(0, "starting")

    initial = _build_initial_message(repo_dir, project_brief)
    if feedback:
        initial += f"\n\nUser feedback on previous plan:\n{feedback}\nProduce a revised plan."
    messages = [{"role": "user", "content": initial}]

    final_plan: Optional[dict] = None

    for step in range(max_steps):
        log.info(f"step {step+1}/{max_steps} → LLM")
        write_progress(step + 1, "→ LLM (waiting response)")
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        log.info(f"step {step+1} ← stop_reason={resp.stop_reason}  "
                 f"in={resp.usage.input_tokens} out={resp.usage.output_tokens}")
        messages.append({"role": "assistant", "content": resp.content})

        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        text_blocks = [b for b in resp.content if getattr(b, "type", None) == "text"]
        for tb in text_blocks:
            if tb.text and tb.text.strip():
                log.info(f"  agent: {tb.text.strip()[:300]}")

        if not tool_uses:
            log.info(f"step {step+1}: no tool calls; ending loop")
            break

        tool_results = []
        for tu in tool_uses:
            log.info(f"  tool call: {tu.name}({tu.input})")
            tool_call_count += 1
            short_arg = ""
            try:
                if tu.name == "list_dir":
                    short_arg = tu.input.get("path", "") or "."
                    out = _tool_list_dir(repo_dir, tu.input)
                elif tu.name == "read_file":
                    short_arg = tu.input.get("path", "")
                    out = _tool_read_file(repo_dir, tu.input)
                    if not out.startswith("ERROR"):
                        rel = short_arg.replace("\\", "/").lstrip("/")
                        if rel and rel not in files_read:
                            files_read.append(rel)
                elif tu.name == "find_files":
                    short_arg = tu.input.get("pattern", "")
                    out = _tool_find_files(repo_dir, tu.input)
                elif tu.name == "submit_plan":
                    out = _validate_plan(repo_dir, tu.input)
                    if out.startswith("OK"):
                        final_plan = dict(tu.input)
                else:
                    out = f"ERROR: unknown tool {tu.name}"
            except Exception as e:
                out = f"ERROR: {type(e).__name__}: {e}"
                log.exception(f"tool {tu.name} failed")
            if len(out) > 50000:
                out = out[:50000] + f"\n\n[... tool_result truncated; full size {len(out)}B ...]"
            write_progress(step + 1, f"{tu.name}({short_arg})" if short_arg else tu.name)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": out,
            })

        messages.append({"role": "user", "content": tool_results})

        if final_plan is not None:
            log.info("submit_plan accepted → finalizing")
            break
    else:
        write_progress(max_steps, "FAILED: no plan emitted", status="error",
                       error="agent finished without submit_plan")
        raise RuntimeError("Agent 2 finished without submitting a plan")

    if not final_plan:
        write_progress(max_steps, "FAILED: no plan emitted", status="error",
                       error="agent loop exited but no plan recorded")
        raise RuntimeError("Agent 2 finished without submitting a plan")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(final_plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(f"plan written: {output_path}")
    write_progress(max_steps, "completed", status="done")
    return output_path
