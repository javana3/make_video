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
launch this project locally — and DO IT YOURSELF. Detect every missing tool,
write commands that install them automatically, write the config files, start
the services. The human's ONLY job is to fill in secret values (e.g. an API
key for a third-party service we have no substitute for). NOTHING ELSE.

═══════════════════════════════════════════════════════════════════
ITERATION ORDER (do not skip steps)
═══════════════════════════════════════════════════════════════════

1. `list_dir` repo root. `read_file` README + quickstart.
2. `read_file` the package manifest (pyproject.toml / package.json / Cargo.toml
   / go.mod / Gemfile / requirements.txt …) to learn the stack.
3. `find_files` for `*.example.{yaml,yml,toml,json,env}`, `.env.example`,
   `config.example.*`, `default_config.*`. Every example file means the
   project expects a real config to exist — YOU produce it via config_writes.
4. For EVERY system tool the project depends on (node, python, cargo, go,
   bun, deno, docker, redis-cli, postgres, ffmpeg, uv, …) call
   `check_tool(name)`. The result is `INSTALLED: <version>` or
   `NOT_FOUND: …`. Do this BEFORE deciding to install anything.
5. For each NOT_FOUND tool: write an `install_commands` entry using the
   host OS's package manager. Examples:
     Windows: `winget install OpenJS.NodeJS.LTS` / `winget install
              Rustlang.Rustup` / `winget install PostgreSQL.PostgreSQL` /
              `scoop install bun` / `choco install python --version=3.11`
     macOS:   `brew install node` / `brew install rust` / `brew install
              postgresql@16`
     Linux:   `apt-get install -y nodejs` / `curl -sSf https://sh.rustup.rs |
              sh` (note: validator rejects `curl|sh` — prefer `rustup-init`
              or the OS package). Detect OS first if needed; the env
              section of the user message tells you the platform.
6. Identify ONE OR MORE services to start. Pick health URLs.
7. Call `submit_plan` exactly once.

═══════════════════════════════════════════════════════════════════
THE ONLY USER TOUCHPOINT: user_secrets_needed
═══════════════════════════════════════════════════════════════════

If the project needs an external secret (API key for a service the parent
pipeline does not have a substitute for — RunningHub, ComfyUI cloud, your
own OpenAI account, a database password the user picks), declare it in
`user_secrets_needed`. The executor will pause, ask the user to FILL IN
that value, then substitute `${VAR_NAME}` in your `config_writes`.

Format:
  user_secrets_needed: [
    {"var_name": "RUNNINGHUB_API_KEY",
     "description": "RunningHub.ai cloud workflows API key — sign up at runninghub.ai/profile to get one.",
     "why_needed": "Pixelle's default video pipeline uses RunningHub-hosted ComfyUI; without this key the image/video generation tabs will fail."},
    {"var_name": "DB_PASSWORD",
     "description": "Password for the local Postgres instance (any value, e.g. 'devpass').",
     "why_needed": "psql connection string in config.yaml needs this."}
  ]

Each entry must answer: what var, what's it for, where can user get it.
Use `user_secrets_needed` SPARINGLY — only when there's truly no auto path
AND no parent credential covers it.

The PARENT pipeline already provides these credentials — use them via
${PARENT_VAR} placeholders in config_writes WITHOUT touching user_secrets:

  ${ARK_KEY_1}, ${ANTHROPIC_API_KEY}, ${ANTHROPIC_BASE_URL},
  ${ARK_BASE_URL_OPENAI}, ${ARK_BASE_URL_ANTHROPIC},
  ${LLM_REASONING}, ${LLM_FAST}, ${LLM_DEEP}, ${LLM_VISION},
  ${MINIMAX_API_KEY}, ${MINIMAX_BASE_URL}, ${MINIMAX_MUSIC_MODEL}

So: if a project needs "LLM api_key + base_url" → use ${ARK_KEY_1} +
${ARK_BASE_URL_OPENAI}, NO user_secrets_needed.

═══════════════════════════════════════════════════════════════════
SCHEMA RULES (host-enforced; violation = ERROR + retry)
═══════════════════════════════════════════════════════════════════

- Every `cwd` MUST be a relative path INSIDE the repo (no absolute, no `..`).
- Every `service` MUST have integer `port` (1024–65535) and `health_url`
  on `http://127.0.0.1:<port>/...` or `http://localhost:<port>/...`.
- Every `config_writes[].content_template` may use ${VAR} placeholders;
  VARs must be either in PARENT_CREDENTIALS or declared in
  user_secrets_needed.
- Commands ARE NOT prefix-restricted — use any tool the project needs.
- Real safety bans: no `rm -rf /`, no `sudo`, no `format C:`, no fork
  bombs, no blind `curl URL | sh`, no `shutdown`/`reboot`. The validator
  will reject these and you must rewrite.

═══════════════════════════════════════════════════════════════════
TONE & SCOPE
═══════════════════════════════════════════════════════════════════

You are the engineer. The user is the boss who CHECKED OUT a repo and now
expects it to run. Do not say "please install …" / "the user must …" /
"manually …". If you wrote that, you're failing. Either:
  (a) call `check_tool`, find out it's already there → skip,
  (b) install it with a real command, or
  (c) it's a SECRET → put it in user_secrets_needed.
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
        "name": "check_tool",
        "description": (
            "Probe whether a system tool is installed on the host. Runs "
            "`<name> --version` (or a variant) and returns the output. Use "
            "BEFORE deciding whether something is a real prereq. If "
            "check_tool reports the tool is present, DO NOT list it in "
            "manual_prereqs and DO NOT add an install step for it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string",
                          "description": "Tool basename (e.g. 'node', 'cargo', 'uv', 'docker', 'go', 'rustc', 'bun', 'winget')."},
            },
            "required": ["name"],
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
                    "description": (
                        "Almost always EMPTY []. Only list a step here when "
                        "the agent has no path to do it programmatically AND "
                        "the user touching it is truly the only option (e.g. "
                        "'install NVIDIA driver from nvidia.com and reboot'). "
                        "If a tool can be installed via winget/choco/brew/apt/"
                        "scoop or downloaded by a script, use install_commands "
                        "instead. NEVER use manual_prereqs for 'install node', "
                        "'install ffmpeg' etc — call check_tool first; if "
                        "missing, install via the OS package manager."
                    ),
                    "items": {"type": "string"},
                },
                "user_secrets_needed": {
                    "type": "array",
                    "description": (
                        "The ONLY user-touchpoint. Declare external secrets "
                        "the agent cannot derive AND no parent credential "
                        "covers (e.g. RunningHub API key, user's own OpenAI "
                        "key). Executor pauses and prompts user to fill these "
                        "values BEFORE running config_writes. Use sparingly — "
                        "if PARENT_CREDENTIALS has a substitute, prefer that."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "var_name": {"type": "string", "description": "Env var name; used as ${VAR_NAME} in config_writes templates."},
                            "description": {"type": "string", "description": "Plain-language what+where for the user (so they know what to paste)."},
                            "why_needed": {"type": "string", "description": "Which feature breaks if missing — so user knows whether they can skip."},
                        },
                        "required": ["var_name", "description", "why_needed"],
                    },
                },
                "config_writes": {
                    "type": "array",
                    "description": (
                        "Config files THE AGENT writes before install runs, to "
                        "satisfy the project's *.example.* templates. "
                        "content_template uses ${PARENT_VAR} placeholders that "
                        "get substituted from the parent pipeline's env at "
                        "execute time. Example: write config.yaml with "
                        "${ARK_KEY_1} for an LLM api_key field."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Repo-relative file path to write."},
                            "content_template": {"type": "string", "description": "File content; ${VAR} placeholders allowed."},
                            "purpose": {"type": "string"},
                        },
                        "required": ["path", "content_template", "purpose"],
                    },
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
                "notes": {"type": "string"},
            },
            "required": ["summary", "install_commands", "services"],
        },
    },
]


# ─── HARD CONSTRAINTS for the no-punt validator ───────────────────────────
# Credentials the parent pipeline can substitute into config_writes templates.
# Keep in sync with .env / .env.example. Single source of truth.
PARENT_CREDENTIALS = {
    "ARK_KEY_1": "火山方舟 Coding Plan key — primary LLM credential (Anthropic + OpenAI compat).",
    "ANTHROPIC_API_KEY": "Same value as ARK_KEY_1 — for projects that read ANTHROPIC_API_KEY by convention.",
    "ANTHROPIC_BASE_URL": "https://ark.cn-beijing.volces.com/api/coding (Anthropic-compat endpoint).",
    "ARK_BASE_URL_OPENAI": "https://ark.cn-beijing.volces.com/api/coding/v3 (OpenAI-compat endpoint, use for projects expecting an openai SDK base_url).",
    "ARK_BASE_URL_ANTHROPIC": "Same as ANTHROPIC_BASE_URL.",
    "LLM_REASONING": "glm-5.1 — default model id usable on either endpoint above.",
    "LLM_FAST": "deepseek-v3.2 — fast/cheap model id for routing calls.",
    "LLM_DEEP": "minimax-m2.7 — deep reasoning fallback model id.",
    "MINIMAX_API_KEY": "MiniMax 国内 token plan key (sk-cp-…) — for native MiniMax music/T2A endpoints (api.minimaxi.com).",
    "MINIMAX_BASE_URL": "https://api.minimaxi.com (MiniMax 国内 host).",
    "MINIMAX_MUSIC_MODEL": "music-2.6 (MiniMax music-gen model id).",
}

# NO whitelist on manual_prereqs and NO punt-phrase regex. They were state
# machines pretending to be safety: any project the whitelist didn't cover
# (Rust / Go / Postgres / bun / Ollama / CUDA / scoop / brew / etc) the
# agent had no path to install. Now: agent CHECKS what's installed via
# the `check_tool` tool, writes auto-install commands using any package
# manager (winget/choco/brew/apt/scoop/curl), and only lists manual_prereqs
# when truly impossible (e.g., GPU driver requiring physical reboot).

# Commands the EXECUTOR blocks unconditionally — these are real safety,
# not "agent must use python only". Anything destructive or escalating.
_BLOCKED_COMMAND_PATTERNS = [
    re.compile(r"\brm\s+-rf?\s+/", re.IGNORECASE),  # rm -rf / or rm -r /
    re.compile(r"\brm\s+.*\\\\", re.IGNORECASE),    # rm with windows-style abs path
    re.compile(r"\bsudo\b", re.IGNORECASE),         # require user, opaque to UI
    re.compile(r"\bsu\s", re.IGNORECASE),
    re.compile(r"\bdoas\b", re.IGNORECASE),
    re.compile(r"\bformat\s+[A-Z]:", re.IGNORECASE),  # format C:
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\s+if=.+of=/dev/", re.IGNORECASE),
    re.compile(r":\(\)\{\s*:\|:&\s*\}", re.IGNORECASE),  # fork bomb
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bcurl\b.+\|\s*(bash|sh|zsh)\b", re.IGNORECASE),  # blind pipe to shell
    re.compile(r"\bwget\b.+\|\s*(bash|sh|zsh)\b", re.IGNORECASE),
]

# Files in the repo that signal "this project needs config" — if present and
# the plan has no config_writes covering them, validator rejects.
_CONFIG_EXAMPLE_GLOBS = (
    "*.example.yaml", "*.example.yml", "*.example.toml",
    "*.example.json", "*.example.env", ".env.example",
    "config.example.*", "default.config.*",
)


def _detect_config_examples(repo_dir: Path) -> list[Path]:
    hits: list[Path] = []
    for pat in _CONFIG_EXAMPLE_GLOBS:
        hits.extend(repo_dir.glob(pat))
        # also one level deep
        hits.extend(repo_dir.glob(f"*/{pat}"))
    seen: set[Path] = set()
    out: list[Path] = []
    for h in hits:
        if h.is_file() and h not in seen:
            seen.add(h)
            out.append(h)
    return out


def _expected_real_path(example_path: Path) -> str:
    """Map config.example.yaml → config.yaml; .env.example → .env."""
    name = example_path.name
    if name == ".env.example":
        return ".env"
    # foo.example.yaml → foo.yaml ;  config.example.* → config.*
    name = name.replace(".example.", ".", 1).replace("example.", "", 1)
    if name.startswith("default."):
        name = name[len("default."):]
    return name


_PLACEHOLDER_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


# NO command prefix whitelist. The previous list (python/pip/npm/...) blocked
# winget/choco/brew/apt/scoop/curl — meaning agent literally had no way to
# install Rust, Go, Postgres, bun, etc. on a fresh box. Now ANY command is
# allowed at the schema layer; _BLOCKED_COMMAND_PATTERNS handles real safety
# (rm -rf /, sudo, format, fork bomb, curl|sh, …).


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


def _tool_check_tool(args: dict) -> str:
    """Probe a system tool. Returns first non-empty line of stdout/stderr.

    Tries `<name> --version` first, falls back to `<name> -v` then bare `<name>`.
    Safe: only runs the literal tool name + a version flag; rejects anything
    with shell metachars in name.
    """
    import shlex
    import subprocess
    name = (args.get("name") or "").strip()
    if not name or not re.match(r"^[A-Za-z0-9_.+-]+$", name):
        return f"ERROR: invalid tool name {name!r} (alphanum+._+- only)"
    # Try a few common version-flag variants.
    variants = [
        [name, "--version"],
        [name, "-V"],
        [name, "-v"],
        [name, "version"],
    ]
    for cmd in variants:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                                  shell=False)
        except FileNotFoundError:
            return f"NOT_FOUND: {name!r} is not on PATH"
        except subprocess.TimeoutExpired:
            continue
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
        out = (r.stdout or "").strip() or (r.stderr or "").strip()
        if r.returncode == 0 or out:
            head = out.splitlines()[0] if out else "(empty output)"
            return f"INSTALLED: {head} [via `{' '.join(cmd)}` rc={r.returncode}]"
    return f"NOT_FOUND: {name!r} responds to no version flag"


def _validate_plan(repo_dir: Path, plan: dict) -> str:
    """Returns 'OK; ...' on pass, 'ERROR: ...' on fail."""
    if not isinstance(plan, dict):
        return "ERROR: plan is not an object"

    install = plan.get("install_commands") or []
    seed = plan.get("seed_commands") or []
    services = plan.get("services") or []
    manual = plan.get("manual_prereqs") or []
    config_writes = plan.get("config_writes") or []
    notes = plan.get("notes") or ""

    if not isinstance(services, list) or len(services) < 1:
        return "ERROR: `services` must be a non-empty array (at least one service to start)."

    # manual_prereqs: only schema-validated (list of strings). Content is
    # agent's call — if it truly can't auto-install (e.g., GPU driver), it
    # can list it here. No whitelist. No punt-phrase regex.
    if not isinstance(manual, list):
        return "ERROR: manual_prereqs must be a list"
    for i, p in enumerate(manual):
        if not isinstance(p, str):
            return f"ERROR: manual_prereqs[{i}] must be a string"

    # ─── HARD CONSTRAINT C: config_writes coverage ──────────────────────
    if not isinstance(config_writes, list):
        return "ERROR: config_writes must be a list"
    cw_paths = {(c.get("path") or "").strip().lstrip("./").replace("\\", "/")
                for c in config_writes if isinstance(c, dict)}
    examples = _detect_config_examples(repo_dir)
    missing: list[str] = []
    for ex in examples:
        rel = ex.relative_to(repo_dir).as_posix()
        expected_name = _expected_real_path(ex)
        # Compute expected path relative to repo (parent dir of the example)
        parent_rel = ex.relative_to(repo_dir).parent.as_posix()
        expected_rel = (f"{parent_rel}/{expected_name}" if parent_rel and parent_rel != "." else expected_name)
        if expected_rel not in cw_paths:
            missing.append(f"{rel} → expected config_writes[].path = {expected_rel!r}")
    if missing:
        return (
            "ERROR: repo has config example file(s) the agent did not write. "
            "Read the example to learn the schema, then add config_writes "
            "entries that produce the real files. Use ${PARENT_VAR} placeholders "
            "for any credential fields. Missing: " + " | ".join(missing)
        )

    # ─── user_secrets_needed schema check ────────────────────────────────
    user_secrets = plan.get("user_secrets_needed") or []
    if not isinstance(user_secrets, list):
        return "ERROR: user_secrets_needed must be a list"
    user_secret_names: set[str] = set()
    for i, sec in enumerate(user_secrets):
        if not isinstance(sec, dict):
            return f"ERROR: user_secrets_needed[{i}] is not an object"
        for f in ("var_name", "description", "why_needed"):
            if not sec.get(f):
                return f"ERROR: user_secrets_needed[{i}] missing '{f}'"
        vn = sec["var_name"]
        if not re.match(r"^[A-Z_][A-Z0-9_]*$", vn):
            return f"ERROR: user_secrets_needed[{i}].var_name={vn!r} must be UPPER_SNAKE_CASE"
        if vn in PARENT_CREDENTIALS:
            return (f"ERROR: user_secrets_needed[{i}].var_name={vn!r} duplicates a "
                       f"PARENT_CREDENTIALS entry — use ${{{vn}}} directly in "
                       f"config_writes WITHOUT user_secrets_needed.")
        user_secret_names.add(vn)

    # ─── HARD CONSTRAINT D: config_writes placeholders must resolve ─────
    allowed_placeholders = set(PARENT_CREDENTIALS) | user_secret_names
    for i, c in enumerate(config_writes):
        if not isinstance(c, dict):
            return f"ERROR: config_writes[{i}] is not an object"
        for f in ("path", "content_template", "purpose"):
            if not c.get(f):
                return f"ERROR: config_writes[{i}] missing '{f}'"
        try:
            _safe_path(repo_dir, c["path"])
        except ValueError as e:
            return f"ERROR: config_writes[{i}].path: {e}"
        for var in _PLACEHOLDER_RE.findall(c["content_template"]):
            if var not in allowed_placeholders:
                return (
                    f"ERROR: config_writes[{i}] uses ${{{var}}} which is "
                    f"neither in PARENT_CREDENTIALS nor declared in "
                    f"user_secrets_needed. Either use a parent var "
                    f"({sorted(PARENT_CREDENTIALS)}) or add an entry to "
                    f"user_secrets_needed so the user fills it in."
                )

    def check_cmd(label: str, c: dict, idx: int) -> Optional[str]:
        """Validate a single command entry.

        NO command-prefix whitelist (agent picks any tool: winget/choco/brew/
        apt/scoop/curl/node/python/cargo/go/…). Only real safety checks:
        destructive patterns, sudo, fork bomb, cwd inside repo.
        """
        if not isinstance(c, dict):
            return f"{label}[{idx}] is not an object"
        cmd = (c.get("command") or "").strip()
        cwd = (c.get("cwd") or "").strip()
        if not cmd:
            return f"{label}[{idx}] missing 'command'"

        # Real safety: destructive or escalation patterns.
        for pat in _BLOCKED_COMMAND_PATTERNS:
            if pat.search(cmd):
                return (f"{label}[{idx}] command {cmd!r} matches blocked safety "
                          f"pattern {pat.pattern!r}. If you genuinely need this "
                          f"capability, list it as a manual_prereq for the user.")

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

    return (f"OK; plan validated ({len(install)} install / {len(seed)} seed / "
            f"{len(services)} services / {len(config_writes)} config_writes / "
            f"{len(manual)} manual_prereqs).")


def _call_with_warning(*, client, model: str, max_tokens: int,
                        system: str, tools: list, messages: list,
                        log, step: int):
    """Plain client.messages.create() with elapsed-time logging + slow-warning.

    NO hard deadline. If glm-5.1 / ARK takes 5 minutes, we let it take 5
    minutes — some projects legitimately need long generations. We:
      1. Log elapsed time per call
      2. Spawn a parallel watchdog thread that emits WARN log lines every
         60s of waiting, so the dev can see "this is taking long" without
         the agent itself being killed
    The agent is responsible for its own outcome. We just give visibility.
    """
    import threading
    import time as _time
    start = _time.monotonic()
    done_event = threading.Event()
    def _watchdog():
        slow_threshold = 60.0
        while not done_event.wait(timeout=slow_threshold):
            elapsed = _time.monotonic() - start
            log.warning(
                f"step {step} LLM call still running after {elapsed:.0f}s — "
                f"this is unusually long (>60s). SDK state: still waiting for "
                f"response. No action taken — letting agent complete."
            )
    wd = threading.Thread(target=_watchdog, daemon=True,
                            name=f"watchdog-step{step}")
    wd.start()
    try:
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            tools=tools, messages=messages,
        )
        elapsed = _time.monotonic() - start
        log.info(f"step {step} ← stop_reason={resp.stop_reason}  "
                   f"in={resp.usage.input_tokens} out={resp.usage.output_tokens}  "
                   f"elapsed={elapsed:.1f}s")
        return resp
    finally:
        done_event.set()


def _build_initial_message(repo_dir: Path, project_brief: Optional[str] = None) -> str:
    sysname = platform.system()  # 'Windows' / 'Linux' / 'Darwin'
    pkg_hints = {
        "Windows": "winget install <pkg> | choco install <pkg> | scoop install <pkg>",
        "Darwin":  "brew install <pkg>",
        "Linux":   "apt-get install -y <pkg> | dnf install -y <pkg> | pacman -S <pkg>",
    }.get(sysname, "use the OS package manager")
    parts = [
        f"Repo path: {repo_dir}",
        f"Host OS:   {sysname} ({platform.platform()})",
        f"Package manager hint: {pkg_hints}",
        "",
        "## Available credentials (from parent pipeline)",
        "Use these as ${VAR} placeholders inside config_writes content_template.",
        "Values are substituted at execute time — you only see the names.",
        "If the project needs an LLM api_key / base_url / model, USE THESE — do",
        "NOT add them to user_secrets_needed.",
    ]
    for k, desc in PARENT_CREDENTIALS.items():
        parts.append(f"  - ${{{k}}}: {desc}")
    parts.append("")
    parts.append("## Workflow reminder")
    parts.append("1. For every system tool the project needs, call `check_tool(name)`")
    parts.append("   FIRST. If INSTALLED, do nothing. If NOT_FOUND, add an")
    parts.append("   install_commands entry that installs it via the package")
    parts.append("   manager above (NOT manual_prereqs).")
    parts.append("2. For every config.example.* / .env.example, produce a real")
    parts.append("   file via config_writes. Use ${PARENT_VAR} for what we have.")
    parts.append("3. ONLY if the project needs a key/secret with no parent")
    parts.append("   substitute → declare it in user_secrets_needed; executor")
    parts.append("   will ask the user to fill that single value.")
    parts.append("4. manual_prereqs should be empty [] in 95%+ of cases.")
    parts.append("")

    examples = _detect_config_examples(repo_dir)
    if examples:
        parts.append("## Config example files DETECTED (you MUST cover each via config_writes)")
        for ex in examples:
            rel = ex.relative_to(repo_dir).as_posix()
            expected_name = _expected_real_path(ex)
            parent_rel = ex.relative_to(repo_dir).parent.as_posix()
            expected = (f"{parent_rel}/{expected_name}"
                        if parent_rel and parent_rel != "." else expected_name)
            parts.append(f"  - {rel}  →  produce {expected}")
        parts.append("Read each example with read_file to learn its schema, then output the real file.")
        parts.append("")

    if project_brief:
        parts.append("## Project brief (Phase 1, for context)")
        parts.append("```markdown")
        parts.append(project_brief[:3000])
        parts.append("```")
        parts.append("")
    parts.append("Begin: list the repo root, read README + every example config, identify "
                 "the stack, decide install/seed/services + config_writes. Then call `submit_plan`.")
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
    # Per-run prompt override
    from ._prompt_override import get_system_prompt
    run_dir = output_path.parent
    effective_system_prompt = get_system_prompt("setup_runner", SYSTEM_PROMPT, run_dir)
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
        # NO hard deadline — some projects legitimately take long. We log
        # elapsed time + emit WARN lines every 60s for visibility; the agent
        # decides its own fate. If user wants to kill, they kill the process.
        from .error_agent import llm_call_with_recovery
        resp = llm_call_with_recovery(
            lambda: _call_with_warning(
                client=client, model=model, max_tokens=4096,
                system=effective_system_prompt, tools=TOOLS, messages=messages,
                log=log, step=step + 1,
            ),
            run_dir=output_path.parent,
            agent="setup_runner",
            step_label=f"step {step + 1} LLM call",
            context_hint={"model": model, "step": step + 1, "max_steps": max_steps,
                          "input_msgs": len(messages)},
            log=log,
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
                elif tu.name == "check_tool":
                    short_arg = tu.input.get("name", "")
                    out = _tool_check_tool(tu.input)
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
