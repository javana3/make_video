"""ErrorAgent — escalation handler when another agent's retries are exhausted.

Trigger: any other agent's LLM call fails 3 retries in a row (handled by
`tools.llm_retry.call_with_retries`). The caller invokes `analyze_failure()`
with the error chain + context snapshot + a reference to which agent failed.

ErrorAgent reads project source (list_dir / read_file / find_files / grep)
and the relevant logs, then calls `suggest_fix()` exactly once with a
structured recommendation. It does NOT modify anything itself — the user
reviews the suggestion in the UI and decides whether to apply.

Model: LLM_DEEP env (default minimax-m2.7) — chosen for stronger reasoning
on debugging-style problems.
"""
from __future__ import annotations
import json
import platform
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..observability.audit import traced_agent
from ..observability.logger import agent_logger
from ..tools.llm import anthropic_client, model_for


SYSTEM_PROMPT = """You are the ErrorAgent in a promo-video pipeline.

Another agent (project_analyzer / setup_runner / demo_driver / remotion_composer /
voice_over) tried to do its job, but its underlying LLM call failed 4 times in
a row (1s / 5s / 15s exponential backoff already burned). The pipeline can't
proceed automatically — your job is to look at WHAT happened and tell the user
how to recover.

You will receive a failure briefing:
- which agent failed
- which step / what it was trying to do
- the chain of errors (type + message for each retry attempt)
- a context_hint dict with model, step number, etc.
- the absolute path to the run_dir (so you can read events.jsonl, errors.jsonl,
  the agent's own log <run_dir>/logs/<agent>.jsonl, any partial output the
  agent already wrote, etc.)

Approach:
1. Use `read_file` on errors.jsonl + the agent's own log to see the full
   error context (request body, response, traceback if any).
2. If it's an API error (quota, auth, rate limit), match the error message
   against known patterns:
     - "AccountQuotaExceeded" → ARK Coding Plan 5h cap; check reset timestamp
       in the error message; tell user to wait OR provide a fresh key
     - "InvalidParameter ... image" → model doesn't support vision; suggest
       a different model in LLM_VISION env
     - "rate limit" / 429 (non-quota) → suggest slower run, or different model
     - 5xx → upstream issue; suggest retry later
     - timeout — usually long context; suggest model with bigger context OR
       reset history
3. If it's a code error (missing file, validator rejection loop), `list_dir`
   the run_dir and `read_file` the partial outputs / config files.
4. Call `suggest_fix` exactly once with your recommendation. Fields:
     - action: ONE of {wait, change_env, change_prompt, change_code,
                       skip_step, give_up, retry_now}
     - reasoning: 1-3 sentences why
     - user_action: concrete steps the user should take (numbered)
     - confidence: 0-1 float
     - related_files: list of paths the user should look at

Tone: precise, blame-free, actionable. Don't speculate beyond evidence.
You are NOT auto-fixing — only suggesting.
"""


TOOLS = [
    {
        "name": "list_dir",
        "description": "List entries in a directory (absolute path required).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file (up to max_bytes; large files truncated).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_bytes": {"type": "integer", "default": 32768},
            },
            "required": ["path"],
        },
    },
    {
        "name": "find_files",
        "description": "Glob match files under a root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "root": {"type": "string"},
                "pattern": {"type": "string"},
            },
            "required": ["root", "pattern"],
        },
    },
    {
        "name": "grep",
        "description": "Search text in files under a root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "root": {"type": "string"},
                "pattern": {"type": "string"},
                "max_hits": {"type": "integer", "default": 20},
            },
            "required": ["root", "pattern"],
        },
    },
    {
        "name": "suggest_fix",
        "description": (
            "Submit your final recommendation. Call exactly once when you've "
            "gathered enough evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["wait", "change_env", "change_prompt",
                              "change_code", "skip_step", "give_up", "retry_now"],
                },
                "reasoning": {"type": "string"},
                "user_action": {"type": "string",
                                  "description": "Numbered concrete steps the user should take."},
                "confidence": {"type": "number"},
                "related_files": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["action", "reasoning", "user_action", "confidence"],
        },
    },
]


def _tool_list_dir(args: dict) -> str:
    p = Path(args["path"])
    if not p.exists():
        return f"ERROR: {p} does not exist"
    if not p.is_dir():
        return f"ERROR: {p} is not a directory"
    items = []
    for entry in sorted(p.iterdir()):
        items.append(f"{'D' if entry.is_dir() else 'F'} {entry.name}")
    return "\n".join(items[:200])


def _tool_read_file(args: dict) -> str:
    p = Path(args["path"])
    if not p.exists():
        return f"ERROR: {p} does not exist"
    if not p.is_file():
        return f"ERROR: {p} is not a file"
    max_bytes = int(args.get("max_bytes", 32768))
    raw = p.read_bytes()[:max_bytes]
    truncated = p.stat().st_size > max_bytes
    text = raw.decode("utf-8", errors="replace")
    return text + (f"\n\n[... truncated; total {p.stat().st_size}B ...]" if truncated else "")


def _tool_find_files(args: dict) -> str:
    root = Path(args["root"])
    if not root.exists():
        return f"ERROR: {root} does not exist"
    hits = list(root.glob(args["pattern"]))[:200]
    if not hits:
        return "(no matches)"
    return "\n".join(str(h) for h in hits)


def _tool_grep(args: dict) -> str:
    root = Path(args["root"])
    if not root.exists():
        return f"ERROR: {root} does not exist"
    import re as _re
    try:
        pat = _re.compile(args["pattern"])
    except Exception as e:
        return f"ERROR: invalid regex: {e}"
    max_hits = int(args.get("max_hits", 20))
    hits: list[str] = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pat.search(line):
                    hits.append(f"{f}:{i}:{line[:200]}")
                    if len(hits) >= max_hits:
                        return "\n".join(hits)
        except Exception:
            continue
    return "\n".join(hits) if hits else "(no matches)"


def _build_briefing(*,
                     failed_agent: str,
                     step_label: str,
                     error_chain: list[dict],
                     context_hint: dict,
                     run_dir: Path) -> str:
    """Compose the user message that kicks off ErrorAgent."""
    parts = [
        f"# Failure briefing",
        f"",
        f"**Failed agent**: `{failed_agent}`",
        f"**Step**: {step_label}",
        f"**Run dir**: `{run_dir}`",
        f"**Host OS**: {platform.system()} ({platform.platform()})",
        f"",
        f"## Error chain (4 retries, oldest first)",
    ]
    for i, e in enumerate(error_chain, 1):
        parts.append(f"### Attempt {i}")
        parts.append(f"- type: `{e.get('error_type','?')}`")
        parts.append(f"- text: {e.get('error_text','')[:1000]}")
    parts.append("")
    parts.append("## Context hint")
    parts.append("```json")
    parts.append(json.dumps(context_hint, ensure_ascii=False, indent=2)[:2000])
    parts.append("```")
    parts.append("")
    parts.append("## Files you can read")
    parts.append(f"- `{run_dir}/errors.jsonl` — full error log incl. this one")
    parts.append(f"- `{run_dir}/events.jsonl` — pipeline events")
    parts.append(f"- `{run_dir}/logs/<agent>.jsonl` — per-agent loguru log")
    parts.append(f"- `{run_dir}/state.json` — pipeline state")
    parts.append(f"- repo source under `{run_dir}/repo/`")
    parts.append("")
    parts.append(
        "Now: use list_dir / read_file / find_files / grep to investigate, "
        "then call suggest_fix exactly once."
    )
    return "\n".join(parts)


@traced_agent("Error Agent · analyze", phase=0)
def analyze_failure(*,
                    failed_agent: str,
                    step_label: str,
                    error_chain: list[dict],
                    context_hint: dict,
                    run_dir: Path,
                    max_steps: int = 12) -> dict:
    """Run ErrorAgent on a failure. Returns the suggested fix dict.

    Side effects:
      - writes <run_dir>/error_suggestions.jsonl with the suggestion (so the
        web UI can list it and let the user review)
    """
    log = agent_logger("error_agent")
    client = anthropic_client()
    model = model_for("deep")  # minimax-m2.7 by default
    log.info(f"start  failed_agent={failed_agent} step={step_label}  model={model}")

    briefing = _build_briefing(
        failed_agent=failed_agent,
        step_label=step_label,
        error_chain=error_chain,
        context_hint=context_hint,
        run_dir=run_dir,
    )
    messages: list[dict] = [{"role": "user", "content": briefing}]
    suggestion: Optional[dict] = None

    for step in range(max_steps):
        log.info(f"step {step+1}/{max_steps} → LLM")
        try:
            resp = client.messages.create(
                model=model, max_tokens=4096, system=SYSTEM_PROMPT,
                tools=TOOLS, messages=messages,
            )
        except Exception as e:
            log.exception(f"ErrorAgent's own LLM call failed: {e}")
            # ErrorAgent itself failed — do NOT recurse, fall back to a static suggestion
            suggestion = {
                "action": "give_up",
                "reasoning": f"ErrorAgent itself could not run: {type(e).__name__}: {e}",
                "user_action": "Inspect run_dir/errors.jsonl manually. Possibly retry the failed agent later when LLM availability returns.",
                "confidence": 0.1,
                "related_files": [str(run_dir / "errors.jsonl")],
                "fallback": True,
            }
            break

        log.info(f"step {step+1} ← stop={resp.stop_reason} in={resp.usage.input_tokens} out={resp.usage.output_tokens}")
        messages.append({"role": "assistant", "content": resp.content})

        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            log.info("no tool_use; loop ends")
            break

        tool_results = []
        for tu in tool_uses:
            try:
                if tu.name == "list_dir":
                    out = _tool_list_dir(tu.input)
                elif tu.name == "read_file":
                    out = _tool_read_file(tu.input)
                elif tu.name == "find_files":
                    out = _tool_find_files(tu.input)
                elif tu.name == "grep":
                    out = _tool_grep(tu.input)
                elif tu.name == "suggest_fix":
                    suggestion = dict(tu.input)
                    out = "OK; suggestion recorded."
                else:
                    out = f"ERROR: unknown tool {tu.name}"
            except Exception as e:
                out = f"ERROR: {type(e).__name__}: {e}"
            if len(out) > 30000:
                out = out[:30000] + f"\n[... truncated; total {len(out)}B ...]"
            tool_results.append({
                "type": "tool_result", "tool_use_id": tu.id, "content": out,
            })

        messages.append({"role": "user", "content": tool_results})

        if suggestion is not None:
            log.info(f"suggest_fix called → action={suggestion.get('action')} confidence={suggestion.get('confidence')}")
            break

    if suggestion is None:
        suggestion = {
            "action": "give_up",
            "reasoning": "ErrorAgent ran out of steps without calling suggest_fix.",
            "user_action": "Inspect <run_dir>/errors.jsonl + <run_dir>/logs/ manually.",
            "confidence": 0.0,
            "related_files": [str(run_dir / "errors.jsonl")],
            "exhausted": True,
        }

    # Persist the suggestion for the UI to surface.
    suggestion_record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "failed_agent": failed_agent,
        "step_label": step_label,
        "error_chain_summary": [
            {"type": e.get("error_type"), "text": e.get("error_text", "")[:200]}
            for e in error_chain
        ],
        "context_hint": context_hint,
        "suggestion": suggestion,
        "status": "pending_review",
    }
    suggestions_path = run_dir / "error_suggestions.jsonl"
    with suggestions_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(suggestion_record, ensure_ascii=False) + "\n")
    log.info(f"suggestion written → {suggestions_path}")
    return suggestion_record


def llm_call_with_recovery(fn, *,
                              run_dir: Path,
                              agent: str,
                              step_label: str,
                              context_hint: Optional[dict] = None,
                              log=None):
    """Single entry point each agent uses around its LLM call.

    Flow:
      1. Retry the call 3 times (1s/5s/15s backoff) via tools.llm_retry
      2. If all attempts fail → invoke this ErrorAgent to write a suggestion
         to <run_dir>/error_suggestions.jsonl
      3. Re-raise so the caller can halt cleanly (UI surfaces the suggestion)
    """
    from ..tools.llm_retry import call_with_retries, RetryExhausted, MAX_ATTEMPTS
    from ..observability.error_log import read_errors
    try:
        return call_with_retries(
            fn, run_dir=run_dir, agent=agent, step_label=step_label,
            context_hint=context_hint or {}, log=log,
        )
    except RetryExhausted:
        recent_errs = [e for e in read_errors(run_dir)
                        if e.get("agent") == agent
                        and e.get("step_label") == step_label][:MAX_ATTEMPTS]
        try:
            analyze_failure(
                failed_agent=agent,
                step_label=step_label,
                error_chain=list(reversed(recent_errs)),
                context_hint=context_hint or {},
                run_dir=run_dir,
            )
        except Exception as e:
            if log is not None:
                log.exception(f"ErrorAgent invocation itself failed: {e}")
        raise


def read_pending_suggestions(run_dir: Path) -> list[dict]:
    """Return all error suggestions with status=pending_review (newest first)."""
    path = run_dir / "error_suggestions.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if rec.get("status") == "pending_review":
                out.append(rec)
        except Exception:
            pass
    return out
