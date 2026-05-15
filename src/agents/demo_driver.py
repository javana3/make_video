"""Agent · DemoDriver — autonomous project demonstrator.

The Demo Driver Agent is what makes the recording meaningful. It:

  1. Reads the project's SOURCE CODE (list_dir / read_file / find_files /
     grep) to figure out, on its own, what's worth demonstrating. Brief
     is passed in only as tone/audience context — never as a checklist.

  2. Operates the running project — web (`browser_*` tools backed by
     `BrowserSession`) or CLI (`pty_*` tools backed by `PtySession`),
     chosen by `mode`.

  3. Stays alive — *no time cap, no step cap modulo a safety stop*.
     Each LLM round, the agent observes (screenshot / visible text /
     terminal screen), decides, acts, observes again. It calls
     `finish_demo` when it (and optionally the user) judges the
     demonstration is complete.

  4. Listens to the user mid-flight: the user can append to
     `live_feedback.jsonl` from the web UI; new messages are spliced
     into the conversation as USER LIVE FEEDBACK before each LLM
     round. `ask_user` blocks until the next entry arrives.

The session (web or CLI) is started by THIS agent before the LLM loop
begins, and stopped by THIS agent on `finish_demo` (or on exception).
The mp4 it produces IS the recording — there is no separate recorder.
"""
from __future__ import annotations

import json
import platform
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger
from ..tools.browser_session import BrowserSession, BrowserSessionResult
from ..tools.llm import anthropic_client, model_for
from ..tools.pty_session import PtySession, PtySessionResult


# ─── system prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the Demo Driver Agent.

A project has just been launched and is running. Your job is to *operate \
it like a real user would*, end-to-end, so that the screen recording \
captures a complete, meaningful demonstration of what this project \
actually does.

═══════════════════════════════════════════════════════════════════════
FIGURE OUT WHAT TO DEMO BY READING THE SOURCE CODE
═══════════════════════════════════════════════════════════════════════
You have file-reading tools (`list_dir`, `read_file`, `find_files`, `grep`).
USE THEM. The truth of "what does this project do, what's worth showing" \
lives in the code: route handlers, CLI commands, main loops, feature \
modules.

You will be given a `project_brief.md` for *tone and audience* context \
only. DO NOT treat its 独特卖点 / selling-points list as a checklist of \
things to demonstrate. Marketing copy describes positioning; the source \
code tells you what actually exists. If the brief mentions a feature \
that's not in the code, you don't demo a fiction — you demo what's \
real.

═══════════════════════════════════════════════════════════════════════
HOW TO OPERATE — WEB MODE (browser_* tools)
═══════════════════════════════════════════════════════════════════════
  browser_goto(url)
  browser_visible_text() / browser_a11y_snapshot() / browser_interactables()
  browser_screenshot()                  — returns PNG (you'll see it)
  browser_click(target, by="auto")      — by: selector | text | role | auto
  browser_fill(selector, text)
  browser_press(key, selector?)         — Enter / Tab / Escape …
  browser_scroll(dy, dx?)
  browser_hover(target)
  browser_wait_for(selector?, text?, timeout_ms?)
  browser_url() / browser_title()

═══════════════════════════════════════════════════════════════════════
HOW TO OPERATE — CLI MODE (pty_* tools)
═══════════════════════════════════════════════════════════════════════
  pty_screen()                          — current 100×30 terminal grid as text
  pty_read_recent(n_lines)              — tail of the screen
  pty_send(text)                        — write to stdin (include \\n if needed)
  pty_wait_for(pattern, timeout_s)      — regex on the screen
  pty_is_alive()
  pty_transcript(tail_lines=200)        — FULL stdout/stderr history, including
                                          tracebacks/errors that scrolled off
                                          the 100x30 screen. Primary debug tool.
  pty_restart(extra_env, extra_args,    — kill+respawn the CLI with new env/args.
              replace_args)               Use AFTER diagnosing a crash, not
                                          blindly. Recording continues.

═══════════════════════════════════════════════════════════════════════
DEBUGGING WHEN SOMETHING GOES WRONG — DO NOT FINISH ON A BROKEN STATE
═══════════════════════════════════════════════════════════════════════
The project may crash, throw a traceback, hang, output garbled text,
exit unexpectedly, return errors, refuse input, etc. When ANY of those
happens, your job is NOT to just record it and call finish_demo. Your
job is to investigate and fix where you can. The pipeline downstream
turns your recording into a promo video — a video that ends with a
Python traceback is worthless to the user.

Iron law for failure recovery:

1. **Diagnose first.** Call `pty_transcript(tail_lines=400)` and read the
   actual error output. The 100x30 visible screen is too small to hold
   a real Python traceback or a long error.
2. **Find the cause in source.** If transcript shows `File ".../foo.py", line N`
   or names a specific module/function, `read_file(...)` it. Find the
   actual code that produced the failure.
3. **Check what's modifiable from outside.** Many failures are fixable
   without editing source:
     - Windows GBK encoding crash on Chinese print → `pty_restart(extra_env={"PYTHONIOENCODING":"utf-8"})`
     - Color-emitting library garbled in pyte → `extra_env={"NO_COLOR":"1"}` or `--no-color` arg
     - Different entry point exists (smoke.py, demo.py) → `pty_restart(replace_args=[...])`
     - Missing config file the project would have created → write a minimal config
       (NOT in this run — note in finish_demo summary so user can fix the project).
4. **Try the fix.** Call `pty_restart` with the targeted change. Watch the
   transcript again. If recovered, continue the demo.
5. **Limit attempts.** Try at most 2-3 distinct fixes. If none work, call
   `finish_demo(completeness='blocked', summary=...)` with a concrete root
   cause and the fixes you tried. Do NOT finish 'partial' silently after
   a crash — call it 'blocked' so the user knows there's an unresolved
   issue in the project itself.

Backends, services, GUIs:
- For web/backend projects, after a browser action that fails: call
  `tail_service_log(name, kind='stderr', lines=200)` for the matching
  service to see the server-side traceback. Read the relevant route
  handler source. Same iron law applies — diagnose, try a fix, retry.
- For desktop GUI projects (no headless mode, no browser, no CLI):
  this pipeline doesn't yet support full GUI driving. Record what's
  available (a terminal launch / help output) and call finish_demo
  with completeness='partial' and an explicit note about the missing
  capability.

═══════════════════════════════════════════════════════════════════════
PACING — DON'T RUSH, DON'T STOP EARLY
═══════════════════════════════════════════════════════════════════════
- After every action, OBSERVE before acting again. Take a screenshot or \
  read screen text and confirm the program responded as expected. If a \
  page hasn't loaded, wait_for. If a CLI tool is processing, pty_wait_for \
  on the next prompt.
- Cover the project's CORE flows. A login screen alone is not a demo. \
  Get into the application, exercise the actual features, navigate \
  multiple views.
- If a project genuinely takes 1000 turns to demonstrate properly, take \
  1000 turns. There is NO time limit and NO step quota. Cutting the \
  demo short produces a worthless promo video; that is the worst outcome.

═══════════════════════════════════════════════════════════════════════
CAPTIONS — TAG THE MOMENTS WORTH NARRATING
═══════════════════════════════════════════════════════════════════════
`mark_caption(zh, en, importance)` — call it at the moment a meaningful \
beat happens on screen (a feature is revealed, a result appears, a \
transition completes). The caption is timestamped at the call moment \
relative to recording start. Importance: 1 (minor beat) … 5 (anchor \
moment).

DO NOT fabricate captions for things that did not happen on screen. The \
screen is the source of truth.

═══════════════════════════════════════════════════════════════════════
USER LIVE FEEDBACK
═══════════════════════════════════════════════════════════════════════
The user can talk to you mid-demo. Their messages appear at the start \
of subsequent turns prefixed `[USER LIVE FEEDBACK]`. Listen. They might \
say "skip the auth flow", "focus on feature X", "you missed Y", "that's \
enough, wrap up". Adjust accordingly. Don't ignore them.

You can also call `ask_user(question)` to pause and wait for input \
when you genuinely need a decision (e.g. credentials they haven't \
shared, ambiguous next step).

═══════════════════════════════════════════════════════════════════════
WRAPPING UP
═══════════════════════════════════════════════════════════════════════
When the demonstration genuinely covers the project end-to-end (or the \
user explicitly tells you to stop), call \
`finish_demo(summary, completeness)`. This stops recording and ends the \
loop.

Antipatterns to avoid:
  ✗ Calling finish_demo after one screen
  ✗ Treating brief.独特卖点 as a checklist
  ✗ Fabricating captions
  ✗ Ignoring live user feedback
  ✗ Burning turns on read_file when you should be operating the program
  ✗ Burning turns operating the program before you've understood the code
  ✗ Calling finish_demo immediately after seeing a crash / traceback /
    unexpected exit. ALWAYS pty_transcript + read_file the implicated
    source FIRST. Try a pty_restart with a targeted fix. Only then,
    if truly stuck, finish_demo(completeness='blocked').
  ✗ Recording a mark_caption that says "(crashed on ...)" or "(error
    because ...)" and then immediately finish_demo. That is exactly
    the failure mode. The caption captures the moment; your NEXT move
    must be diagnosis + recovery, not retreat.
"""


# ─── tool schemas (passed to Anthropic) ────────────────────────────────
_SOURCE_TOOLS = [
    {
        "name": "list_dir",
        "description": "List entries in a directory inside the project repo. Path is repo-relative.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
        },
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file by repo-relative path. Up to 64KB by default.",
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
        "description": "Glob for files in the repo (e.g. '**/*.py', 'src/routes/**').",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": "Search repo files for a regex; returns matching file:line:text rows. Use to locate features/handlers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "max_matches": {"type": "integer", "default": 50},
            },
            "required": ["pattern"],
        },
    },
]

_BROWSER_TOOLS = [
    {"name": "browser_goto", "description": "Navigate to URL. Pre-launched session.",
     "input_schema": {"type": "object",
                       "properties": {"url": {"type": "string"},
                                       "timeout_ms": {"type": "integer", "default": 15000}},
                       "required": ["url"]}},
    {"name": "browser_visible_text", "description": "Return visible body text (truncated to 8000 chars).",
     "input_schema": {"type": "object", "properties": {"max_chars": {"type": "integer", "default": 8000}}}},
    {"name": "browser_a11y_snapshot", "description": "Compact accessibility tree as a text outline.",
     "input_schema": {"type": "object", "properties": {"max_chars": {"type": "integer", "default": 6000}}}},
    {"name": "browser_interactables", "description": "Catalog of clickable / typeable elements with stable selectors.",
     "input_schema": {"type": "object", "properties": {"max_items": {"type": "integer", "default": 60}}}},
    {"name": "browser_screenshot",
     "description": "PNG screenshot of the current viewport (you will see it as an image).",
     "input_schema": {"type": "object", "properties": {"full_page": {"type": "boolean", "default": False}}}},
    {"name": "browser_click",
     "description": "Click an element. by='auto' tries selector then text=…",
     "input_schema": {"type": "object",
                       "properties": {"target": {"type": "string"},
                                       "by": {"type": "string", "enum": ["selector", "text", "role", "auto"], "default": "auto"}},
                       "required": ["target"]}},
    {"name": "browser_fill", "description": "Type text into a form field. Selector is CSS.",
     "input_schema": {"type": "object",
                       "properties": {"selector": {"type": "string"}, "text": {"type": "string"}},
                       "required": ["selector", "text"]}},
    {"name": "browser_press", "description": "Send a keyboard key (Enter / Tab / Escape / ArrowDown …). Optional selector to focus first.",
     "input_schema": {"type": "object",
                       "properties": {"key": {"type": "string"}, "selector": {"type": "string"}},
                       "required": ["key"]}},
    {"name": "browser_scroll", "description": "Scroll page wheel. Positive dy = down.",
     "input_schema": {"type": "object",
                       "properties": {"dy": {"type": "integer", "default": 600},
                                       "dx": {"type": "integer", "default": 0}}}},
    {"name": "browser_hover", "description": "Hover an element (selector or text=…).",
     "input_schema": {"type": "object", "properties": {"target": {"type": "string"}},
                       "required": ["target"]}},
    {"name": "browser_wait_for", "description": "Wait for selector visible OR text visible OR plain delay (omit selector+text → just wait timeout_ms).",
     "input_schema": {"type": "object",
                       "properties": {"selector": {"type": "string"},
                                       "text": {"type": "string"},
                                       "timeout_ms": {"type": "integer", "default": 30000}}}},
    {"name": "browser_url", "description": "Current page URL.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "browser_title", "description": "Current page title.",
     "input_schema": {"type": "object", "properties": {}}},
]

_PTY_TOOLS = [
    {"name": "pty_screen", "description": "Snapshot of the current 100x30 terminal screen as plain text (rows joined with newlines).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "pty_read_recent", "description": "Tail of the terminal screen (last n_lines lines).",
     "input_schema": {"type": "object", "properties": {"n_lines": {"type": "integer", "default": 30}}}},
    {"name": "pty_send", "description": "Write to stdin. Include trailing \\n if you want to submit a line.",
     "input_schema": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"]}},
    {"name": "pty_wait_for", "description": "Poll the terminal screen for a regex; return match boolean.",
     "input_schema": {"type": "object",
                       "properties": {"pattern": {"type": "string"},
                                       "timeout_s": {"type": "number", "default": 30.0}},
                       "required": ["pattern"]}},
    {"name": "pty_transcript",
     "description": (
         "Return the FULL stdout/stderr history of the CLI process(es) "
         "since session start, including bytes the 100x30 pyte screen has "
         "already scrolled past. THIS IS THE PRIMARY DEBUG TOOL when the "
         "program crashed, hit a traceback, hung, or behaved unexpectedly — "
         "the visible screen is too small to hold a Python traceback or a "
         "long error message. Restart markers separate sessions if "
         "pty_restart was called."
     ),
     "input_schema": {"type": "object",
                       "properties": {
                           "tail_lines": {"type": "integer",
                                           "description": "Last N lines of transcript. Defaults to 200 if neither is given."},
                           "head_lines": {"type": "integer",
                                           "description": "First N lines instead (for very early start-up errors)."},
                       }}},
    {"name": "pty_restart",
     "description": (
         "Kill the current CLI process and respawn it. The video recording "
         "continues — the crash + restart + recovered demo become one "
         "continuous video. Use AFTER you have diagnosed a failure (via "
         "pty_transcript + read_file) AND have a concrete hypothesis for a "
         "fix. Examples:\n"
         "  - Windows GBK encoding crash → extra_env={\"PYTHONIOENCODING\":\"utf-8\"}\n"
         "  - Wrong CLI flag → replace_args=[\"python\",\"-u\",\"main.py\",\"--no-color\"]\n"
         "  - Different subcommand entirely → replace_args=[...]\n"
         "DO NOT restart blindly more than 2-3 times — if you can't fix it "
         "via env/args alone, call finish_demo with completeness='blocked' "
         "and a clear summary of the unfixable issue."
     ),
     "input_schema": {"type": "object",
                       "properties": {
                           "extra_env": {"type": "object",
                                          "description": "Env vars to merge into existing env (e.g. {\"PYTHONIOENCODING\":\"utf-8\"})."},
                           "extra_args": {"type": "array",
                                           "items": {"type": "string"},
                                           "description": "Args to append to the current command argv."},
                           "replace_args": {"type": "array",
                                             "items": {"type": "string"},
                                             "description": "Entirely replace the command argv (instead of appending)."},
                       }}},
    {"name": "pty_is_alive", "description": "Whether the underlying process is still running.",
     "input_schema": {"type": "object", "properties": {}}},
]

_OBSERVABILITY_TOOLS = [
    {"name": "list_services",
     "description": "List backend services running in this run (name / port / status / health_url / log paths). Use this to discover what services exist before tailing their logs.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "tail_service_log",
     "description": (
         "Read the last N lines of a service's stdout/stderr/both log. The "
         "PRIMARY way to see backend errors — Python tracebacks, FastAPI 500 "
         "reasons, ComfyUI workflow failures, etc. live in stderr. Use when "
         "the UI is stuck, returns an error, or after any browser action that "
         "should have triggered backend work."
     ),
     "input_schema": {
         "type": "object",
         "properties": {
             "name": {"type": "string", "description": "Service name from list_services."},
             "kind": {"type": "string", "enum": ["stderr", "stdout", "both"],
                       "default": "stderr"},
             "lines": {"type": "integer", "default": 100,
                        "description": "Number of trailing lines to return."},
         },
         "required": ["name"],
     }},
    {"name": "browser_console_log",
     "description": (
         "Get JS console messages captured by the browser (errors, warnings, "
         "logs from the page's own scripts). Useful when a button click does "
         "nothing — there's often a JS error in console. Web mode only."
     ),
     "input_schema": {
         "type": "object",
         "properties": {
             "level": {"type": "string",
                        "enum": ["error", "warning", "info", "log", "debug", "all"],
                        "default": "error"},
             "limit": {"type": "integer", "default": 50},
             "since_action": {"type": "integer", "default": 0,
                                "description": "Only entries since N-th browser action (0 = all). Useful: read browser_url first to learn current n_actions value."},
         },
     }},
    {"name": "browser_network_failures",
     "description": (
         "Get 4xx/5xx HTTP responses observed by the browser (failed XHR / "
         "fetch / page resource loads). Use when an API call from the page "
         "fails silently. Web mode only."
     ),
     "input_schema": {
         "type": "object",
         "properties": {
             "min_status": {"type": "integer", "default": 400},
             "limit": {"type": "integer", "default": 50},
             "since_action": {"type": "integer", "default": 0},
         },
     }},
]


_CONTROL_TOOLS = [
    {"name": "mark_caption",
     "description": "Tag the CURRENT recording timestamp with a bilingual caption. Call when a meaningful beat happens on screen.",
     "input_schema": {"type": "object",
                       "properties": {"zh": {"type": "string"},
                                       "en": {"type": "string"},
                                       "importance": {"type": "integer", "default": 3,
                                                       "description": "1 (minor) … 5 (anchor)"}},
                       "required": ["zh", "en"]}},
    {"name": "ask_user",
     "description": "Ask the user a question and BLOCK waiting for a reply. Use sparingly — only when you need a real decision (credentials, ambiguous direction).",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}},
                       "required": ["question"]}},
    {"name": "log_thought",
     "description": "Write a thought to the agent log (does not affect captions or video). Use to document your plan.",
     "input_schema": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"]}},
    {"name": "finish_demo",
     "description": "Conclude the demo. Stops recording, ends the agent loop. Call only when you have meaningfully demonstrated the project end-to-end.",
     "input_schema": {"type": "object",
                       "properties": {"summary": {"type": "string"},
                                       "completeness": {"type": "string",
                                                         "enum": ["full", "partial", "blocked"]}},
                       "required": ["summary", "completeness"]}},
]


# ─── helpers: source-reading (shared style with SetupRunner) ───────────
def _safe_path(repo_dir: Path, rel: str) -> Path:
    rel = (rel or ".").lstrip("/").lstrip("\\") or "."
    p = (repo_dir / rel).resolve()
    if p != repo_dir.resolve() and repo_dir.resolve() not in p.parents:
        raise ValueError(f"path escapes repo: {rel!r}")
    return p


def _tool_list_dir(repo_dir: Path, args: dict) -> str:
    p = _safe_path(repo_dir, args.get("path", "."))
    if not p.exists() or not p.is_dir():
        return f"ERROR: {args.get('path', '.')!r} not a directory"
    out = []
    for e in sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))[:200]:
        if e.is_dir():
            out.append(f"DIR   {e.name}/")
        else:
            out.append(f"FILE  {e.name}  ({e.stat().st_size}B)")
    return "\n".join(out) if out else "(empty)"


def _tool_read_file(repo_dir: Path, args: dict) -> str:
    p = _safe_path(repo_dir, args["path"])
    if not p.exists() or not p.is_file():
        return f"ERROR: {args['path']!r} not a file"
    n = int(args.get("max_bytes", 65536))
    raw = p.read_bytes()[:n]
    text = raw.decode("utf-8", errors="replace")
    if p.stat().st_size > n:
        text += f"\n\n[... truncated; total {p.stat().st_size}B ...]"
    return text


def _tool_find_files(repo_dir: Path, args: dict) -> str:
    matches = sorted(repo_dir.glob(args["pattern"]))[:200]
    if not matches:
        return "(no matches)"
    return "\n".join(str(m.relative_to(repo_dir)).replace("\\", "/") for m in matches)


def _tool_grep(repo_dir: Path, args: dict) -> str:
    pattern = args["pattern"]
    path_arg = args.get("path", ".")
    max_matches = int(args.get("max_matches", 50))
    base = _safe_path(repo_dir, path_arg)
    if not base.exists():
        return f"ERROR: {path_arg!r} not found"
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"ERROR: bad regex: {e}"
    out: list[str] = []
    files = [base] if base.is_file() else list(base.rglob("*"))
    for f in files:
        if not f.is_file():
            continue
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").split("\n"), 1):
                if compiled.search(line):
                    rel = str(f.relative_to(repo_dir)).replace("\\", "/")
                    out.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(out) >= max_matches:
                        break
        except Exception:
            continue
        if len(out) >= max_matches:
            break
    return "\n".join(out) if out else "(no matches)"


# ─── live feedback channel ─────────────────────────────────────────────
class FeedbackChannel:
    """Append-only JSONL the web UI writes to; driver reads new entries."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")
        self._cursor = self.path.stat().st_size

    def read_new(self) -> list[dict]:
        sz = self.path.stat().st_size
        if sz <= self._cursor:
            return []
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(self._cursor)
            data = f.read()
            self._cursor = sz
        out: list[dict] = []
        for line in data.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"text": line, "ts": datetime.now(timezone.utc).isoformat()})
        return out

    def wait_new(self, poll_s: float = 0.5,
                 stop_predicate=lambda: False) -> Optional[dict]:
        """Block until a new entry appears or stop_predicate() returns True."""
        while True:
            new = self.read_new()
            if new:
                return new[0]
            if stop_predicate():
                return None
            time.sleep(poll_s)


# ─── result data ───────────────────────────────────────────────────────
@dataclass
class DemoDriverResult:
    mode: str                       # "web" | "cli"
    recording_path: str
    duration_s: float
    captions_path: str
    summary_path: str
    n_captions: int
    n_steps: int
    completeness: str
    finish_summary: str


# ─── progress writer (web UI consumer) ─────────────────────────────────
class _Progress:
    def __init__(self, path: Optional[Path], started_iso: str, started_mono: float):
        self.path = path
        self.started_iso = started_iso
        self.started_mono = started_mono
        self.step = 0
        self.last_action = "starting"
        self.last_caption: Optional[dict] = None
        self.captions_count = 0
        self.tool_calls = 0
        self.status = "running"
        self.error: Optional[str] = None
        self.pending_question: Optional[str] = None

    def write(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "phase": "2c-demo-driver",
            "status": self.status,
            "step": self.step,
            "last_action": self.last_action,
            "captions_count": self.captions_count,
            "last_caption": self.last_caption,
            "tool_calls": self.tool_calls,
            "started_at": self.started_iso,
            "elapsed_s": round(time.monotonic() - self.started_mono, 1),
            "pending_question": self.pending_question,
            "last_update": datetime.now(timezone.utc).isoformat(),
            "error": self.error,
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# ─── main entry ────────────────────────────────────────────────────────
@traced_agent("Agent · Demo Driver", phase=2)
def run_demo_driver(*,
                    run_dir: Path,
                    repo_dir: Path,
                    project_brief: str,
                    mode: str,                       # "web" | "cli"
                    web_url: Optional[str] = None,
                    cli_command: Optional[list[str]] = None,
                    cli_cwd: Optional[Path] = None,
                    feedback_path: Optional[Path] = None,
                    progress_path: Optional[Path] = None,
                    output_recording: Optional[Path] = None,
                    captions_path: Optional[Path] = None,
                    summary_path: Optional[Path] = None,
                    safety_step_cap: int = 1000) -> DemoDriverResult:
    """Run the demo driver loop.

    Args:
      mode:           "web" (uses BrowserSession + web_url) or "cli"
                       (uses PtySession + cli_command/cli_cwd)
      web_url:        starting URL for web mode (the running service)
      cli_command:    argv list for CLI mode (e.g. ["python", "main.py"])
      feedback_path:  user → driver channel (default run_dir/live_feedback.jsonl)
      progress_path:  driver → web UI status (default run_dir/demo_driver_progress.json)
      safety_step_cap: a hard upper bound on LLM rounds, NOT a duration cap.
                       Set high (default 1000); the agent itself decides when
                       to call finish_demo.
    """
    assert mode in ("web", "cli"), f"mode must be web|cli, got {mode!r}"
    log = agent_logger("demo_driver")
    started_at = datetime.now(timezone.utc)
    started_mono = time.monotonic()

    # ─── paths
    feedback_path = feedback_path or (run_dir / "live_feedback.jsonl")
    progress_path = progress_path or (run_dir / "demo_driver_progress.json")
    output_recording = output_recording or (run_dir / "recordings" / "demo.mp4")
    captions_path = captions_path or (run_dir / "demo_captions.jsonl")
    summary_path = summary_path or (run_dir / "demo_summary.md")

    output_recording.parent.mkdir(parents=True, exist_ok=True)
    captions_path.parent.mkdir(parents=True, exist_ok=True)
    captions_path.write_text("", encoding="utf-8")  # truncate any old captions

    feedback = FeedbackChannel(feedback_path)
    progress = _Progress(progress_path, started_at.isoformat(), started_mono)
    progress.write()

    # ─── start the right session
    session: Any  # BrowserSession | PtySession
    if mode == "web":
        if not web_url:
            raise ValueError("web mode requires web_url")
        log.info(f"WEB mode  url={web_url}")
        session = BrowserSession.start(
            record_dir=run_dir / "_browser_record",
            viewport_w=1920, viewport_h=1080,
            headless=True,
        )
        session_t0 = time.monotonic()
        try:
            session.goto(web_url, timeout_ms=20000)
        except Exception as e:
            log.warning(f"initial goto failed: {e}; agent may retry")
    else:
        if not cli_command:
            raise ValueError("cli mode requires cli_command")
        cwd = cli_cwd or repo_dir
        log.info(f"CLI mode  cmd={cli_command}  cwd={cwd}")
        session = PtySession.start(
            command=cli_command, cwd=cwd,
            frames_dir=run_dir / "_cli_frames",
            fps=15, cols=100, rows=30,
            width=1920, height=1080,
        )
        session_t0 = time.monotonic()

    # ─── tools
    # Service log readers (list_services / tail_service_log) are always
    # available. Browser console / network tools are web-only.
    tools = list(_SOURCE_TOOLS) + list(_CONTROL_TOOLS)
    tools += [t for t in _OBSERVABILITY_TOOLS
              if t["name"] in ("list_services", "tail_service_log")]
    if mode == "web":
        tools += list(_BROWSER_TOOLS)
        tools += [t for t in _OBSERVABILITY_TOOLS
                  if t["name"] in ("browser_console_log", "browser_network_failures")]
    else:
        tools += list(_PTY_TOOLS)

    # ─── conversation
    initial_text = _build_initial_message(repo_dir, project_brief, mode,
                                            web_url=web_url, cli_command=cli_command)
    messages: list[dict] = [{"role": "user", "content": initial_text}]
    captions: list[dict] = []
    finish_payload: Optional[dict] = None

    # ─── tool dispatch
    def dispatch(name: str, args: dict) -> Any:
        # source-reading
        if name == "list_dir":      return _tool_list_dir(repo_dir, args)
        if name == "read_file":     return _tool_read_file(repo_dir, args)
        if name == "find_files":    return _tool_find_files(repo_dir, args)
        if name == "grep":          return _tool_grep(repo_dir, args)
        # control
        if name == "mark_caption":
            ts = round(time.monotonic() - session_t0, 3)
            cap = {"t": ts, "zh": args["zh"], "en": args["en"],
                   "importance": int(args.get("importance", 3)),
                   "ts_iso": datetime.now(timezone.utc).isoformat()}
            captions.append(cap)
            with captions_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(cap, ensure_ascii=False) + "\n")
            progress.captions_count = len(captions)
            progress.last_caption = cap
            return f"caption recorded at t={ts}s ({len(captions)} total)"
        if name == "ask_user":
            q = args["question"]
            log.info(f"ask_user: {q[:120]}")
            progress.pending_question = q
            progress.write()
            # Block-poll feedback channel
            entry = feedback.wait_new(poll_s=0.5,
                                       stop_predicate=lambda: not session.is_alive())
            progress.pending_question = None
            if entry is None:
                return "ERROR: session ended before user replied"
            return f"USER REPLIED: {entry.get('text', json.dumps(entry, ensure_ascii=False))}"
        if name == "log_thought":
            log.info(f"thought: {args['text'][:300]}")
            return "logged"
        if name == "finish_demo":
            nonlocal finish_payload
            finish_payload = {
                "summary": args["summary"],
                "completeness": args.get("completeness", "full"),
            }
            return "finish_demo accepted; loop will exit after this turn"
        # web ops
        if name == "browser_goto":
            return session.goto(args["url"], timeout_ms=int(args.get("timeout_ms", 15000)))
        if name == "browser_visible_text":
            return session.visible_text(max_chars=int(args.get("max_chars", 8000)))
        if name == "browser_a11y_snapshot":
            return session.a11y_snapshot(max_chars=int(args.get("max_chars", 6000)))
        if name == "browser_interactables":
            return session.interactables(max_items=int(args.get("max_items", 60)))
        if name == "browser_screenshot":
            png = session.screenshot(full_page=bool(args.get("full_page", False)))
            return {"_image_b64": __import__("base64").b64encode(png).decode(),
                    "media_type": "image/png"}
        if name == "browser_click":
            return session.click(args["target"], by=args.get("by", "auto"))
        if name == "browser_fill":
            return session.fill(args["selector"], args["text"])
        if name == "browser_press":
            return session.press(args["key"], selector=args.get("selector"))
        if name == "browser_scroll":
            return session.scroll(dy=int(args.get("dy", 600)), dx=int(args.get("dx", 0)))
        if name == "browser_hover":
            return session.hover(args["target"])
        if name == "browser_wait_for":
            return session.wait_for(selector=args.get("selector"),
                                     text=args.get("text"),
                                     timeout_ms=int(args.get("timeout_ms", 30000)))
        if name == "browser_url":   return {"url": session.url(), "n_actions": session.n_actions if mode == "web" else 0}
        if name == "browser_title": return {"title": session.title()}
        # observability — backend service logs (always available)
        if name == "list_services":
            from ..tools.service_observability import list_services
            return list_services(run_dir)
        if name == "tail_service_log":
            from ..tools.service_observability import tail_service_log
            return tail_service_log(
                run_dir=run_dir, name=args["name"],
                kind=args.get("kind", "stderr"),
                lines=int(args.get("lines", 100)),
            )
        # observability — browser-side (web mode only)
        if name == "browser_console_log":
            level = args.get("level", "error")
            if level == "all":
                level = None
            return session.console_log(
                level=level,
                since_action=int(args.get("since_action", 0)),
                limit=int(args.get("limit", 50)),
            )
        if name == "browser_network_failures":
            return session.network_failures(
                since_action=int(args.get("since_action", 0)),
                min_status=int(args.get("min_status", 400)),
                limit=int(args.get("limit", 50)),
            )
        # cli ops
        if name == "pty_screen":      return session.screen_text()
        if name == "pty_read_recent": return session.read_recent(int(args.get("n_lines", 30)))
        if name == "pty_send":
            session.send(args["text"])
            return "sent"
        if name == "pty_wait_for":
            ok = session.wait_for(args["pattern"], timeout_s=float(args.get("timeout_s", 30.0)))
            return {"matched": ok, "screen_tail": session.read_recent(15)}
        if name == "pty_is_alive":    return {"alive": session.is_alive(),
                                                "exit_code": session.exit_code()}
        if name == "pty_transcript":
            tail = args.get("tail_lines")
            head = args.get("head_lines")
            if tail is None and head is None:
                tail = 200
            return session.transcript(
                tail_lines=int(tail) if tail is not None else None,
                head_lines=int(head) if head is not None else None,
            )
        if name == "pty_restart":
            return session.restart(
                extra_env=args.get("extra_env") or None,
                extra_args=args.get("extra_args") or None,
                replace_args=args.get("replace_args") or None,
            )
        return f"ERROR: unknown tool {name!r}"

    # ─── main loop
    # Use a vision-capable model: Demo Driver calls browser_screenshot which
    # feeds PNG to the LLM. glm-5.1 + minimax-m2.7 are text-only — they error
    # out with 400 'Model do not support image input'. LLM_VISION env var
    # (default kimi-k2.6) selects an ARK Coding Plan model that accepts images.
    client = anthropic_client()
    model = model_for("vision") if mode == "web" else model_for("reasoning")
    # Per-run prompt override: user may have edited via the Prompts panel.
    from ._prompt_override import get_system_prompt
    effective_system_prompt = get_system_prompt("demo_driver", SYSTEM_PROMPT, run_dir)
    log.info(f"loop start  model={model}  mode={mode}  safety_cap={safety_step_cap}  "
                f"prompt_override={'YES' if effective_system_prompt != SYSTEM_PROMPT else 'NO'}")

    try:
        for step in range(safety_step_cap):
            progress.step = step + 1

            # Splice live feedback before each round
            new_fb = feedback.read_new()
            if new_fb:
                fb_text = "\n".join(
                    f"[USER LIVE FEEDBACK]: {e.get('text', json.dumps(e, ensure_ascii=False))}"
                    for e in new_fb
                )
                log.info(f"live feedback in: {fb_text[:200]}")
                messages.append({"role": "user", "content": fb_text})

            progress.last_action = f"→ LLM (step {step+1})"
            progress.write()

            from .error_agent import llm_call_with_recovery
            resp = llm_call_with_recovery(
                # thinking={"type":"disabled"} — glm-5.1 otherwise spends the
                # full max_tokens budget on hidden reasoning and emits no
                # tool_use blocks. Same fix as quality_judge.py & setup_runner.
                lambda: client.messages.create(
                    model=model, max_tokens=4096,
                    thinking={"type": "disabled"},
                    system=effective_system_prompt, tools=tools, messages=messages,
                ),
                run_dir=run_dir,
                agent="demo_driver",
                step_label=f"step {step + 1} LLM call",
                context_hint={"model": model, "step": step + 1, "mode": mode,
                              "input_msgs": len(messages)},
                log=log,
            )
            log.info(f"step {step+1}  stop={resp.stop_reason}  "
                     f"in={resp.usage.input_tokens}  out={resp.usage.output_tokens}")
            messages.append({"role": "assistant", "content": resp.content})

            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            for tb in (b for b in resp.content if getattr(b, "type", None) == "text"):
                if tb.text and tb.text.strip():
                    log.info(f"  agent: {tb.text.strip()[:300]}")

            if not tool_uses:
                log.info(f"step {step+1}: no tool calls; ending loop")
                break

            tool_results = []
            for tu in tool_uses:
                progress.tool_calls += 1
                progress.last_action = f"{tu.name}({_short_args(tu.input)})"
                progress.write()
                try:
                    out = dispatch(tu.name, tu.input or {})
                except Exception as e:
                    out = f"ERROR: {type(e).__name__}: {e}"
                    log.exception(f"tool {tu.name} failed")
                tool_results.append(_format_tool_result(tu.id, out))
                # After EVERY tool call, save current page screenshot to a
                # known path so the UI can show a 2s-delay live preview.
                # Atomic write via .tmp + replace so reads never see a torn
                # PNG. Skipped for cli mode (no browser session).
                if mode == "web":
                    try:
                        png = session.screenshot(full_page=False)
                        tmp = run_dir / "demo_preview.tmp.png"
                        tmp.write_bytes(png)
                        tmp.replace(run_dir / "demo_preview.png")
                    except Exception:
                        pass

            messages.append({"role": "user", "content": tool_results})

            if finish_payload is not None:
                log.info("finish_demo accepted; exiting loop")
                break
        else:
            log.warning(f"hit safety_step_cap={safety_step_cap} without finish_demo")
            finish_payload = {"summary": "(safety cap hit; no explicit finish_demo)",
                              "completeness": "partial"}
    except Exception:
        progress.status = "error"
        progress.error = "exception in driver loop"
        progress.write()
        log.exception("demo driver loop failed")
        # still attempt to stop session below

    # ─── stop session, finalise outputs
    log.info("stopping session → finalising recording")
    progress.last_action = "stopping session + transcoding"
    progress.write()
    try:
        sresult = session.stop(output_recording)  # both BrowserSession & PtySession
        rec_duration = float(getattr(sresult, "duration_s", 0))
    except Exception as e:
        log.exception(f"session.stop failed: {e}")
        rec_duration = 0.0

    finish_payload = finish_payload or {"summary": "(no finish_demo)",
                                          "completeness": "blocked"}
    summary_md = (
        f"# Demo Driver Summary\n\n"
        f"- mode: {mode}\n"
        f"- recording: {output_recording}\n"
        f"- duration: {rec_duration:.2f}s\n"
        f"- steps: {progress.step}\n"
        f"- captions: {len(captions)}\n"
        f"- completeness: {finish_payload['completeness']}\n\n"
        f"## Agent's summary\n\n{finish_payload['summary']}\n"
    )
    summary_path.write_text(summary_md, encoding="utf-8")

    progress.status = "done"
    progress.last_action = f"completed ({finish_payload['completeness']})"
    progress.write()

    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="demo_driver",
                 name="demo_recording", path=str(output_recording),
                 duration_s=rec_duration, n_captions=len(captions),
                 completeness=finish_payload["completeness"])

    return DemoDriverResult(
        mode=mode,
        recording_path=str(output_recording),
        duration_s=rec_duration,
        captions_path=str(captions_path),
        summary_path=str(summary_path),
        n_captions=len(captions),
        n_steps=progress.step,
        completeness=finish_payload["completeness"],
        finish_summary=finish_payload["summary"],
    )


def _short_args(args: Any, limit: int = 80) -> str:
    s = json.dumps(args, ensure_ascii=False, default=str) if args else ""
    return s[:limit]


def _format_tool_result(tu_id: str, out: Any) -> dict:
    """Anthropic tool_result content: text or image+text mix."""
    if isinstance(out, dict) and "_image_b64" in out:
        return {
            "type": "tool_result",
            "tool_use_id": tu_id,
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": out.get("media_type", "image/png"),
                    "data": out["_image_b64"],
                }},
                {"type": "text", "text": "screenshot above (current viewport)"},
            ],
        }
    if isinstance(out, (dict, list)):
        text = json.dumps(out, ensure_ascii=False, default=str)
    else:
        text = str(out)
    if len(text) > 50000:
        text = text[:50000] + f"\n\n[... truncated; full size {len(text)}B ...]"
    return {"type": "tool_result", "tool_use_id": tu_id, "content": text}


def _build_initial_message(repo_dir: Path, project_brief: str, mode: str,
                            web_url: Optional[str], cli_command: Optional[list]) -> str:
    parts = [
        f"## Repo  `{repo_dir}`",
        f"## Host  {platform.system()} ({platform.platform()})",
        f"## Mode  **{mode}**",
    ]
    if mode == "web":
        parts.append(f"## Target URL  {web_url}")
        parts.append("Browser session is already launched and `goto({url})` was attempted; "
                     "use `browser_screenshot` / `browser_visible_text` / "
                     "`browser_a11y_snapshot` / `browser_interactables` to see what's there. "
                     "Then start operating it.")
    else:
        parts.append(f"## Command  `{' '.join(cli_command or [])}`")
        parts.append("CLI session is already spawned. Use `pty_screen` to see current "
                     "terminal state; `pty_send`/`pty_wait_for` to operate.")
    parts.append("")
    parts.append("## Project brief (TONE/AUDIENCE only — NOT a checklist)")
    parts.append("```markdown")
    parts.append(project_brief.strip()[:4000])
    parts.append("```")
    parts.append("")
    parts.append("Begin: read enough source code to understand what this project is, "
                 "then operate it end-to-end and mark captions for meaningful beats. "
                 "Call finish_demo when (and only when) you have demonstrated the "
                 "project's core functionality. NO time limit.")
    return "\n".join(parts)
