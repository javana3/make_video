"""Agent 1 · ProjectAnalyzer (WORKFLOW.md §1).

Reads a cloned GitHub project and produces project_brief.md describing:
  - 产品一句话定位
  - 作用与目的
  - 目标受众
  - 3–5 个独特卖点
  - 视觉关键词
  - 不超过 3 个竞品参考

Iterates with the User until approved (Gate #1). Each iteration is one Agent
invocation; user feedback is injected as additional context on rerun.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from ..observability.audit import traced_agent
from ..observability.logger import agent_logger
from ..tools.llm import anthropic_client, model_for


Mode = Literal["standard", "deep"]


SYSTEM_PROMPT_BASE = """You are Agent 1 ProjectAnalyzer in a promo-video production pipeline.

Your job: analyze a cloned GitHub repository and produce `project_brief.md`,
which downstream phases (HTML design, video composition, voice-over) will all
consume. The brief defines positioning, audience, selling points, and visual
direction.

The brief MUST contain these sections in this order, written in 中文 (Chinese), formatted as Markdown:

# {Project Name} · 项目简报

## 产品一句话定位
(one sentence — must be punchy and concrete, not corporate fluff)

## 作用与目的
(one paragraph — what it does, why it exists, who/what problem it serves)

## 目标受众
- bullet 1
- bullet 2
- (3–5 bullets)

## 独特卖点
- 卖点 1 — short concrete description
- 卖点 2 — ...
- (3–5 selling points; each tied to specific features in the codebase)

## 视觉关键词
极简 / B&W / 金色 / 电影感 / ... (5–8 keywords for the OpenDesign collaboration)

## 竞品参考
- 竞品 1 (URL if known) — what makes it comparable
- (up to 3)

Approach:
1. Use `list_dir` on the repo root, then read README first.
2. Read main entry points (e.g. main.py / index.ts / src/...) and any prompt
   or config files that reveal *distinctive* behavior or product logic.
3. Don't dump source code into the brief — synthesize.
4. When ready, call `emit_brief` exactly once with TWO arguments:
   - `markdown`: the final brief content
   - `claim_sources`: array of {claim, source_file} entries.
     **Coverage requirement (HARD RULE, enforced by the host)**:
     `len(claim_sources)` MUST be ≥ the count of bullets in
     "独特卖点" + "技术架构亮点" + "核心功能实现" sections combined.
     Provide ONE entry per bullet. The `claim` field should match the bullet's
     lead text. The `source_file` MUST be one of the files you have read via
     `read_file`. The host will REJECT and force a retry if:
       (a) a source_file was not actually read, or
       (b) claim_sources count < substantive-bullet count.
     **If a bullet has no source-file support, REMOVE the bullet from the
     brief — do NOT leave ungrounded claims.**

If a tool result ends with `[... truncated ...]`, the file was cut off; if you
need the rest to ground a claim, re-call `read_file` with `max_bytes=131072`.

Be concise. Aim for the brief itself to be ~600–1000 characters, not a wall of text.
"""


DEEP_MODE_ADDENDUM = """

═══════════════════════════════════════════════════════════════════
DEEP ANALYSIS MODE — User has explicitly requested thorough exploration.
═══════════════════════════════════════════════════════════════════

The standard pass (README + maybe one design doc) is NOT sufficient for this
run. You MUST also:

1. **Stack inspection**: read package.json / pyproject.toml / requirements.txt
   / Cargo.toml — identify primary dependencies and what they reveal.
2. **Main entry**: read the main entry point(s) — main.py / app.py / index.ts
   / src/main.* — understand the runtime architecture.
3. **Distinctive logic**: sample 3–5 source files that contain the project's
   unique secret sauce (LLM prompt files, simulation engines, custom
   algorithms, novel data structures).
4. **All docs**: read DESIGN.md / ARCHITECTURE.md / docs/*.md if present.

Target 15–25 tool calls before `emit_brief`. The resulting brief should
demonstrate insight that **could not** come from README alone — cite specific
algorithms, file structures, or implementation patterns when describing
selling points.

In `claim_sources`, prefer source-code files (.py / .ts / .js / .sql / .rs /
.go) over docs. Aim for ≥1 distinct source-code file per "技术架构亮点" and
"核心功能实现" bullet. README citations alone are not sufficient evidence of
deep analysis — the host UI labels source-only-from-README as suspicious.
"""


def _system_prompt(mode: Mode, run_dir: Optional[Path] = None) -> str:
    """Resolve effective system prompt. Per-run override (saved by web UI's
    Prompts panel at <run_dir>/prompts/project_analyzer.txt) takes precedence
    over the module default; mode affects only the default path.
    """
    if run_dir is not None:
        from ._prompt_override import get_system_prompt
        default = SYSTEM_PROMPT_BASE + (DEEP_MODE_ADDENDUM if mode == "deep" else "")
        return get_system_prompt("project_analyzer", default, run_dir)
    if mode == "deep":
        return SYSTEM_PROMPT_BASE + DEEP_MODE_ADDENDUM
    return SYSTEM_PROMPT_BASE


TOOLS = [
    {
        "name": "list_dir",
        "description": "List entries (files and subdirs) in a directory relative to the repo root. Use \"\" or \".\" for root.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the repo. Returns up to 64KB by default; large files are truncated and the result ends with '[... truncated; total file size NB ...]' — re-call with `max_bytes=131072` if you need the rest.",
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
        "description": "Find files matching a glob pattern (e.g. **/*.py, **/README*).",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "emit_brief",
        "description": (
            "Emit the final brief. The host validates `claim_sources` — every "
            "`source_file` MUST be in the set of files you have read via "
            "`read_file`. If validation fails, the tool returns an error and "
            "you must call emit_brief again (after reading missing files or "
            "removing unsupported claims)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "markdown": {
                    "type": "string",
                    "description": "Final project_brief.md content (Chinese).",
                },
                "claim_sources": {
                    "type": "array",
                    "description": "One entry per substantive claim. Required.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {
                                "type": "string",
                                "description": "Short paraphrase of the bullet/claim being cited.",
                            },
                            "source_file": {
                                "type": "string",
                                "description": "Repo-relative path to a file you read via read_file.",
                            },
                            "evidence": {
                                "type": "string",
                                "description": "Optional: a quote or line range from the file supporting the claim.",
                            },
                        },
                        "required": ["claim", "source_file"],
                    },
                },
            },
            "required": ["markdown", "claim_sources"],
        },
    },
]


def _safe_path(repo_dir: Path, rel: str) -> Path:
    """Resolve `rel` inside `repo_dir`. Reject path traversal."""
    rel = rel.lstrip("/").lstrip("\\") or "."
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
    if len(entries) > 200:
        out.append(f"... ({len(entries) - 200} more entries)")
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
    return text + (f"\n\n[... truncated; total file size {p.stat().st_size}B ...]" if truncated else "")


def _tool_find_files(repo_dir: Path, args: dict) -> str:
    pattern = args["pattern"]
    matches = sorted(repo_dir.glob(pattern))[:100]
    if not matches:
        return "(no matches)"
    return "\n".join(str(m.relative_to(repo_dir)).replace("\\", "/") for m in matches)


def _build_initial_message(repo_dir: Path, repo_url: str,
                           feedback: Optional[str] = None,
                           previous_brief: Optional[str] = None) -> str:
    parts = [
        f"Repo URL:    {repo_url}",
        f"Local path:  {repo_dir}",
        "",
    ]
    if previous_brief:
        parts.append("Previous brief draft (for context):")
        parts.append("```markdown")
        parts.append(previous_brief)
        parts.append("```")
        parts.append("")
    if feedback:
        parts.append("User feedback on the previous brief — incorporate this:")
        parts.append(feedback)
        parts.append("")
        parts.append("Produce a revised brief. Re-explore the repo only if needed to address the feedback.")
    else:
        parts.append("Begin: list the repo root, read README, then explore distinguishing files.")
        parts.append("Call `emit_brief` when the brief is complete.")
    return "\n".join(parts)


@traced_agent("Agent 1 ProjectAnalyzer", phase=1)
def run_project_analyzer(repo_dir: Path,
                         repo_url: str,
                         output_path: Path,
                         feedback: Optional[str] = None,
                         previous_brief: Optional[str] = None,
                         mode: Mode = "standard",
                         max_steps: Optional[int] = None,
                         progress_path: Optional[Path] = None) -> Path:
    """Run the agent loop. Writes the brief to `output_path`.

    Also writes:
    - `<output_path.parent>/<output_path.stem>_meta.json` with sources/tool_calls
    - `progress_path` (if provided) updated each step with running progress
      (UI polls this for the live progress card).
    """
    import time as _time
    if max_steps is None:
        max_steps = 60 if mode == "deep" else 30

    log = agent_logger("agent1_analyzer")
    client = anthropic_client()
    model = model_for("reasoning")
    started_at = datetime.now(timezone.utc)
    started_mono = _time.monotonic()
    log.info(
        f"start  repo={repo_dir.name}  model={model}  mode={mode}  "
        f"iteration={'revise' if feedback else 'first'}"
    )

    files_read: list[str] = []
    tool_call_count = 0

    def write_progress(step: int, last_action: str, status: str = "running",
                       error: Optional[str] = None) -> None:
        if progress_path is None:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": mode,
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

    messages = [
        {"role": "user", "content": _build_initial_message(
            repo_dir, repo_url, feedback=feedback, previous_brief=previous_brief
        )},
    ]

    final_brief: Optional[str] = None
    final_claim_sources: list[dict] = []

    for step in range(max_steps):
        log.info(f"step {step+1}/{max_steps} → LLM")
        write_progress(step + 1, "→ LLM (waiting response)")
        from .error_agent import llm_call_with_recovery
        resp = llm_call_with_recovery(
            lambda: client.messages.create(
                model=model,
                max_tokens=8192,
                system=_system_prompt(mode, run_dir=output_path.parent),
                tools=TOOLS,
                messages=messages,
            ),
            run_dir=output_path.parent,
            agent="project_analyzer",
            step_label=f"step {step + 1} LLM call",
            context_hint={"model": model, "step": step + 1, "max_steps": max_steps,
                          "mode": mode, "input_msgs": len(messages)},
            log=log,
        )
        if resp.stop_reason == "max_tokens":
            log.warning(f"step {step+1}: stop_reason=max_tokens "
                        f"(in={resp.usage.input_tokens} out={resp.usage.output_tokens}); "
                        f"response may be cut off")
        log.info(f"step {step+1} ← stop_reason={resp.stop_reason}  in={resp.usage.input_tokens} out={resp.usage.output_tokens}")

        # Append assistant turn verbatim (preserves tool_use ids for follow-up)
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
                        rel = _to_rel(repo_dir, tu.input.get("path", ""))
                        if rel and rel not in files_read:
                            files_read.append(rel)
                elif tu.name == "find_files":
                    short_arg = tu.input.get("pattern", "")
                    out = _tool_find_files(repo_dir, tu.input)
                elif tu.name == "emit_brief":
                    md = tu.input.get("markdown") or ""
                    sources = tu.input.get("claim_sources") or []
                    out = _validate_emit_brief(md, sources, files_read)
                    if out.startswith("OK"):
                        final_brief = md
                        final_claim_sources = sources
                else:
                    out = f"ERROR: unknown tool {tu.name}"
            except Exception as e:
                out = f"ERROR: {type(e).__name__}: {e}"
                log.exception(f"tool {tu.name} failed")
            # Truncation marker for any tool_result over 50KB
            if len(out) > 50000:
                out = out[:50000] + f"\n\n[... tool_result truncated; full size {len(out)}B ...]"
            short_arg_disp = _to_rel(repo_dir, short_arg) if short_arg else ""
            write_progress(step + 1,
                           f"{tu.name}({short_arg_disp})" if short_arg_disp
                           else tu.name)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": out,
            })

        messages.append({"role": "user", "content": tool_results})

        if final_brief is not None:
            log.info("emit_brief called → finalizing")
            break
    else:
        log.warning(f"hit max_steps={max_steps} without emit_brief")

    if not final_brief:
        write_progress(max_steps, "FAILED: no brief emitted", status="error",
                       error="agent finished without emit_brief")
        raise RuntimeError("Agent 1 finished without emitting a brief")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_brief, encoding="utf-8")
    log.info(f"brief written: {output_path} ({len(final_brief)} chars)")

    # Persist sources metadata next to the brief: <stem>_meta.json
    sources_path = output_path.parent / f"{output_path.stem}_meta.json"
    sources_meta = {
        "mode": mode,
        "files_read": files_read,
        "tool_calls": tool_call_count,
        "claim_sources": final_claim_sources,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    sources_path.write_text(
        json.dumps(sources_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Also write into briefs/<mode>.md so the web UI's briefs_panel picks it up.
    # The panel renders versioned files (standard.md / deep.md) for iterate workflow;
    # the top-level project_brief.md is the canonical artifact for downstream phases.
    briefs_dir = output_path.parent / "briefs"
    briefs_dir.mkdir(parents=True, exist_ok=True)
    (briefs_dir / f"{mode}.md").write_text(final_brief, encoding="utf-8")
    (briefs_dir / f"{mode}_meta.json").write_text(
        json.dumps(sources_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(
        f"sources recorded: {len(files_read)} file(s) read, "
        f"{tool_call_count} tool calls, mode={mode}"
    )
    write_progress(max_steps, "completed", status="done")

    return output_path


_SUBSTANTIVE_SECTIONS = ("独特卖点", "技术架构亮点", "核心功能实现")


_BULLET_PREFIX_RE = re.compile(r"^\s*(?:-|\*|\d+\.|\d+、)\s*")


def _extract_substantive_bullets(markdown: str) -> list[str]:
    """Find every bullet headline in the substantive sections.

    Accepts bullet prefixes: `-`, `*`, `1.`, `1、` (numbered list, half/full-width).
    Headline is the **bold** lead text, or the phrase before `:` / `：` / `—` / `--`.
    """
    bullets: list[str] = []
    section_re = re.compile(r"^##\s*(.+)\s*$", re.MULTILINE)
    matches = list(section_re.finditer(markdown))
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        if not any(s in name for s in _SUBSTANTIVE_SECTIONS):
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end]
        for line in body.split("\n"):
            if not _BULLET_PREFIX_RE.match(line):
                continue
            content = _BULLET_PREFIX_RE.sub("", line, count=1).strip()
            if not content:
                continue
            mm = re.match(r"\*\*([^*]+)\*\*", content)
            if mm:
                bullets.append(mm.group(1).strip().rstrip(":："))
                continue
            for sep in ("——", "—", "：", ":"):
                if sep in content:
                    head = content.split(sep, 1)[0].strip()
                    if head:
                        bullets.append(head)
                    break
    return bullets


def _bullet_covered_by_claim(bullet: str, claim: str) -> bool:
    """Loose match: claim must share a meaningful token with bullet."""
    norm = lambda s: re.sub(r"[\s·\-_/、，,。\.（）()：:；;]+", "", s.lower())
    nb, nc = norm(bullet), norm(claim)
    if not nb or not nc:
        return False
    if nb in nc or nc in nb:
        return True
    # Take 4-char windows of bullet and check if any are in claim
    for i in range(len(nb) - 3):
        if nb[i:i + 4] in nc:
            return True
    return False


def _validate_emit_brief(markdown: str, claim_sources: list,
                         files_read: list[str]) -> str:
    """Validate emit_brief input. Return string starting with 'OK' on pass,
    'ERROR: ...' on fail (used as the tool_result so the agent retries)."""
    if not isinstance(markdown, str) or not markdown.strip():
        return "ERROR: `markdown` must be a non-empty string."
    if not isinstance(claim_sources, list) or not claim_sources:
        return ("ERROR: `claim_sources` must be a non-empty array of "
                "{claim, source_file} entries — at least one entry per "
                "substantive bullet (selling point / tech / core feature).")
    files_read_norm = {f.replace("\\", "/").strip("/") for f in files_read}
    bad: list[str] = []
    seen_files: set[str] = set()
    for i, s in enumerate(claim_sources):
        if not isinstance(s, dict):
            return f"ERROR: claim_sources[{i}] is not an object"
        sf = (s.get("source_file") or "").replace("\\", "/").strip("/")
        if not sf:
            return f"ERROR: claim_sources[{i}] missing 'source_file'"
        if sf not in files_read_norm:
            bad.append(sf)
        else:
            seen_files.add(sf)
    if bad:
        return ("ERROR: claim_sources references files NOT yet read: "
                f"{sorted(set(bad))}. Files you have read: "
                f"{sorted(files_read_norm)}. Either call `read_file` on the "
                f"missing files first, or drop those entries from "
                f"`claim_sources`, then re-call `emit_brief`.")

    # Coverage check: claim_sources must have ≥ 1 entry per substantive bullet.
    required_bullets = _extract_substantive_bullets(markdown)
    if len(claim_sources) < len(required_bullets):
        return (f"ERROR: brief has {len(required_bullets)} substantive bullets "
                f"across 独特卖点 / 技术架构亮点 / 核心功能实现 sections "
                f"({required_bullets!r}), but `claim_sources` only has "
                f"{len(claim_sources)} entries. Add ONE entry per bullet — "
                f"the `claim` field should match the bullet's lead text and "
                f"`source_file` should cite a file that supports it. If a "
                f"bullet has no source-file support, REMOVE it from the brief "
                f"instead of leaving it ungrounded. Then re-call emit_brief.")

    # Coverage by topic: each bullet must appear (loosely) in some claim text.
    # This stops the agent from padding with vague catch-all entries
    # (e.g., '项目架构与设计文档', '产品定位与功能概述').
    claim_texts = [str(s.get("claim") or "") for s in claim_sources]
    uncovered = [b for b in required_bullets
                 if not any(_bullet_covered_by_claim(b, c) for c in claim_texts)]
    if uncovered:
        return (f"ERROR: claim_sources count is sufficient ({len(claim_sources)}) "
                f"but {len(uncovered)} bullet(s) have NO matching `claim` text: "
                f"{uncovered!r}. Each bullet's `claim` field should mention the "
                f"bullet's lead concept (e.g., bullet 'Dixon-Coles 概率模型' "
                f"needs a claim_sources entry with claim text containing "
                f"'Dixon-Coles' or '概率模型'). Generic entries like "
                f"'产品定位与功能概述' do not count. Fix and re-call emit_brief.")

    return (f"OK; brief recorded with {len(claim_sources)} claim sources "
            f"covering {len(seen_files)} distinct files for "
            f"{len(required_bullets)} substantive bullets.")


def _to_rel(repo_dir: Path, path_str: str) -> str:
    """Convert a path returned by the Agent to a repo-relative POSIX path."""
    if not path_str:
        return ""
    try:
        p = Path(path_str)
        if p.is_absolute():
            try:
                p = p.relative_to(repo_dir.resolve())
            except ValueError:
                return ""
        return str(p).replace("\\", "/")
    except Exception:
        return ""
