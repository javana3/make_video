"""Agent 3 · RemotionComposer — cutting plan phase (M3a).

Reads project_brief + recording metadata. Produces a structured cutting_plan
with scenes, backgrounds, text overlays, and transitions. Host validates the
plan against R3 (head/tail 5s skip), R4 (background per scene type), R5
(crossfade 15 frames).

Plan is the input to M3b (deterministic code generation → Remotion TSX).
"""
from __future__ import annotations

import json
import re
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..observability.audit import traced_agent
from ..observability.logger import agent_logger
from ..tools.llm import anthropic_client, model_for


SYSTEM_PROMPT = """You are Agent 3 RemotionComposer in a promo-video production pipeline.

Phase 1 (project_brief) and Phase 2 (services + recording) are done. Your job
in Phase 3a: design a CUTTING PLAN for a 30–60 second promo video.

You will be given:
- project_brief.md (positioning, audience, selling points, 视觉关键词)
- recording metadata (duration_s, width, height, fps, codec, source_path)

You produce a structured `cutting_plan` describing scenes, backgrounds, text
overlays, and transitions. The host translates this plan into a Remotion TSX
composition (M3b) and renders it (M3c).

═══════════════════════════════════════════════════════════════════
HARD RULES (host-enforced; violations return ERROR and you must retry)
═══════════════════════════════════════════════════════════════════

R3 (recording head/tail 5s skip):
  For every scene whose background.type == "recording":
    - background.start_in_source_s >= 5.0
    - background.start_in_source_s + scene.duration_s <= recording.duration_s - 5.0
  Reason: ffmpeg recordings always have unstable head/tail frames.

R4 (background per scene type):
  Title-style scenes (text font_size_px > 80 AND content len <= 10 chars):
    → MUST use background.type == "recording" with darken 0.65–0.85
    Reason: big text on plain solid color is flat; recording behind looks cinematic.
  Body-style scenes (small text, multiple elements, lists):
    → MUST use background.type == "color" or "gradient"
    Reason: small text overlaid on busy recording is unreadable.
  Pure recording display scenes (no text overlay):
    → background.type == "recording", darken 0.

R5 (crossfade 15 frames between scenes):
  For each pair of consecutive scenes Sn, Sn+1, you SHOULD provide a transition
  with kind="crossfade" and duration_s == 0.5 (= 15 frames @ 30fps).
  Cut transitions allowed but visually jarring; prefer crossfade.

Approach:
1. Read project_brief.md to understand 视觉关键词 + 卖点.
2. Plan 5–7 scenes spanning 30–45 seconds total.
3. Open with title scene (recording bg + darken + big text 卖点 1).
4. Middle scenes alternate between "showcase recording" and "small-text body".
5. End with outro (recording bg + darken + brand statement).
6. Call `emit_cutting_plan` with the structured plan.

Be specific. Pull text DIRECTLY from 独特卖点 / 产品一句话定位. Don't invent.
"""


TOOLS = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file by repo-relative path. Allowed: project_brief.md, briefs/*.md.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "emit_cutting_plan",
        "description": (
            "Emit the final cutting plan. Host validates R3/R4/R5 + schema; "
            "returns ERROR on violation, you must retry with fixes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "1-sentence overview."},
                "fps": {"type": "integer", "description": "Render fps; use 30."},
                "resolution_w": {"type": "integer", "description": "Render width; use 1920."},
                "resolution_h": {"type": "integer", "description": "Render height; use 1080."},
                "scenes": {
                    "type": "array",
                    "minItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "e.g. 'S1', 'intro'."},
                            "duration_s": {"type": "number"},
                            "background": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["color", "gradient", "recording"],
                                    },
                                    "color": {"type": "string", "description": "#RRGGBB; for color/gradient."},
                                    "gradient_to": {"type": "string", "description": "#RRGGBB; for gradient."},
                                    "source_path": {"type": "string", "description": "Repo-relative recording path; for recording."},
                                    "start_in_source_s": {"type": "number", "description": "For recording; ≥ 5.0."},
                                },
                                "required": ["type"],
                            },
                            "darken": {
                                "type": "number",
                                "description": "0..0.85; only for recording bg with text overlay.",
                            },
                            "elements": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string", "enum": ["text"]},
                                        "content": {"type": "string"},
                                        "font_size_px": {"type": "integer"},
                                        "color": {"type": "string"},
                                        "x": {"type": "string", "description": "'center' or '<int>px' or '<int>%'."},
                                        "y": {"type": "string"},
                                        "anim_in": {
                                            "type": "string",
                                            "enum": ["fadeIn", "slideUp", "slideDown", "scale", "none"],
                                        },
                                        "anim_out": {
                                            "type": "string",
                                            "enum": ["fadeOut", "slideUp", "slideDown", "scale", "none"],
                                        },
                                    },
                                    "required": ["type", "content", "font_size_px"],
                                },
                            },
                        },
                        "required": ["id", "duration_s", "background", "elements"],
                    },
                },
                "transitions": {
                    "type": "array",
                    "description": "Optional; missing transitions default to crossfade 0.5s.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from_scene": {"type": "string"},
                            "to_scene": {"type": "string"},
                            "kind": {"type": "string", "enum": ["crossfade", "cut"]},
                            "duration_s": {"type": "number"},
                        },
                        "required": ["from_scene", "to_scene", "kind"],
                    },
                },
            },
            "required": ["summary", "fps", "resolution_w", "resolution_h", "scenes"],
        },
    },
]


def _validate_plan(plan: dict, recording_duration_s: float) -> str:
    """Returns 'OK; ...' on pass, 'ERROR: ...' on fail."""
    if not isinstance(plan, dict):
        return "ERROR: cutting_plan is not an object"

    fps = plan.get("fps")
    if fps != 30:
        return f"ERROR: fps must be 30, got {fps}. Remotion compositions use 30fps."
    if plan.get("resolution_w") != 1920 or plan.get("resolution_h") != 1080:
        return "ERROR: resolution must be 1920x1080 for promo videos"

    scenes = plan.get("scenes") or []
    if len(scenes) < 3:
        return f"ERROR: need ≥ 3 scenes, got {len(scenes)}"

    # Build id index for transition validation
    ids: list[str] = []
    for i, s in enumerate(scenes):
        sid = s.get("id") or f"S{i+1}"
        if sid in ids:
            return f"ERROR: scenes[{i}].id={sid!r} is duplicated"
        ids.append(sid)
        dur = s.get("duration_s", 0)
        if dur < 0.5 or dur > 30:
            return f"ERROR: scenes[{i}].duration_s={dur} out of range 0.5..30"

        bg = s.get("background") or {}
        bg_type = bg.get("type")
        if bg_type not in ("color", "gradient", "recording"):
            return f"ERROR: scenes[{i}].background.type must be color|gradient|recording"

        # R3: recording head/tail 5s
        if bg_type == "recording":
            start = bg.get("start_in_source_s")
            if start is None:
                return f"ERROR: scenes[{i}] background.recording missing start_in_source_s"
            if start < 5.0:
                return (f"ERROR: scenes[{i}].background.start_in_source_s={start} "
                        f"violates R3 (must be ≥ 5.0; head 5s of recording is unstable)")
            end = start + dur
            tail_limit = recording_duration_s - 5.0
            if end > tail_limit:
                return (f"ERROR: scenes[{i}] recording_clip ends at {end:.2f}s but "
                        f"recording is only {recording_duration_s:.2f}s (R3: must end "
                        f"by {tail_limit:.2f}s — last 5s of recording unusable). "
                        f"Either shorten scene.duration_s or pick smaller start_in_source_s.")
            if not bg.get("source_path"):
                return f"ERROR: scenes[{i}] background.recording missing source_path"

        # R4: background per scene type
        elements = s.get("elements") or []
        text_els = [e for e in elements if e.get("type") == "text"]
        if text_els:
            biggest = max(int(e.get("font_size_px", 0)) for e in text_els)
            shortest_big_text = min(
                (len(e.get("content", "")) for e in text_els if int(e.get("font_size_px", 0)) > 80),
                default=999,
            )
            is_title_style = biggest > 80 and shortest_big_text <= 10
            is_body_style = biggest <= 80 or any(
                len(e.get("content", "")) > 30 for e in text_els
            )
            if is_title_style and bg_type == "color":
                return (f"ERROR: scenes[{i}] is title-style (big text font={biggest}px, "
                        f"short content) but background is solid color — R4 says title "
                        f"scenes need recording bg + darken 0.65-0.85.")
            if is_body_style and bg_type == "recording" and (s.get("darken") or 0) < 0.6:
                return (f"ERROR: scenes[{i}] body-style content over recording with "
                        f"darken={s.get('darken')} — small text on busy bg is unreadable. "
                        f"Either change to color/gradient bg, or set darken ≥ 0.7 (R4).")

    # R5 transitions
    transitions = plan.get("transitions") or []
    for t in transitions:
        if t.get("kind") == "crossfade":
            dur = t.get("duration_s", 0)
            if dur < 0.4:  # 12 frames @ 30fps; allow slight slop
                return (f"ERROR: transition {t.get('from_scene')} → {t.get('to_scene')} "
                        f"crossfade duration_s={dur} too short (R5: ≥ 0.5s = 15 frames).")

    total = sum(s.get("duration_s", 0) for s in scenes)
    if total < 15 or total > 90:
        return f"ERROR: total scene duration {total:.1f}s out of range 15..90s"

    return f"OK; cutting plan validated ({len(scenes)} scenes, {total:.1f}s total)."


def _safe_path(repo_dir: Path, rel: str) -> Path:
    rel = (rel or ".").lstrip("/").lstrip("\\") or "."
    p = (repo_dir / rel).resolve()
    if p != repo_dir.resolve() and repo_dir.resolve() not in p.parents:
        raise ValueError(f"path escapes run_dir: {rel!r}")
    return p


def _tool_read_file(run_dir: Path, args: dict) -> str:
    p = _safe_path(run_dir, args["path"])
    if not p.exists() or not p.is_file():
        return f"ERROR: {args['path']!r} does not exist"
    raw = p.read_bytes()[:65536]
    return raw.decode("utf-8", errors="replace")


def _build_initial_message(project_brief: str, recording_meta: dict,
                           feedback: Optional[str] = None,
                           previous_plan: Optional[dict] = None) -> str:
    parts = [
        "## Project brief",
        "```markdown",
        project_brief.strip(),
        "```",
        "",
        "## Recording metadata",
        f"- source_path: {recording_meta['source_path']!r} (use this exactly in background.source_path)",
        f"- duration_s:  {recording_meta['duration_s']:.2f}",
        f"- size:        {recording_meta['width']}x{recording_meta['height']}",
        f"- codec:       {recording_meta['codec']}",
        "",
    ]
    if previous_plan:
        parts.append("## Previous cutting plan (for revision)")
        parts.append("```json")
        parts.append(json.dumps(previous_plan, ensure_ascii=False, indent=2)[:4000])
        parts.append("```")
        parts.append("")
    if feedback:
        parts.append("## User feedback to incorporate")
        parts.append(feedback)
        parts.append("")
        parts.append("Produce a revised cutting_plan.")
    else:
        parts.append("Begin: design a 30-45s plan. Pull text directly from 独特卖点 of the brief. "
                     "Call `emit_cutting_plan` with the structured plan.")
    return "\n".join(parts)


@traced_agent("Agent 3 RemotionComposer · plan", phase=3)
def run_cutting_planner(run_dir: Path,
                        project_brief: str,
                        recording_meta: dict,
                        output_path: Path,
                        feedback: Optional[str] = None,
                        previous_plan: Optional[dict] = None,
                        progress_path: Optional[Path] = None,
                        max_steps: int = 20) -> Path:
    """Run the agent loop. Writes cutting_plan.json. Returns its path."""
    log = agent_logger("agent3_remotion")
    client = anthropic_client()
    model = model_for("reasoning")
    started_at = datetime.now(timezone.utc)
    started_mono = _time.monotonic()
    log.info(f"start  recording_dur={recording_meta.get('duration_s'):.1f}s  model={model}")

    files_read: list[str] = []
    tool_call_count = 0

    def write_progress(step: int, last_action: str, status: str = "running",
                       error: Optional[str] = None) -> None:
        if progress_path is None:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "phase": "3a-cutting-plan",
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
    messages = [{
        "role": "user",
        "content": _build_initial_message(project_brief, recording_meta,
                                          feedback=feedback, previous_plan=previous_plan),
    }]
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
        if resp.stop_reason == "max_tokens":
            log.warning(f"step {step+1}: stop_reason=max_tokens")
        log.info(f"step {step+1} ← stop_reason={resp.stop_reason}  in={resp.usage.input_tokens} out={resp.usage.output_tokens}")
        messages.append({"role": "assistant", "content": resp.content})

        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        text_blocks = [b for b in resp.content if getattr(b, "type", None) == "text"]
        for tb in text_blocks:
            if tb.text and tb.text.strip():
                log.info(f"  agent: {tb.text.strip()[:300]}")

        if not tool_uses:
            break

        tool_results = []
        for tu in tool_uses:
            log.info(f"  tool: {tu.name}({str(tu.input)[:200]})")
            tool_call_count += 1
            try:
                if tu.name == "read_file":
                    out = _tool_read_file(run_dir, tu.input)
                    if not out.startswith("ERROR"):
                        rel = tu.input.get("path", "").replace("\\", "/").lstrip("/")
                        if rel and rel not in files_read:
                            files_read.append(rel)
                elif tu.name == "emit_cutting_plan":
                    out = _validate_plan(tu.input, recording_meta["duration_s"])
                    if out.startswith("OK"):
                        final_plan = dict(tu.input)
                else:
                    out = f"ERROR: unknown tool {tu.name}"
            except Exception as e:
                out = f"ERROR: {type(e).__name__}: {e}"
                log.exception(f"tool {tu.name}")
            if len(out) > 50000:
                out = out[:50000] + f"\n[... truncated; {len(out)}B ...]"
            short_arg = tu.input.get("path") or tu.name
            write_progress(step + 1, f"{tu.name}({short_arg})")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": out,
            })

        messages.append({"role": "user", "content": tool_results})

        if final_plan is not None:
            log.info("emit_cutting_plan accepted → finalizing")
            break
    else:
        write_progress(max_steps, "FAILED: no plan emitted", status="error",
                       error="agent loop exhausted")
        raise RuntimeError("Agent 3 finished without emitting a cutting_plan")

    if not final_plan:
        write_progress(max_steps, "FAILED: no plan emitted", status="error",
                       error="agent loop exhausted")
        raise RuntimeError("Agent 3 finished without emitting a cutting_plan")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(final_plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(f"cutting_plan written: {output_path}")
    write_progress(max_steps, "completed", status="done")
    return output_path
