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

Phase 1 (project_brief), Phase 2 (Demo Driver recording of the project actually
running), and OpenDesigner (hyperframes / html design assets) are done.
Your job in Phase 3a: design a CUTTING PLAN for the promo video that mixes
those assets meaningfully.

You will be given:
- project_brief.md (positioning, audience, 独特卖点 — TONE/AUDIENCE only,
  not a checklist)
- demo_captions.jsonl (the Demo Driver's bilingual captions, timestamped to
  the recording — you can choose to surface these as overlay text)
- available_assets: every clip/page you can use as a scene background, with
  paths and durations. Three kinds of background sources exist:
    • `recording` — the Demo Driver's screen capture of the project
                     (real interaction, real screens — this is your
                     authenticity asset)
    • `hyperframe` — OpenDesigner's motion_film output: short polished
                     animation .mp4s designed by an HTML→MP4 designer-agent
                     (this is your design-feel asset)
    • `html`        — OpenDesigner's static_hero output: a designed HTML
                     page (rasterised at render time as a still or scrolled
                     viewport — use for hero/intro/outro visual beats)

YOU decide how to mix them. There is no preset ratio. Use real demo
recording where authenticity matters; use hyperframes/html where polish
matters. A typical good mix opens on a hyperframe or html beat, drops
into demo recording for the meat, and closes on a hyperframe — but you
might decide differently depending on what the assets are like and what
the project does.

═══════════════════════════════════════════════════════════════════
HARD RULES (host-enforced; violations return ERROR and you must retry)
═══════════════════════════════════════════════════════════════════

R3 (recording head/tail 5s skip):
  For every scene whose background.type == "recording":
    - background.start_in_source_s >= 5.0
    - background.start_in_source_s + scene.duration_s <= recording.duration_s - 5.0
  Reason: screen captures often have unstable head/tail frames.
  (Hyperframes / html have NO head/tail rule — OpenDesigner output is clean.)

R4 (legibility — text on busy backgrounds):
  Title-style scenes (text font_size_px > 80 AND content len <= 10 chars):
    → MUST use background.type ∈ {recording, hyperframe} with darken 0.65–0.85
      OR background.type == "html" (designed pages have built-in legibility)
    Reason: big short text needs visual weight — solid color is flat.
  Body-style scenes (small text, multiple elements, lists):
    → MUST use background.type ∈ {color, gradient, html} (with html being a
      designed page, not arbitrary content)
      OR recording/hyperframe with darken ≥ 0.7.
    Reason: small text on busy bg is unreadable.

R5 (crossfade 15 frames between scenes):
  Consecutive scenes SHOULD have transitions: kind="crossfade", duration_s=0.5
  (= 15 frames @ 30fps). Cuts are allowed but jarring.

R6 (path existence):
  Every background.source_path must exist in available_assets exactly. The
  host will check.

Approach:
1. Read project_brief.md for tone (NOT for content selection).
2. Inspect available_assets to see what each clip/page actually looks like
   (you can use `read_file` on demo_captions.jsonl to see what was
   captured during the demo).
3. Plan 5–8 scenes spanning 30–45 seconds total.
4. Decide mix: where to use real demo (authenticity), where to use
   hyperframes/html (polish). Justify each choice in scene.notes if helpful.
5. Call `emit_cutting_plan`.

Be specific and source-truth-aligned. Pull caption text from demo_captions
where it lines up; pull hero text from 独特卖点 only when those phrases
match what's actually shown on screen.
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
                                        "enum": ["color", "gradient", "recording",
                                                  "hyperframe", "html"],
                                    },
                                    "color": {"type": "string", "description": "#RRGGBB (color/gradient)."},
                                    "gradient_to": {"type": "string", "description": "#RRGGBB (gradient)."},
                                    "source_path": {"type": "string",
                                                     "description": "Run-dir-relative path. recording → recordings/test.mp4; hyperframe → hyperframes/<name>.mp4; html → html_asset/index.html (or similar)."},
                                    "start_in_source_s": {"type": "number",
                                                            "description": "Where to seek into the source. recording: ≥ 5.0; hyperframe: ≥ 0; html: ignored."},
                                    "html_scroll_y_pct": {"type": "number",
                                                            "description": "html only: vertical scroll position 0..100 (% of page height). Default 0."},
                                    "html_zoom": {"type": "number",
                                                    "description": "html only: zoom factor (e.g. 1.2 zooms in). Default 1.0."},
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


def _validate_plan(plan: dict, available_assets: dict) -> str:
    """Returns 'OK; ...' on pass, 'ERROR: ...' on fail.

    available_assets shape:
      {
        "recording":  {"source_path": str, "duration_s": float, "width": int, "height": int} | None,
        "hyperframes": [{"source_path": str, "duration_s": float, ...}, ...],
        "html_pages":  [{"source_path": str}, ...],
      }
    """
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

    rec = available_assets.get("recording")
    rec_dur = float(rec["duration_s"]) if rec else 0.0
    hyperframes_by_path: dict[str, dict] = {
        h["source_path"]: h for h in (available_assets.get("hyperframes") or [])
    }
    html_paths: set[str] = {
        h["source_path"] for h in (available_assets.get("html_pages") or [])
    }

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
        if bg_type not in ("color", "gradient", "recording", "hyperframe", "html"):
            return f"ERROR: scenes[{i}].background.type must be color|gradient|recording|hyperframe|html"

        # ── recording rules
        if bg_type == "recording":
            if not rec:
                return f"ERROR: scenes[{i}] uses recording bg but no recording asset is available"
            start = bg.get("start_in_source_s")
            if start is None:
                return f"ERROR: scenes[{i}] background.recording missing start_in_source_s"
            if start < 5.0:
                return (f"ERROR: scenes[{i}].background.start_in_source_s={start} "
                        f"violates R3 (must be ≥ 5.0; head 5s of recording is unstable)")
            end = start + dur
            tail_limit = rec_dur - 5.0
            if end > tail_limit:
                return (f"ERROR: scenes[{i}] recording_clip ends at {end:.2f}s but "
                        f"recording is only {rec_dur:.2f}s (R3: must end by "
                        f"{tail_limit:.2f}s). Shorten duration or pick earlier start.")
            if not bg.get("source_path"):
                return f"ERROR: scenes[{i}] background.recording missing source_path"
            if bg["source_path"] != rec["source_path"]:
                return (f"ERROR: scenes[{i}].background.source_path={bg['source_path']!r} "
                        f"doesn't match the available recording {rec['source_path']!r}")

        # ── hyperframe rules
        if bg_type == "hyperframe":
            sp = bg.get("source_path")
            if not sp:
                return f"ERROR: scenes[{i}] background.hyperframe missing source_path"
            if sp not in hyperframes_by_path:
                return (f"ERROR: scenes[{i}].background.source_path={sp!r} not in "
                        f"available_assets.hyperframes ({list(hyperframes_by_path)}).")
            start = float(bg.get("start_in_source_s") or 0)
            if start < 0:
                return f"ERROR: scenes[{i}] hyperframe start_in_source_s={start} must be ≥ 0"
            hyper_dur = float(hyperframes_by_path[sp]["duration_s"])
            if start + dur > hyper_dur + 0.05:
                return (f"ERROR: scenes[{i}] hyperframe slice [{start:.2f},{start+dur:.2f}] "
                        f"exceeds clip length {hyper_dur:.2f}s")

        # ── html rules
        if bg_type == "html":
            sp = bg.get("source_path")
            if not sp:
                return f"ERROR: scenes[{i}] background.html missing source_path"
            if sp not in html_paths:
                return (f"ERROR: scenes[{i}].background.source_path={sp!r} not in "
                        f"available_assets.html_pages ({sorted(html_paths)}).")

        # ── R4: legibility
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
                return (f"ERROR: scenes[{i}] title-style text on solid color — R4: "
                        f"use recording/hyperframe (darken 0.65-0.85) or html.")
            if (is_body_style and bg_type in ("recording", "hyperframe")
                and (s.get("darken") or 0) < 0.6):
                return (f"ERROR: scenes[{i}] body-style text over busy bg "
                        f"({bg_type}) with darken={s.get('darken')} — R4: "
                        f"use color/gradient/html, or set darken ≥ 0.7.")

    # R5 transitions
    transitions = plan.get("transitions") or []
    for t in transitions:
        if t.get("kind") == "crossfade":
            dur = t.get("duration_s", 0)
            if dur < 0.4:
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


def _build_initial_message(project_brief: str, available_assets: dict,
                           feedback: Optional[str] = None,
                           previous_plan: Optional[dict] = None) -> str:
    parts = [
        "## Project brief (TONE/AUDIENCE only — not a content checklist)",
        "```markdown",
        project_brief.strip()[:3000],
        "```",
        "",
        "## Available assets — these are everything you can use as a background",
    ]
    rec = available_assets.get("recording")
    if rec:
        parts.append("### `recording` (Demo Driver capture of the project actually running)")
        parts.append(f"- source_path: {rec['source_path']!r}  (use exactly this string)")
        parts.append(f"- duration_s:  {rec['duration_s']:.2f}")
        parts.append(f"- size:        {rec['width']}x{rec['height']}")
        if rec.get("codec"):
            parts.append(f"- codec:       {rec['codec']}")
        parts.append("- R3: must skip first 5s and last 5s.")
        parts.append("")
    else:
        parts.append("### `recording` — none available (project had no demo recording).")
        parts.append("")

    hyperframes = available_assets.get("hyperframes") or []
    if hyperframes:
        parts.append("### `hyperframe` clips (OpenDesigner motion_film polished animations)")
        for h in hyperframes:
            parts.append(f"- source_path: {h['source_path']!r}  duration_s: {h['duration_s']:.2f}  size: {h.get('width', '?')}x{h.get('height', '?')}")
            if h.get("notes"):
                parts.append(f"    notes: {h['notes']}")
        parts.append("- No head/tail skip rule — these are clean.")
        parts.append("")
    else:
        parts.append("### `hyperframe` — none available.")
        parts.append("")

    html_pages = available_assets.get("html_pages") or []
    if html_pages:
        parts.append("### `html` pages (OpenDesigner static_hero designed pages)")
        for h in html_pages:
            parts.append(f"- source_path: {h['source_path']!r}")
            if h.get("title"):
                parts.append(f"    title: {h['title']!r}")
        parts.append("- Optional html_scroll_y_pct (0..100) and html_zoom (default 1.0).")
        parts.append("")
    else:
        parts.append("### `html` — none available.")
        parts.append("")

    if available_assets.get("captions_path"):
        parts.append("## Demo captions")
        parts.append(f"You can `read_file({available_assets['captions_path']!r})` to see "
                     f"the bilingual captions the Demo Driver tagged during recording — "
                     f"each entry has `t` (seconds into recording), `zh`, `en`, `importance`.")
        parts.append("")

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
        parts.append("Produce a revised cutting_plan that addresses the feedback.")
    else:
        parts.append("Begin: 30–45s, 5–8 scenes. YOU decide the recording-vs-hyperframe-vs-html "
                     "mix based on what each asset is good for and what this project needs to "
                     "communicate. Call `emit_cutting_plan` when done.")
    return "\n".join(parts)


@traced_agent("Agent 3 RemotionComposer · plan", phase=3)
def run_cutting_planner(run_dir: Path,
                        project_brief: str,
                        available_assets: dict,
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
    rec_dur = (available_assets.get("recording") or {}).get("duration_s", 0)
    n_hyper = len(available_assets.get("hyperframes") or [])
    n_html = len(available_assets.get("html_pages") or [])
    log.info(f"start  recording_dur={rec_dur:.1f}s  hyperframes={n_hyper}  "
             f"html_pages={n_html}  model={model}")

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
        "content": _build_initial_message(project_brief, available_assets,
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
                    out = _validate_plan(tu.input, available_assets)
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
