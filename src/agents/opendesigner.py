"""M2a · Agent 6 OpenDesigner.

Sits between the User (via our Web UI tab) and the Open Design daemon.
Open Design already has its own opinionated LLM workflow (discovery form,
direction picker, 31 skills, 138 design systems, five-axis critique). We
don't reinvent any of that. Agent 6's job is small but specific:

  1. Bootstrap: pick skill + design system + initial prompt from project_brief.md
     (single LLM call to translate brief into Open Design vocabulary).
  2. Iterate: forward user feedback to OpenDesign on the SAME conversation,
     so the underlying coding-agent CLI does incremental edits to the same
     index.html (vs. regenerating from scratch).
  3. Adopt: pull archive → unzip into run_dir/html_asset/ to close M2a.

State is persisted to run_dir/opendesign/state.json so the Web UI (which
makes stateless HTTP calls per turn) can resume mid-conversation.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger
from ..tools.llm import anthropic_client, model_for
from ..tools.opendesign_client import (
    OpenDesignEndpoint,
    create_project,
    download_archive,
    list_design_systems,
    list_skills,
    pick_available_agent,
    read_artifact_bytes,
    send_prompt_stream,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class OpenDesignSession:
    """Persistent state for one OpenDesign collaboration."""
    web_url: str
    daemon_url: str
    project_id: str
    conversation_id: str
    agent_id: str           # opencode / claude / cursor-agent
    skill_id: Optional[str]
    design_system_id: Optional[str]
    project_name: str
    initial_prompt: str = ""  # Agent's first-pass prompt — auto-sent in turn 1
    brief: str = ""           # full project_brief.md text — context for LLM translation
    mode: str = "static_hero" # "motion_film" → hyperframes (.mp4), "static_hero" → web-prototype (.html)
    history: list[dict] = field(default_factory=list)  # turns
    adopted_path: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "OpenDesignSession":
        return cls(**json.loads(text))


def _state_path(run_dir: Path) -> Path:
    return run_dir / "opendesign" / "state.json"


def load_session(run_dir: Path) -> Optional[OpenDesignSession]:
    p = _state_path(run_dir)
    if not p.exists():
        return None
    return OpenDesignSession.from_json(p.read_text(encoding="utf-8"))


def save_session(run_dir: Path, sess: OpenDesignSession) -> None:
    p = _state_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(sess.to_json(), encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM-assisted picking
# ---------------------------------------------------------------------------

_BOOTSTRAP_SYSTEM = """\
You are a visual director picking the right Open Design skill + opening prompt
for a product promo video pipeline. The pipeline's final deliverable is ALWAYS
a video — your job is to plan the visual asset Open Design should produce.

═══ DEFAULT MODE · MOTION FILM (use this almost always) ═══
Target skill: **`hyperframes`** — Open Design's HTML→MP4 video framework
(GSAP timeline + data-* attributes + 60fps frame capture; supports TTS,
audio-reactive visuals, captions, scene transitions, kinetic typography).
Output: a real .mp4 file in the project root.

This is the right choice for almost every brief in this pipeline — the
user's downstream UI lets them choose whether the resulting .mp4 becomes
the FINAL video or a HERO segment that Phase 3 Remotion will composite
into a longer cut.

For hyperframes, design_system_id can be any system (hyperframes uses it
as a base palette). Default to a low-saturation tech DS — `vercel`,
`linear`, `apple`, `github` — unless the brief's visual_keywords
explicitly call for editorial/warm/playful.

`initial_prompt` for hyperframes MUST encode a video brief:
   - duration: prefer 15-25s — Douyin/TikTok-vertical pacing rewards
     fast cuts and a sub-30s total; use 30s only if the product
     genuinely needs the extra time. 45s is too long for this format.
   - scene arc (3-6 named scenes with one-line beat each, drawn from
     the product's 独特卖点 or core features — never invent new copy)
   - palette (2-3 hex colors max, anchored to visual_keywords)
   - typography (one display font + one mono/body) — sized for mobile
     viewing: headlines ≥ 80px, body ≥ 40px at 1080x1920 canvas
   - motion style (restrained / explosive / cinematic — pick one)
   - **hard constraints (NON-NEGOTIABLE — VERTICAL FORMAT)**:
     * **canvas 1080×1920** (portrait, 9:16) — this is the Douyin /
       TikTok / 抖音 / 小红书 / YouTube Shorts native format. NOT
       1920×1080. Every scene composition must work in 9:16.
     * Compose for the **center 70% column** — viewers' thumbs cover
       the bottom 15% (action bar) and the top 10% (status/comments).
       Put the killer headline at ~25-35% from top; the call-to-action
       at ~75-85% from top.
     * **No tiny side-by-side layouts** — there is no horizontal space
       for two columns. Stack vertically.
     * No external image assets, no audio unless brief specifically
       asks for narration.
     * The first 2 seconds must hook a swipe-happy viewer — bold
       statement frame, no slow fade-in from black for 2 full seconds.

═══ FALLBACK MODE · STATIC HERO (rare — only if brief explicitly says so) ═══
Target skill: `web-prototype` / `magazine-poster` / `social-carousel`.
Use ONLY when the brief explicitly asks for a still page (e.g. "we just
need a landing page mock", "no animation"). Do not use this just because
the product is a SaaS — the OUTPUT is video, not the product.

═══ Output format ═══
STRICTLY a JSON object inside a ```json fenced block, no prose:
{
  "skill_id": "...",
  "design_system_id": "...",
  "initial_prompt": "...",
  "mode": "motion_film" | "static_hero"
}

The `mode` field must match: skill=hyperframes → motion_film; otherwise
static_hero."""


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict:
    m = _FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))
    return json.loads(text)


def llm_pick_setup(daemon_url: str, brief: str,
                    run_dir: Optional[Path] = None) -> dict:
    """One LLM call → {skill_id, design_system_id, initial_prompt}."""
    log = agent_logger("agent6_opendesigner")
    skills = list_skills(daemon_url)
    dss = list_design_systems(daemon_url)

    skill_summary = "\n".join(
        f"  {s.get('id','?'):28s} mode={s.get('mode','?'):10s} {(s.get('summary') or '')[:80]}"
        for s in skills[:60]
    )
    ds_summary = "\n".join(
        f"  {d.get('id','?'):20s} {(d.get('summary') or d.get('title') or '')[:80]}"
        for d in dss[:60]
    )
    user_msg = (
        f"=== project_brief.md ===\n{brief}\n\n"
        f"=== AVAILABLE SKILLS (first 60 of {len(skills)}) ===\n{skill_summary}\n\n"
        f"=== AVAILABLE DESIGN SYSTEMS (first 60 of {len(dss)}) ===\n{ds_summary}"
    )

    client = anthropic_client()
    model = model_for("reasoning")
    log.info(f"LLM pick_setup model={model} skills={len(skills)} ds={len(dss)}")
    # thinking={"type":"disabled"} — glm-5.1 otherwise burns max_tokens on
    # hidden reasoning and emits empty text, breaking _extract_json. Same
    # fix as quality_judge.py / setup_runner.py.
    # max_tokens=2400 — the JSON includes `initial_prompt` (a multi-paragraph
    # video brief that easily runs 1000+ tokens). At 900 it gets truncated mid
    # string and json.loads fails with "Expecting ',' delimiter".
    resp = client.messages.create(
        model=model,
        max_tokens=2400,
        thinking={"type": "disabled"},
        system=_BOOTSTRAP_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    log.info(f"LLM pick_setup response ({len(text)}B, stop={resp.stop_reason}, "
             f"in={resp.usage.input_tokens}, out={resp.usage.output_tokens})")
    try:
        data = _extract_json(text)
    except Exception as e:
        # Save the raw output for forensic debugging if a run_dir is known.
        if run_dir is not None:
            raw_path = run_dir / "opendesign" / "bootstrap_llm_raw.txt"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(text, encoding="utf-8")
        raise RuntimeError(
            f"LLM pick_setup returned non-JSON: {type(e).__name__}: {e}. "
            f"stop_reason={resp.stop_reason}, "
            f"out_tokens={resp.usage.output_tokens}. "
            f"Try increasing max_tokens or relaxing the schema."
        ) from e
    # Validate ids actually exist; fall back to safe defaults if LLM hallucinated
    valid_skill_ids = {s["id"] for s in skills}
    valid_ds_ids = {d["id"] for d in dss}
    if data.get("skill_id") not in valid_skill_ids:
        log.warning(f"LLM picked invalid skill {data.get('skill_id')!r}, falling back to web-prototype")
        data["skill_id"] = "web-prototype" if "web-prototype" in valid_skill_ids else next(iter(valid_skill_ids))
    if data.get("design_system_id") not in valid_ds_ids:
        log.warning(f"LLM picked invalid DS {data.get('design_system_id')!r}, falling back to vercel")
        data["design_system_id"] = "vercel" if "vercel" in valid_ds_ids else next(iter(valid_ds_ids))
    if not isinstance(data.get("initial_prompt"), str) or not data["initial_prompt"].strip():
        raise ValueError(f"LLM returned no valid initial_prompt: {data}")
    # Infer mode from skill if LLM forgot to set it (or set wrong)
    expected_mode = "motion_film" if data["skill_id"] == "hyperframes" else "static_hero"
    if data.get("mode") not in ("motion_film", "static_hero"):
        data["mode"] = expected_mode
    elif data["mode"] != expected_mode:
        log.warning(f"LLM mode={data['mode']!r} mismatches skill {data['skill_id']!r}; correcting to {expected_mode!r}")
        data["mode"] = expected_mode
    log.info(f"picked: skill={data['skill_id']} ds={data['design_system_id']} mode={data['mode']}")
    return data


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

@traced_agent("Agent 6 OpenDesigner · Bootstrap", phase=2)
def bootstrap(run_dir: Path,
              endpoint: OpenDesignEndpoint,
              brief: str,
              project_name: str,
              preferred_agents: Optional[list[str]] = None) -> OpenDesignSession:
    """Create project + send first prompt. Returns session AFTER first turn finishes."""
    log = agent_logger("agent6_opendesigner")
    agent_id = pick_available_agent(endpoint.daemon_url,
                                     preferred_agents or ["opencode", "claude", "cursor-agent"])
    setup = llm_pick_setup(endpoint.daemon_url, brief, run_dir=run_dir)
    log.info(f"setup picked: skill={setup['skill_id']} ds={setup['design_system_id']}")

    proj = create_project(
        endpoint.daemon_url, name=project_name,
        skill_id=setup["skill_id"],
        design_system_id=setup["design_system_id"],
    )
    sess = OpenDesignSession(
        web_url=endpoint.web_url, daemon_url=endpoint.daemon_url,
        project_id=proj["project_id"], conversation_id=proj["conversation_id"],
        agent_id=agent_id,
        skill_id=setup["skill_id"], design_system_id=setup["design_system_id"],
        project_name=project_name,
        initial_prompt=setup["initial_prompt"],
        brief=brief,
        mode=setup.get("mode", "static_hero"),
    )
    save_session(run_dir, sess)
    log.info(f"project created: {sess.project_id}")
    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="agent6_opendesigner",
                 name="opendesign_session", path=str(run_dir / "opendesign" / "state.json"),
                 project_id=sess.project_id, skill_id=sess.skill_id,
                 design_system_id=sess.design_system_id)
    return sess


_TRANSLATE_SYSTEM_STATIC = """\
You are a visual director translating one round of user feedback into a precise
prompt for a coding agent (OpenCode) editing a single index.html via Open
Design's skill framework. The current project is a STATIC HTML hero page.

You receive:
- The product brief (for unchangeable copy/constraints).
- The list of past turns (what the user has asked so far + their feedback).
- The user's latest natural-language feedback.

Your output is the next prompt sent to OpenCode. It MUST:
- Be specific about EXACT changes (CSS values, copy lines, layout positions),
  never vague. "字太大" → "reduce headline font-size from 96px to 72px";
  "金黑配色" → "background #0a0a0a, accent #d4af37 on borders + headlines".
- Reference index.html as the file to edit. Incremental edits, NOT rewrite.
- Forbid external assets (images, CDN fonts not already in index.html).
- Keep it under ~6 lines. Plain prose. No markdown headers."""


_TRANSLATE_SYSTEM_MOTION = """\
You are a visual director translating one round of user feedback into a precise
prompt for a coding agent (OpenCode) editing a HyperFrames composition (Open
Design's HTML→MP4 video framework). The current project is a MOTION FILM (an
.mp4 generated from HTML+GSAP timeline + data-* timing attributes).

You receive:
- The product brief.
- Past turns (user feedback + what was sent).
- The user's latest natural-language feedback.

Your output is the next prompt sent to OpenCode. It MUST:
- Translate temporal feedback into specific timeline edits. "节奏太快" →
  "stretch the S2_features sprite from 4s to 6s; ease the title entry with
  power3.out instead of expo.out". "字停留太短" → "extend hold phase of the
  headline tween by 0.8s before exit fade".
- Translate visual feedback into GSAP/CSS specifics. "金黑配色" → "swap
  --canvas to #0a0a0a, --accent to #d4af37, regenerate the gradient at
  scene S1's `data-bg` style".
- Reference the HyperFrames composition file in `.hyperframes-cache/<slot>/
  index.html`. OpenCode should edit ONLY that file (not files in project
  root) and re-dispatch render via OD daemon (`media generate --surface
  video --model hyperframes-html --composition-dir <slot>`).
- Forbid adding new external assets. The render must stay at
  1080×1920 (vertical 9:16, Douyin/TikTok format) / 30fps unless
  user asked otherwise.
- Keep it under ~8 lines. Plain prose."""


def translate_feedback(sess: OpenDesignSession, user_feedback: str) -> str:
    """One LLM call: user natural-language feedback → precise OpenCode prompt.

    System prompt branches on `sess.mode` so motion_film feedback gets
    translated to GSAP/timeline edits, not CSS edits.
    """
    log = agent_logger("agent6_opendesigner")
    history_summary = "\n".join(
        f"  Turn {t['turn_index']}: {t['user_message'][:200]}"
        for t in sess.history
    ) if sess.history else "  (none)"

    user_msg = (
        f"=== brief ===\n{sess.brief[:2000]}\n\n"
        f"=== past turns ===\n{history_summary}\n\n"
        f"=== user feedback (latest) ===\n{user_feedback}"
    )
    system_prompt = (
        _TRANSLATE_SYSTEM_MOTION if sess.mode == "motion_film" else _TRANSLATE_SYSTEM_STATIC
    )
    client = anthropic_client()
    model = model_for("reasoning")
    log.info(f"translate_feedback mode={sess.mode} model={model} feedback_len={len(user_feedback)} history={len(sess.history)}")
    # max_tokens=3000 — translated prompts can be long when user feedback is
    # detailed (or when feedback comes from the critic with revision_prompt
    # already 2k+ chars). 600 used to truncate.
    resp = client.messages.create(
        model=model,
        max_tokens=3000,
        thinking={"type": "disabled"},
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    if not text:
        raise ValueError("LLM returned empty translation")
    return text


def iterate_stream(run_dir: Path,
                   user_message: Optional[str] = None,
                   raw_user_feedback: Optional[str] = None,
                   timeout_s: float = 1200.0) -> Iterator[dict]:
    """Forward one turn to OpenDesign. Yields SSE events.

    Three modes:
      1. user_message="..."  — send literal text (advanced/escape hatch)
      2. raw_user_feedback="..." — Agent translates → send (normal user feedback)
      3. (none) — first turn: use sess.initial_prompt verbatim (Agent's first pass)

    Generator: can't use @traced_agent (span would close before yields).
    Span + events handled inline.
    """
    from ..observability.tracer import get_tracer
    sess = load_session(run_dir)
    if sess is None:
        raise RuntimeError("no OpenDesign session — call bootstrap first")
    log = agent_logger("agent6_opendesigner")
    bus = get_run_context().get("event_bus")
    tracer = get_tracer("video-workflow")
    span_name = f"Agent 6 OpenDesigner · Turn {len(sess.history) + 1}"
    if bus is not None:
        bus.emit("agent_start", agent="agent6_opendesigner", phase=2,
                 turn_index=len(sess.history) + 1)

    is_first_turn = len(sess.history) == 0
    translation_summary = None

    if user_message:
        prompt = user_message
        log.info(f"iterate (literal) project={sess.project_id} msg_len={len(prompt)}")
    elif raw_user_feedback:
        translation_summary = {
            "raw_feedback": raw_user_feedback,
        }
        # Surface translation start so the UI can show "Agent translating..."
        yield {"event": "agent.translate.start", "data": {"raw": raw_user_feedback}}
        prompt = translate_feedback(sess, raw_user_feedback)
        translation_summary["translated_prompt"] = prompt
        yield {"event": "agent.translate.done", "data": {"prompt": prompt}}
        log.info(f"iterate (translated) project={sess.project_id} prompt_len={len(prompt)}")
    elif is_first_turn and sess.initial_prompt:
        prompt = sess.initial_prompt
        log.info(f"iterate (first turn / agent-driven) project={sess.project_id} prompt_len={len(prompt)}")
    else:
        raise ValueError("iterate needs user_message OR raw_user_feedback OR (first-turn + initial_prompt)")

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    last_event = {}
    n_events = 0
    for evt in send_prompt_stream(
        sess.daemon_url, sess.project_id, sess.conversation_id,
        prompt,
        agent_id=sess.agent_id,
        skill_id=sess.skill_id,
        design_system_id=sess.design_system_id,
        timeout=timeout_s,
    ):
        n_events += 1
        last_event = evt
        yield evt
        if evt.get("event") in {"end", "error"}:
            break

    finished_at = datetime.now(timezone.utc).isoformat()
    elapsed = time.monotonic() - t0
    sess.history.append({
        "turn_index": len(sess.history) + 1,
        "raw_feedback": raw_user_feedback,         # what user typed (or None for first turn)
        "user_message": prompt,                     # what was actually sent (translated or literal)
        "is_first_turn": is_first_turn,
        "translated": bool(raw_user_feedback),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_s": round(elapsed, 1),
        "n_events": n_events,
        "final_status": (last_event.get("data") or {}).get("status") if isinstance(last_event.get("data"), dict) else None,
    })
    save_session(run_dir, sess)
    log.info(f"iterate done in {elapsed:.1f}s, {n_events} events, status={sess.history[-1]['final_status']}")
    if bus is not None:
        bus.emit("agent_done", agent="agent6_opendesigner", phase=2,
                 turn_index=len(sess.history),
                 elapsed_s=round(elapsed, 1),
                 n_events=n_events,
                 final_status=sess.history[-1].get("final_status"))


def read_html(run_dir: Path) -> bytes:
    sess = load_session(run_dir)
    if sess is None:
        raise RuntimeError("no OpenDesign session — call bootstrap first")
    return read_artifact_bytes(sess.daemon_url, sess.project_id, "index.html")


def list_artifacts(run_dir: Path) -> dict:
    """Probe daemon's project file list. Returns {primary_kind, primary_name,
    files: [{name, size, kind}]}.

    primary_kind ∈ {"video", "html", "none"} drives Web UI preview mode.
    For motion_film mode the user cares about the .mp4; for static_hero the
    user cares about index.html.
    """
    sess = load_session(run_dir)
    if sess is None:
        return {"primary_kind": "none", "primary_name": None, "files": []}
    from ..tools.opendesign_client import list_project_files
    try:
        files = list_project_files(sess.daemon_url, sess.project_id)
    except Exception:
        return {"primary_kind": "none", "primary_name": None, "files": []}
    # Filter out cache-only / hidden files
    visible = [f for f in files if not f.get("name", "").startswith(".")]
    # Pick primary: in motion mode prefer first .mp4; in static prefer index.html
    if sess.mode == "motion_film":
        mp4 = next((f for f in visible if f.get("name", "").lower().endswith(".mp4")), None)
        if mp4:
            return {"primary_kind": "video", "primary_name": mp4["name"], "files": visible}
    html = next((f for f in visible if f.get("name", "").lower() == "index.html"), None)
    if html:
        return {"primary_kind": "html", "primary_name": "index.html", "files": visible}
    # Fallback: any mp4 then any html
    mp4 = next((f for f in visible if f.get("name", "").lower().endswith(".mp4")), None)
    if mp4:
        return {"primary_kind": "video", "primary_name": mp4["name"], "files": visible}
    if visible:
        f = visible[0]
        kind = "video" if f.get("name", "").lower().endswith(".mp4") else "html"
        return {"primary_kind": kind, "primary_name": f["name"], "files": visible}
    return {"primary_kind": "none", "primary_name": None, "files": []}


@traced_agent("Agent 6 OpenDesigner · Adopt", phase=2)
def adopt(run_dir: Path, as_role: str = "auto") -> dict:
    """Pull artifact and route to right destination per `as_role`.

    Roles:
      - "auto"  : route by session.mode (motion_film → final, static_hero → hero)
      - "hero"  : land at run_dir/hero/intro.<ext> + run_dir/html_asset/ (mp4 → hero,
                  html → html_asset). Phase 3 cutting_plan can then embed it.
      - "final" : land .mp4 at run_dir/outputs/final.mp4 (skip Phase 3-5).

    For motion_film mode (mp4 output), we ALSO pull index.html + cache files
    via download_archive into run_dir/opendesign_artifacts/ for audit/replay.
    """
    log = agent_logger("agent6_opendesigner")
    sess = load_session(run_dir)
    if sess is None:
        raise RuntimeError("no OpenDesign session — call bootstrap first")

    if as_role == "auto":
        as_role = "final" if sess.mode == "motion_film" else "hero"
    if as_role not in ("hero", "final"):
        raise ValueError(f"as_role must be 'hero'|'final'|'auto', got {as_role!r}")

    info = list_artifacts(run_dir)
    primary_kind = info["primary_kind"]
    primary_name = info["primary_name"]
    if primary_name is None:
        raise RuntimeError("no artifact yet on daemon — wait for OpenCode to finish")

    log.info(f"adopt as={as_role}  primary={primary_kind}/{primary_name}  mode={sess.mode}")

    import shutil
    out: dict = {"as_role": as_role, "primary_kind": primary_kind, "primary_name": primary_name}

    # 1. Always archive everything for audit
    audit_dir = run_dir / "opendesign_artifacts"
    if audit_dir.exists():
        shutil.rmtree(audit_dir)
    archive = download_archive(sess.daemon_url, sess.project_id, audit_dir)
    out["audit_dir"] = str(audit_dir)
    out["audit_files"] = archive["n_files"]

    # 2. Place primary artifact at its routing destination
    if primary_kind == "video":
        primary_bytes = read_artifact_bytes(sess.daemon_url, sess.project_id, primary_name)
        if as_role == "hero":
            target = run_dir / "hero" / "intro.mp4"
        else:  # final
            target = run_dir / "outputs" / "final.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(primary_bytes)
        out["primary_target"] = str(target)
        out["primary_bytes"] = len(primary_bytes)
    elif primary_kind == "html":
        # static hero: land HTML into html_asset/ regardless of role label
        html_target = run_dir / "html_asset"
        if html_target.exists():
            shutil.rmtree(html_target)
        html_result = download_archive(sess.daemon_url, sess.project_id, html_target)
        out["primary_target"] = str(html_target)
        out["primary_files"] = html_result["n_files"]
    else:
        raise RuntimeError(f"unknown primary_kind={primary_kind!r}")

    sess.adopted_path = out["primary_target"]
    save_session(run_dir, sess)
    log.info(f"adopt done: {out['primary_target']} ({out.get('primary_bytes', out.get('primary_files'))})")

    bus = get_run_context().get("event_bus")
    if bus is not None:
        # Use specific event names so downstream Phases can branch on them
        if as_role == "hero":
            event_name = "hero_video" if primary_kind == "video" else "html_asset"
        else:
            event_name = "final_video"
        bus.emit("asset_verified", agent="agent6_opendesigner",
                 name=event_name, path=out["primary_target"],
                 as_role=as_role, primary_kind=primary_kind,
                 audit_dir=out["audit_dir"])
    return out
