"""Agent 6b · OpenDesign Critic.

A senior visual director that looks at OpenDesign's output, rates it, and
either passes it through to the user OR sends a concrete revision_prompt
back to OpenDesign and waits for a new version. Repeat until pass (or a
soft iteration cap is hit).

IRON LAW: the critic decides pass/fail. The loop has a soft cap to bound
cost — when hit, the latest artifact is surfaced to the user with the
critic's final verdict; we do NOT force a failure. User can resume the
loop with "再改一轮" or a hint.
"""
from __future__ import annotations

import base64
import io
import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger

from ..observability.audit import get_run_context, traced_agent, traced_step
from ..observability.logger import agent_logger
from ..tools.ffbin import ffmpeg, ffprobe
from ..tools.llm import anthropic_client, model_for
from .opendesigner import (
    OpenDesignSession,
    adopt as od_adopt,
    iterate_stream,
    load_session,
)


# ───────────────────────────────────────────────────────────────
# Progress + result data
# ───────────────────────────────────────────────────────────────

def _progress_path(run_dir: Path) -> Path:
    return run_dir / "od_critic_progress.json"


@dataclass
class CritiqueVerdict:
    """One critic pass over an artifact."""
    verdict: str               # "pass" | "fail"
    score: int                 # 1-10, critic's own number
    issues: list[str]          # bullet points; non-empty when fail
    revision_prompt: str       # OpenCode-ready instruction when fail; "" on pass
    overall: str               # 1-2 sentence summary the user reads
    iteration: int             # 1-indexed
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    artifact_path: str = ""    # relative to run_dir
    artifact_kind: str = ""    # "mp4" | "html"


# ───────────────────────────────────────────────────────────────
# Critic system prompt
# ───────────────────────────────────────────────────────────────

_CRITIC_SYSTEM = """\
You are a senior art director reviewing an AI-generated visual asset for a
product promo video pipeline. The asset is either a ~30s motion-film MP4
(rendered from HyperFrames HTML+GSAP) or a static HTML hero page.

You will receive:
  - The project brief (tone, audience, visual keywords).
  - A sequence of frames sampled at 1 fps (mp4) OR a full-page screenshot
    (html). Frames are in temporal order.
  - Optionally, the previous iterations' verdicts and the latest user hint.

══════════════════════════════════════════════════════════════════════
FORMAT — VERTICAL / DOUYIN / TIKTOK / 抖音 (NON-NEGOTIABLE)
══════════════════════════════════════════════════════════════════════
The output target is a vertical short-form video — Douyin / TikTok /
小红书 / YouTube Shorts. **Canvas = 1080×1920 (9:16 portrait), not
1920×1080**. If the frames in front of you are landscape, that's an
automatic fail R0 below (regardless of how pretty the content is).

A portrait promo lives or dies on the FIRST 2 SECONDS — viewers swipe
in 800ms if it doesn't grab. The composition must respect mobile
viewing: thumb-area bottom 15%, status-area top 10%, headlines in the
high-middle.

══════════════════════════════════════════════════════════════════════
PASSING BAR — CONCRETE REQUIREMENTS A SHORT-FORM PROMO MUST MEET
══════════════════════════════════════════════════════════════════════
A passing video (verdict=pass) MUST satisfy ALL of these. Anything less
is a fail. Check each one against the actual frames in front of you:

  R0 · VERTICAL FORMAT (9:16 portrait)
     Every frame's aspect ratio must be ~9:16 (1080×1920 typical).
     Landscape 16:9 frames or square 1:1 frames are an automatic
     R0 fail with score ≤ 2 regardless of content quality.

  R0.5 · MOBILE-LEGIBLE COMPOSITION
     Headlines big enough to read on a phone (≥ 80px equivalent at
     1080×1920). Critical content in the center 70% vertical band
     (not buried in the top 10% status zone or bottom 15% thumb
     zone). No horizontal split layouts pretending the canvas is
     wide.

  R1 · IDENTIFY THE PRODUCT (within first 4 seconds)
     The product's actual name AND a one-line tagline are visible.
     If frame 0-4 doesn't tell a stranger what product this is,
     it fails R1.

  R2 · SHOW THE PRODUCT, NOT METAPHORS
     At least 2 frames must depict the actual product surface — a
     literal screenshot/mockup/UI element showing what the user uses.
     A pitch diagram with 2 dots, abstract orbs, particle systems,
     phone-shaped black rectangles, or "code rain" are NOT showing
     the product. They are decoration. Decoration without product =
     fail R2.

  R3 · ONE CLEAR USP DRIVING THE NARRATIVE
     The brief lists 独特卖点 / unique selling points. Pick ONE — the
     most visual one — and let it drive the 30s. Don't try to cover
     all 6. A video showing 6 USPs with 5 seconds each is incoherent;
     a video showing 1 USP for 25s with hook+demo+payoff is a promo.

  R4 · LEGIBLE COPY
     Every visible text element must be readable at a 360p YouTube
     thumbnail size. If a frame has 6+ lines of micro-text in a
     dashboard mockup, viewers can't read it — that's "fake content".
     Pull big numbers, big quotes, single concepts.

  R5 · MOTION WITH PURPOSE (mp4 only)
     Every scene transition must be motivated (zoom into the spot you
     care about; cut on a beat). Cross-dissolves that just bridge two
     unrelated frames are filler. Random particles drifting aren't
     motion — they're noise.

  R6 · DOES NOT REGRESS FROM PRIOR ITERATIONS
     If prior_iterations[].issues mention "phone mockup blank", and
     THIS iteration's phone mockup is still blank (or worse, replaced
     with a black rectangle), that is a CRITICAL regression — fail
     hard with score ≤2. Improvements must be incremental and
     additive, not lateral or worse.

══════════════════════════════════════════════════════════════════════
CRITIQUE STYLE — WHAT TO FLAG (in order of severity)
══════════════════════════════════════════════════════════════════════
Critique LIKE A DEMANDING DESIGN DIRECTOR. Be specific. No empty praise.

  Tier 1 sins (auto-fail score ≤ 3)
    - AI slop tells: generic gradient "tech" backgrounds, stock "code
      rain", "neural network blobs", floating particles, 2010-era
      Tron-grid/glow-scanline cliché.
    - Placeholder content: lorem ipsum, "...", "TBD", micro-text that's
      illegible, empty dashboards.
    - Product invisible (R1 / R2 failure).
    - Regression from prior iter (R6 failure).

  Tier 2 sins (score 3-5)
    - Layout drift, no alignment grid, every frame centered for safety.
    - Typography hierarchy unclear; display/body/mono indistinguishable.
    - Color palette muddy or clashing; visual keywords from brief absent.
    - Motion: default linear easing, unmotivated transitions, kinetic
      type that just appears.

  Tier 3 sins (score 5-7)
    - Pacing slightly off, one or two beats dragging.
    - Minor copy issues, single illegible element.
    - One transition feels forced.

  Pass-grade (8+)
    - All R1-R6 met. Tier 1 absent. Tier 2 mostly absent. A real
      designer would ship this for client review.

══════════════════════════════════════════════════════════════════════
revision_prompt — HOW TO WRITE THE FIX INSTRUCTION
══════════════════════════════════════════════════════════════════════
The revision_prompt is sent to an OpenCode CLI editing HyperFrames HTML.
OpenCode is dumb if overloaded — it tries to satisfy everything you ask
at once and produces incoherent output when given 10 demands.

RULES:
  • Pick the TOP 2-3 issues (the worst Tier 1 + Tier 2 from this iter).
    Ignore the rest. Lesser issues will surface on the next round if
    they still matter after the big ones are fixed.
  • State each fix as a SPECIFIC action — what to add, where, with
    what value. Never abstract: "make it more elegant" is useless.
    Say: "Replace the abstract orb in scene 2 with a 1080×720 mockup
    of the product's dashboard view (use HTML+CSS to render real-
    looking metric tiles: '+87%' / '124k Users' / heat-map grid)".
  • PRESERVE what's working — explicitly tell OpenCode what to KEEP
    from the current version so it doesn't throw away progress.
    "Keep the dark navy background, the JetBrains Mono typography,
    and the 'Goalcast' wordmark in scene 1 — fix only scenes 3 & 5."
  • Each fix builds on prior — don't restart the design from scratch
    every iteration. If iter 2 introduced a good color palette, iter
    3's revision must NOT change colors unless palette itself was the
    issue.
  • Be terse. 600-1500 chars total. Long instructions = OpenCode
    fries.

══════════════════════════════════════════════════════════════════════
OUTPUT — STRICTLY a single JSON object inside a ```json fenced block,
no extra prose anywhere.

{
  "verdict": "pass" | "fail",
  "score": 1-10,
  "overall": "one sentence summary the user reads",
  "issues": ["specific issue 1", "specific issue 2", ...],
  "revision_prompt": "TOP 2-3 fixes + what to preserve, in OpenCode-\
ready prose. 600-1500 chars. Empty string if verdict=pass."
}

Use verdict=pass ONLY if R1-R6 all met. Don't pass to be polite. If a
real designer would hand it back, fail it.
"""


# ───────────────────────────────────────────────────────────────
# Frame / screenshot extraction
# ───────────────────────────────────────────────────────────────

def _probe_duration(mp4_path: Path) -> float:
    r = subprocess.run(
        [ffprobe(), "-v", "quiet", "-print_format", "json",
         "-show_format", str(mp4_path)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return float(json.loads(r.stdout)["format"]["duration"])


def _extract_mp4_frames(mp4_path: Path, out_dir: Path,
                       fps: float = 1.0, max_frames: int = 30,
                       width: int = 480) -> list[Path]:
    """Sample frames at `fps` per second, downscale to `width`px wide as JPEG,
    return ordered paths. Cap at `max_frames`.

    Defaults tuned for vision-LLM context: 480px wide × JPEG q=60 keeps each
    frame ~30-60KB so 30 frames stay under ~2MB total in the request body.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in list(out_dir.glob("kf_*.png")) + list(out_dir.glob("kf_*.jpg")):
        try: f.unlink()
        except: pass
    pattern = out_dir / "kf_%04d.jpg"
    cmd = [
        ffmpeg(), "-y", "-i", str(mp4_path),
        "-vf", f"fps={fps},scale={width}:-2",
        "-vframes", str(max_frames),
        "-q:v", "5",  # JPEG q ~ 0(best)..31(worst); 5 ≈ q60
        str(pattern),
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    return sorted(out_dir.glob("kf_*.jpg"))


def _screenshot_html(html_path: Path, out_path: Path,
                    viewport_w: int = 1440, viewport_h: int = 900) -> Path:
    """Use a headless browser to render the HTML as a full-page screenshot."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError("playwright not installed — cannot screenshot HTML")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": viewport_w, "height": viewport_h})
        page = ctx.new_page()
        # file:// URI
        page.goto(html_path.absolute().as_uri(), wait_until="networkidle", timeout=30000)
        page.screenshot(path=str(out_path), full_page=True)
        browser.close()
    return out_path


def _png_to_b64(p: Path, max_kb: int = 500) -> str:
    """Read png, optionally re-encode to fit max_kb, return base64."""
    data = p.read_bytes()
    # crude size check — ARK image input has a per-image cap
    if len(data) > max_kb * 1024:
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            buf = io.BytesIO()
            # Re-encode at progressively lower quality until under cap.
            q = 85
            while q >= 40:
                buf.seek(0); buf.truncate(0)
                img.convert("RGB").save(buf, format="JPEG", quality=q, optimize=True)
                if buf.tell() <= max_kb * 1024:
                    return base64.b64encode(buf.getvalue()).decode("ascii")
                q -= 10
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass
    return base64.b64encode(data).decode("ascii")


def _img_block(p: Path) -> dict:
    """Build an Anthropic-compatible image content block from a png path."""
    # Determine media type from suffix or assume png after our re-encode
    suffix = p.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        media = "image/jpeg"
    else:
        media = "image/png"
    # If we re-encoded to JPEG inside _png_to_b64, the returned b64 IS jpeg
    # bytes even if the source path is .png. Detect via magic header.
    data = base64.b64decode(_png_to_b64(p))
    if data[:3] == b"\xff\xd8\xff":
        media = "image/jpeg"
    else:
        media = "image/png"
    return {"type": "image", "source": {
        "type": "base64", "media_type": media,
        "data": base64.b64encode(data).decode("ascii"),
    }}


# ───────────────────────────────────────────────────────────────
# Vision call
# ───────────────────────────────────────────────────────────────

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_verdict_json(text: str) -> dict:
    m = _FENCE.search(text)
    blob = m.group(1) if m else text
    return json.loads(blob)


def _call_critic_llm(brief: str, images: list[Path], artifact_kind: str,
                     prior_iterations: list[CritiqueVerdict],
                     user_hint: Optional[str], log) -> CritiqueVerdict:
    client = anthropic_client()
    model = model_for("vision")  # kimi-k2.6 — ARK image-capable
    log.info(f"critic LLM model={model} kind={artifact_kind} images={len(images)} "
             f"prior_iters={len(prior_iterations)} hint={'Y' if user_hint else 'N'}")

    parts: list[str] = [
        f"=== project_brief (excerpt) ===\n{brief[:3000]}",
        "",
        f"=== artifact kind: {artifact_kind} ===",
        (f"=== frames sampled at 1 fps, in temporal order: {len(images)} frames ==="
         if artifact_kind == "mp4"
         else "=== full-page HTML screenshot follows ==="),
    ]
    if prior_iterations:
        parts.append("")
        parts.append("=== prior critic iterations (for context — do not repeat already-fixed issues) ===")
        for v in prior_iterations[-3:]:  # last 3
            parts.append(
                f"  iter#{v.iteration} verdict={v.verdict} score={v.score}\n"
                f"    issues: {' | '.join(v.issues[:5])}\n"
                f"    sent revision: {v.revision_prompt[:200]}{'…' if len(v.revision_prompt) > 200 else ''}"
            )
    if user_hint:
        parts.append("")
        parts.append("=== user hint (must inform your critique + revision_prompt) ===")
        parts.append(user_hint)

    content: list[dict] = [{"type": "text", "text": "\n".join(parts)}]
    for img in images:
        content.append(_img_block(img))

    # thinking={"type":"disabled"} — kimi-k2.6 burns the max_tokens budget on
    # hidden reasoning otherwise. Same fix as setup_runner/quality_judge.
    # max_tokens=8000 — critic outputs grow long by iter 3/4 (3 verdicts of
    # context + 10 detailed issues + multi-paragraph revision_prompt). At
    # 2000 the JSON gets truncated mid-string. 8000 leaves comfortable headroom.
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        thinking={"type": "disabled"},
        system=_CRITIC_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    log.info(f"critic LLM response ({len(text)}B) stop={resp.stop_reason}: {text[:500]}{'…' if len(text) > 500 else ''}")
    try:
        data = _parse_verdict_json(text)
    except Exception as e:
        log.exception(f"critic JSON parse failed: {e}")
        # IMPORTANT: when critic itself errors, surface fail with EMPTY
        # revision_prompt so the outer loop bails to "needs_user" instead
        # of sending nonsense back to OpenDesign.
        return CritiqueVerdict(
            verdict="fail", score=0,
            issues=[f"critic itself failed to produce structured output: {type(e).__name__}: {e} "
                    f"(stop_reason={resp.stop_reason}, in={resp.usage.input_tokens}, "
                    f"out={resp.usage.output_tokens}, raw_head={text[:200]!r})"],
            revision_prompt="",  # empty => outer loop surfaces to user
            overall="Critic agent could not produce a verdict — review the artifact manually.",
            iteration=0,
        )
    return CritiqueVerdict(
        verdict=str(data.get("verdict", "fail")).lower(),
        score=int(data.get("score", 0)),
        issues=list(data.get("issues") or []),
        revision_prompt=str(data.get("revision_prompt", "") or ""),
        overall=str(data.get("overall", "") or "").strip(),
        iteration=0,  # filled by caller
    )


# ───────────────────────────────────────────────────────────────
# Artifact + adopt
# ───────────────────────────────────────────────────────────────

def _find_artifact(run_dir: Path) -> tuple[str, Path]:
    """Return (kind, path) of the latest adopted OpenDesign artifact.
    Prefers hero/intro.mp4 (motion_film) over html_asset/*.html (static_hero)."""
    hero_mp4 = run_dir / "hero" / "intro.mp4"
    if hero_mp4.exists():
        return ("mp4", hero_mp4)
    html_dir = run_dir / "html_asset"
    if html_dir.exists():
        htmls = sorted(html_dir.glob("*.html"))
        if htmls:
            return ("html", htmls[0])
    raise RuntimeError(
        "no OpenDesign artifact adopted yet — "
        "expected hero/intro.mp4 OR html_asset/*.html"
    )


def _ensure_adopted(run_dir: Path, log) -> tuple[str, Path]:
    """If no artifact yet, call adopt() once. Returns (kind, path)."""
    try:
        return _find_artifact(run_dir)
    except RuntimeError:
        pass
    log.info("no artifact on disk yet, calling adopt(as_role='hero')")
    od_adopt(run_dir, as_role="hero")
    return _find_artifact(run_dir)


# ───────────────────────────────────────────────────────────────
# Progress persistence
# ───────────────────────────────────────────────────────────────

def _write_progress(run_dir: Path, payload: dict) -> None:
    p = _progress_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"last_update": datetime.now(timezone.utc).isoformat(), **payload}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                 encoding="utf-8")


def read_progress(run_dir: Path) -> Optional[dict]:
    p = _progress_path(run_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ───────────────────────────────────────────────────────────────
# Main loop
# ───────────────────────────────────────────────────────────────

@traced_agent("Agent 6b · OpenDesign Critic Loop", phase=2)
def run_critic_loop(run_dir: Path,
                    soft_max_iterations: int = 5,
                    user_hint: Optional[str] = None) -> dict:
    """Run critic ↔ OpenDesign loop until critic passes OR soft cap hit.

    Returns a summary dict the caller can use to surface to the UI:
      {
        "status": "passed" | "soft_cap" | "error",
        "iterations": [<CritiqueVerdict dicts>],
        "final_artifact": str,         # rel path
        "final_artifact_kind": str,
        "elapsed_s": float,
      }
    """
    log = agent_logger("agent6b_critic")
    sess = load_session(run_dir)
    if sess is None:
        raise RuntimeError("no OpenDesign session — bootstrap first")
    brief = sess.brief

    iterations: list[CritiqueVerdict] = []
    frames_dir = run_dir / "od_critic_frames"
    t0 = time.monotonic()
    final_status = "soft_cap"  # default if we exit via cap

    _write_progress(run_dir, {
        "stage": "starting",
        "soft_max_iterations": soft_max_iterations,
        "user_hint": user_hint,
        "iterations": [],
    })

    for i in range(1, soft_max_iterations + 1):
        log.info(f"=== critic iteration {i}/{soft_max_iterations} ===")
        _write_progress(run_dir, {
            "stage": f"adopt+critique",
            "current_iteration": i,
            "soft_max_iterations": soft_max_iterations,
            "user_hint": user_hint,
            "iterations": [asdict(v) for v in iterations],
        })

        # 1. Make sure we have the latest artifact on disk
        try:
            kind, art_path = _ensure_adopted(run_dir, log)
        except Exception as e:
            log.exception(f"adopt failed at iter {i}: {e}")
            _write_progress(run_dir, {
                "stage": "error",
                "error": f"adopt failed: {type(e).__name__}: {e}",
                "iterations": [asdict(v) for v in iterations],
            })
            final_status = "error"
            break

        # 2. Sample frames / screenshot
        if kind == "mp4":
            try:
                dur = _probe_duration(art_path)
                # 1 fps default, cap at 60 frames for context-window sanity
                images = _extract_mp4_frames(art_path, frames_dir / f"iter_{i}",
                                              fps=1.0, max_frames=60)
                log.info(f"extracted {len(images)} keyframes (mp4 dur={dur:.1f}s)")
            except Exception as e:
                log.exception(f"frame extract failed: {e}")
                images = []
        else:  # html
            try:
                shot = _screenshot_html(art_path, frames_dir / f"iter_{i}" / "page.png")
                images = [shot]
                log.info(f"screenshotted HTML page: {shot}")
            except Exception as e:
                log.exception(f"html screenshot failed: {e}")
                images = []

        if not images:
            log.warning(f"no images captured at iter {i}; bailing")
            iterations.append(CritiqueVerdict(
                verdict="fail", score=0, iteration=i,
                issues=["could not extract frames / screenshot the artifact"],
                revision_prompt="",
                overall="Critic could not see the artifact — please retry or inspect manually.",
                artifact_path=str(art_path.relative_to(run_dir)).replace("\\", "/"),
                artifact_kind=kind,
            ))
            final_status = "error"
            break

        # 3. Call vision critic
        v = _call_critic_llm(brief, images, kind, iterations,
                              user_hint if i == 1 else None, log)
        v.iteration = i
        v.artifact_path = str(art_path.relative_to(run_dir)).replace("\\", "/")
        v.artifact_kind = kind
        iterations.append(v)
        log.info(f"iter {i} verdict={v.verdict} score={v.score}  overall={v.overall[:80]}")

        _write_progress(run_dir, {
            "stage": "verdict_in",
            "current_iteration": i,
            "soft_max_iterations": soft_max_iterations,
            "iterations": [asdict(x) for x in iterations],
        })

        if v.verdict == "pass":
            final_status = "passed"
            log.info(f"critic passed at iter {i}")
            break

        if not v.revision_prompt.strip():
            log.warning(f"iter {i} failed but produced empty revision_prompt — "
                       "treating as 'critic unable to articulate fix', surface to user")
            final_status = "needs_user"
            break

        # If this was the last allowed iteration, don't bother retrying —
        # surface to user with the fail verdict + revision_prompt visible.
        if i >= soft_max_iterations:
            log.info(f"iter {i} == soft_max; not sending revision, surfacing to user")
            final_status = "soft_cap"
            break

        # 4. Send revision_prompt back to OpenDesign and consume its SSE
        log.info(f"sending revision back to OpenDesign (len={len(v.revision_prompt)}B)")
        _write_progress(run_dir, {
            "stage": f"opendesign_retrying",
            "current_iteration": i,
            "soft_max_iterations": soft_max_iterations,
            "iterations": [asdict(x) for x in iterations],
            "revision_in_flight": v.revision_prompt[:400],
        })
        try:
            n_evt = 0
            for evt in iterate_stream(run_dir, user_message=v.revision_prompt,
                                       raw_user_feedback=v.revision_prompt,
                                       timeout_s=1500.0):
                n_evt += 1
            log.info(f"opendesign retry done, {n_evt} SSE events")
        except Exception as e:
            log.exception(f"opendesign retry failed at iter {i}: {e}")
            _write_progress(run_dir, {
                "stage": "error",
                "error": f"opendesign retry failed: {type(e).__name__}: {e}",
                "iterations": [asdict(x) for x in iterations],
            })
            final_status = "error"
            break

        # 5. Re-adopt the new artifact for next iter
        try:
            od_adopt(run_dir, as_role="hero")
        except Exception as e:
            log.warning(f"re-adopt after retry failed: {e}")

    final_kind: str = ""
    final_art: str = ""
    try:
        k, p = _find_artifact(run_dir)
        final_kind = k
        final_art = str(p.relative_to(run_dir)).replace("\\", "/")
    except Exception:
        pass

    summary = {
        "status": final_status,
        "iterations": [asdict(v) for v in iterations],
        "final_artifact": final_art,
        "final_artifact_kind": final_kind,
        "elapsed_s": round(time.monotonic() - t0, 1),
        "user_hint": user_hint,
        "soft_max_iterations": soft_max_iterations,
    }
    _write_progress(run_dir, {"stage": "done", **summary})
    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="agent6b_critic",
                 name="od_critic_loop", path=str(_progress_path(run_dir)),
                 status=final_status, n_iterations=len(iterations),
                 final_score=iterations[-1].score if iterations else 0)
    return summary
