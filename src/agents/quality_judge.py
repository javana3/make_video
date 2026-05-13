"""QualityJudge — LLM-as-a-Judge for each phase's agent output.

Triggered automatically after each phase completes:
  - Phase 1 brief (project_brief.md)
  - Phase 2a setup_plan (setup_plan.json)
  - Phase 3 cutting_plan (cutting_plan.json)
  - Phase 5 voiceover_script (voiceover_script.json)

Phase 4 BGM is deterministic (MusicGen/MiniMax music-2.6), NOT an LLM
output — skipped. The final video v1_bgm_voice_final.mp4 is the user's job
to rate manually via the UI (1-5 stars).

Each judge call:
  - Reads the artifact + project context
  - Single LLM call (no tool use loop) returning a structured score
  - Saves to <run_dir>/scores.jsonl (source of truth)

The judge itself emits an OTEL span (via @traced_agent) so it shows up
in Phoenix UI alongside the agent it's judging, but the score values
live in scores.jsonl + the project's /scores page — Phoenix has no
simple "attach score to most-recent trace by name" endpoint.

Model default: LLM_REASONING (glm-5.1). User can override per-phase via
the standard prompt override mechanism (key = quality_judge_<phase>).
"""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..observability.audit import traced_agent
from ..observability.logger import agent_logger
from ..tools.llm import anthropic_client, model_for
from ..tools.score_log import save_local_score


# Per-phase criteria. Each phase has its own rubric.
_RUBRICS = {
    "brief": {
        "criteria": [
            "precision        — claims grounded in actual repo content, not hallucinated",
            "completeness     — positioning / audience / 独特卖点 / 视觉关键词 / 竞品 all present and substantive",
            "code_alignment   — 独特卖点 phrases verifiably reflect what's in the source",
            "tone_match       — tone fits a 30s promo video (concise, confident)",
            "actionable       — gives RemotionComposer enough to design 6 scenes",
        ],
        "artifact_path": "project_brief.md",
        "artifact_kind": "markdown",
    },
    "setup_plan": {
        "criteria": [
            "completeness     — install / config / services all covered for this stack",
            "correctness      — commands actually match the project's documented setup",
            "auto_install     — uses winget/brew/apt etc. for missing system tools rather than punting",
            "config_coverage  — every *.example.* config file has a matching config_writes entry",
            "credential_use   — uses parent ${ARK_KEY_1} etc. instead of asking user for LLM keys",
            "user_secrets_minimal — user_secrets_needed only declared when truly necessary",
        ],
        "artifact_path": "setup_plan.json",
        "artifact_kind": "json",
    },
    "cutting_plan": {
        "criteria": [
            "scene_coherence  — 5-8 scenes form a narrative arc (hook / pitch / proof / CTA)",
            "duration_target  — total 30-45s",
            "asset_mix        — recording / hyperframe / html mix justified by content",
            "darken_legibility — text overlays on busy bg use darken 0.65-0.85",
            "caption_grounding — overlay text reflects what's actually shown, not made up",
            "transition       — crossfades between scenes (R5 rule)",
        ],
        "artifact_path": "cutting_plan.json",
        "artifact_kind": "json",
    },
    "voiceover_script": {
        "criteria": [
            "scene_alignment  — each cue's t_start/t_end matches a cutting_plan scene boundary",
            "bilingual_quality — zh-CN and en-US both natural, no machine-translation feel",
            "duration_fit     — each cue fits the on-screen window (≈12 zh chars/sec, 15 en/sec)",
            "complement_not_repeat — voice complements visuals rather than reading captions aloud",
            "hype_appropriate — energy matches a 30s promo (confident, not flat)",
        ],
        "artifact_path": "voice/voiceover_script_bilingual.json",
        "artifact_kind": "json",
    },
    "demo_captions": {
        "criteria": [
            "accuracy        — caption text matches what's happening on-screen at that timestamp",
            "pacing          — captions appear at a natural reading rhythm, not crammed or sparse",
            "informativeness — captions explain WHY this UI step matters, not just describe pixels",
            "narrative_arc   — captions form a coherent demo story (hook → action → payoff)",
            "concision       — each caption stays short enough to read in its on-screen window",
        ],
        "artifact_path": "demo_captions.jsonl",
        "artifact_kind": "jsonl",
    },
    "voiceover_zh": {
        "criteria": [
            "naturalness    — sounds like a native Chinese speaker, not translated",
            "rhythm         — each cue reads naturally in ≈12 字/秒 with breath room",
            "tone_promo     — confident promo tone, no flat narration",
            "term_choice    — technical terms localized correctly (e.g., 框架 vs framework)",
            "duration_fit   — each cue's character count fits its t_end - t_start window",
        ],
        "artifact_path": "voice/voiceover_script_zh-CN.json",
        "artifact_kind": "json",
    },
    "voiceover_en": {
        "criteria": [
            "naturalness    — reads like a native English copywriter, not translated-from-Chinese",
            "rhythm         — each cue at ≈15 chars/sec with natural cadence",
            "tone_promo     — confident promo voice; avoid corporate-speak",
            "term_choice    — technical terms idiomatic (e.g., 'one-click' vs '一键')",
            "duration_fit   — each cue's word count fits its t_end - t_start window",
        ],
        "artifact_path": "voice/voiceover_script_en-US.json",
        "artifact_kind": "json",
    },
}


# Cross-phase consistency judge — separate from per-phase rubrics. Reads
# ALL prior artifacts and rates how well the project's narrative survives
# the brief → setup → cutting_plan → voiceover chain. This is what catches
# "Phase 1 promised feature X but Phase 5 voiceover never mentions it".
_CROSS_PHASE_CRITERIA = [
    "core_message      — brief's top 3 unique selling points each appear (named or paraphrased) in cutting_plan scene text AND voiceover_script",
    "visual_keywords   — brief's visual keywords (color/style/mood) reflected in cutting_plan scene asset choices + on-screen text styling",
    "audience_fit      — brief's target audience drives cutting_plan tone + voiceover language complexity (technical vs accessible)",
    "factuality_chain  — stack/tools auto-detected in setup_plan appear accurately in cutting_plan captions (no invented tech)",
    "scope_coherence   — cutting_plan total_duration ≈ 30-45s, voiceover cue count aligns with scene count, no scenes left silent",
    "competitive_diff  — brief's '独特卖点' / vs-competitor framing surfaces in at least one voiceover cue or scene caption",
]


SYSTEM_PROMPT_TEMPLATE = """You are QualityJudge — an LLM-as-a-Judge evaluating one
phase of a promo-video pipeline.

Phase under review: **{phase}**
Artifact: **{artifact_path}**

You will receive:
1. The full project brief (so you know what this project IS)
2. The artifact's full content

Your job: evaluate the artifact against this rubric:

{rubric_text}

Score EACH criterion 1-10. Then compute an overall 1-10 score
(weighted average — explain the weights in your comment).

OUTPUT FORMAT (single JSON object inside ```json fence):
{{
  "overall": 7.5,
  "breakdown": {{
    "<criterion_short_name>": 8,
    "<criterion_short_name>": 7,
    ...
  }},
  "comment": "Brief 2-3 sentence summary of strengths and weaknesses.",
  "concrete_issues": [
    {{"severity": "high|medium|low", "text": "specific issue with quoted evidence"}}
  ]
}}

Be specific, not generic. Quote actual text from the artifact when calling out issues.
Don't be a pushover — if it's mediocre, score 5-6, not 7-8. Don't be cruel either —
if it's solid for a 30s promo, score 7-9.
"""


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict:
    m = _JSON_FENCE.search(text)
    raw = m.group(1) if m else text
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find any top-level JSON object as fallback
        m2 = re.search(r"\{[\s\S]*\}", text)
        if m2:
            return json.loads(m2.group(0))
        raise


@traced_agent("Quality Judge", phase=0)
def score_phase(phase: str, run_dir: Path) -> Optional[dict]:
    """Run the judge on one phase's output. Returns the score dict (or None on hard failure).

    Phase must be one of: brief / setup_plan / cutting_plan / voiceover_script.
    """
    log = agent_logger("quality_judge")
    if phase not in _RUBRICS:
        log.error(f"unknown phase {phase!r}")
        return None
    rubric = _RUBRICS[phase]

    artifact_path = run_dir / rubric["artifact_path"]
    if not artifact_path.exists():
        log.warning(f"artifact missing: {artifact_path}; skipping judge")
        return None

    brief_path = run_dir / "project_brief.md"
    project_brief = brief_path.read_text(encoding="utf-8") if brief_path.exists() else "(no brief)"
    artifact_text = artifact_path.read_text(encoding="utf-8")

    rubric_text = "\n".join(f"  - {c}" for c in rubric["criteria"])
    system = SYSTEM_PROMPT_TEMPLATE.format(
        phase=phase, artifact_path=rubric["artifact_path"], rubric_text=rubric_text,
    )
    user_msg = (
        f"=== project_brief.md ===\n{project_brief[:4000]}\n\n"
        f"=== {rubric['artifact_path']} ===\n{artifact_text[:8000]}\n"
    )

    client = anthropic_client()
    model = model_for("reasoning")
    log.info(f"start  phase={phase}  model={model}  artifact={rubric['artifact_path']}")

    # Use the recovery wrapper so failed judge calls also escalate sensibly.
    # `thinking={"type": "disabled"}` — official Anthropic SDK extended-thinking
    # parameter; ARK passes it through to glm-5.1. Without this, glm-5.1 spends
    # the entire max_tokens budget on hidden reasoning and emits no text block.
    # Refs:
    #   https://www.volcengine.com/docs/82379/1956279
    #   https://docs.z.ai/guides/llm/glm-5.1
    from .error_agent import llm_call_with_recovery
    try:
        resp = llm_call_with_recovery(
            lambda: client.messages.create(
                model=model, max_tokens=2048, system=system,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": user_msg}],
            ),
            run_dir=run_dir,
            agent="quality_judge",
            step_label=f"score {phase}",
            context_hint={"phase": phase, "artifact": rubric["artifact_path"], "model": model},
            log=log,
        )
    except Exception as e:
        log.exception(f"judge LLM call exhausted: {e}")
        return None

    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    try:
        data = _extract_json(text)
    except Exception as e:
        log.exception(f"judge output not parseable: {e}  raw={text[:500]}")
        return None

    overall = float(data.get("overall", 0))
    breakdown = data.get("breakdown") or {}
    comment = data.get("comment") or ""
    issues = data.get("concrete_issues") or []

    log.info(f"phase={phase}  overall={overall}  breakdown={breakdown}")

    # Save local
    record = {
        "phase": phase,
        "artifact_path": rubric["artifact_path"],
        "model": model,
        "overall": overall,
        "breakdown": breakdown,
        "comment": comment,
        "concrete_issues": issues,
        "source": "auto_judge",
    }
    save_local_score(run_dir, record)
    log.info(f"saved score → scores.jsonl (overall={overall}/10)")
    return record


@traced_agent("Quality Judge · cross-phase consistency", phase=0)
def score_cross_phase_consistency(run_dir: Path) -> Optional[dict]:
    """Cross-phase consistency judge.

    Reads ALL prior artifacts (brief / setup_plan / cutting_plan / voiceover)
    in one LLM call and rates how well the project's narrative survives
    the chain. This catches "Phase 1 promised feature X but Phase 5
    voiceover never mentions it" — the kind of drop-the-ball that per-phase
    rubrics miss because each only sees its own artifact.

    Should be fired AFTER all 4 source artifacts exist (post Phase 5).
    """
    log = agent_logger("quality_judge")

    brief_p = run_dir / "project_brief.md"
    setup_p = run_dir / "setup_plan.json"
    cutting_p = run_dir / "cutting_plan.json"
    voice_p = run_dir / "voice" / "voiceover_script_bilingual.json"

    missing = [str(p.name) for p in (brief_p, setup_p, cutting_p, voice_p) if not p.exists()]
    if missing:
        log.warning(f"cross-phase judge skipped; missing artifacts: {missing}")
        return None

    rubric_text = "\n".join(f"  - {c}" for c in _CROSS_PHASE_CRITERIA)
    system = (
        "You are QualityJudge evaluating CROSS-PHASE consistency of a promo-video "
        "pipeline. You'll see 4 artifacts produced by 4 different agents: a "
        "project brief, a setup plan, a cutting (video edit) plan, and a "
        "bilingual voiceover script. Your job is to rate how well the project's "
        "narrative is preserved across the chain — does what Phase 1 promised "
        "actually surface in the final video plan?\n\n"
        f"Rubric (each 1-10):\n{rubric_text}\n\n"
        "OUTPUT FORMAT (single JSON object inside ```json fence):\n"
        "{\n"
        '  "overall": 7.5,\n'
        '  "breakdown": {"core_message": 8, "visual_keywords": 7, ...},\n'
        '  "comment": "2-3 sentence summary",\n'
        '  "concrete_issues": [{"severity": "high|medium|low", "text": "with quoted evidence from artifacts"}]\n'
        "}\n\n"
        "Cite the actual artifact text when calling out drift. Be specific, "
        "not generic."
    )

    user_msg = (
        f"=== project_brief.md ===\n{brief_p.read_text(encoding='utf-8')[:3500]}\n\n"
        f"=== setup_plan.json ===\n{setup_p.read_text(encoding='utf-8')[:2500]}\n\n"
        f"=== cutting_plan.json ===\n{cutting_p.read_text(encoding='utf-8')[:3500]}\n\n"
        f"=== voiceover_script_bilingual.json ===\n{voice_p.read_text(encoding='utf-8')[:2500]}\n"
    )

    client = anthropic_client()
    model = model_for("reasoning")
    log.info(f"start  cross-phase  model={model}")

    from .error_agent import llm_call_with_recovery
    try:
        resp = llm_call_with_recovery(
            lambda: client.messages.create(
                model=model, max_tokens=2048, system=system,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": user_msg}],
            ),
            run_dir=run_dir, agent="quality_judge",
            step_label="score cross_phase",
            context_hint={"phase": "cross_phase", "model": model},
            log=log,
        )
    except Exception as e:
        log.exception(f"cross-phase judge LLM call exhausted: {e}")
        return None

    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    try:
        data = _extract_json(text)
    except Exception as e:
        log.exception(f"cross-phase output not parseable: {e}  raw={text[:500]}")
        return None

    record = {
        "phase": "cross_phase_consistency",
        "artifact_path": "(brief + setup_plan + cutting_plan + voiceover)",
        "model": model,
        "overall": float(data.get("overall", 0)),
        "breakdown": data.get("breakdown") or {},
        "comment": data.get("comment") or "",
        "concrete_issues": data.get("concrete_issues") or [],
        "source": "auto_judge",
    }
    save_local_score(run_dir, record)
    log.info(f"cross-phase saved → scores.jsonl (overall={record['overall']}/10)")
    return record


@traced_agent("Final Video Rating · user", phase=0)
def record_user_video_rating(run_dir: Path, rating: float,
                                 comment: str = "") -> dict:
    """User's 1-5 star rating of the final video.

    Called from the web UI when the user clicks a star. Distinct from
    auto_judge — source='user'. Saved to scores.jsonl (the /scores page
    is the canonical viewer).
    """
    log = agent_logger("quality_judge")
    if not (1.0 <= rating <= 5.0):
        raise ValueError(f"rating must be 1-5, got {rating}")

    record = {
        "phase": "final_video",
        "rating_1_to_5": rating,
        "comment": comment,
        "source": "user",
    }
    save_local_score(run_dir, record)
    log.info(f"user rating={rating}/5 saved → scores.jsonl")
    return record
