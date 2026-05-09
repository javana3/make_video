"""M5 Step 1 · VoiceOver agent — propose voiceover_script.json from brief + plan.

Single-shot LLM call: input = project_brief + cutting_plan scenes (with
durations), output = list of voice cues with t_start/t_end aligned to scenes.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger
from ..tools.llm import anthropic_client, model_for


SYSTEM_PROMPT = """\
You are a voiceover scriptwriter for a 30-45 second product promo video.

You receive:
- project_brief.md describing the product
- cutting_plan.json listing scenes with durations and on-screen text/visuals

Your job is to propose a voiceover script in two languages (zh-CN and en-US):
- Each cue MUST align to scene boundaries (use scene t_start / t_end)
- Not every scene needs voice — leave purely visual scenes silent
- Cues must FIT the on-screen time: short, punchy, never crammed
- Voice should COMPLEMENT visuals, not duplicate the on-screen text verbatim
- Tone: hype, confident, sports-broadcast energy. No marketing fluff.

Output a single JSON object:
{
  "zh-CN": [
    {"id": "S1", "t_start": 0.0, "t_end": 3.0, "text": "...", "lang": "zh-CN"},
    ...
  ],
  "en-US": [
    {"id": "S1", "t_start": 0.0, "t_end": 3.0, "text": "...", "lang": "en-US"},
    ...
  ]
}

Rules:
- t_start / t_end are floats (seconds, like 5.0 not "5s")
- Each text ≤ 60 chars zh, ≤ 90 chars en (roughly 1 sec speech per 12 chars zh / 15 chars en)
- Maximum 6 cues per language
- IDs match cutting_plan scene IDs (S1, S2, ... or "outro" / "intro")
- Output ONLY the JSON object inside a fenced code block, no commentary"""


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict:
    m = _FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))
    # Fallback: try to parse the whole thing
    try:
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"could not extract JSON from LLM output: {e}\n--text--\n{text[:500]}")


def _build_scene_summary(plan: dict) -> str:
    lines = ["Scene timeline (cumulative t_start computed from durations):"]
    t = 0.0
    for sc in plan.get("scenes", []):
        sid = sc.get("id", "?")
        d = float(sc.get("duration_s", 0))
        bg = sc.get("background", {}).get("type", "?")
        texts = [el.get("content", "") for el in sc.get("elements", [])
                 if el.get("type") == "text"]
        text_summary = " | ".join(texts) if texts else "(no text)"
        lines.append(f"  {sid}: t={t:.1f}s..{t+d:.1f}s  d={d:.1f}s  bg={bg}  text={text_summary[:80]}")
        t += d
    lines.append(f"Total video duration: {t:.1f}s")
    return "\n".join(lines)


@traced_agent("Agent 5 Voice · Step1 Script", phase=5)
def propose_script(run_dir: Path,
                   project_brief: str,
                   cutting_plan: dict,
                   output_path: Path,
                   feedback: Optional[str] = None) -> Path:
    """Run single-shot LLM to write voiceover_script_bilingual.json.

    Also writes per-language files: voiceover_script_zh-CN.json, voiceover_script_en-US.json
    """
    log = agent_logger("agent5_voice")
    client = anthropic_client()
    model = model_for("reasoning")

    started = datetime.now(timezone.utc)
    log.info(f"start  model={model}  feedback={'yes' if feedback else 'no'}")

    scene_summary = _build_scene_summary(cutting_plan)
    user_msg = (
        f"=== project_brief.md ===\n{project_brief}\n\n"
        f"=== cutting_plan summary ===\n{scene_summary}\n"
    )
    if feedback:
        user_msg += f"\n=== USER FEEDBACK on previous draft ===\n{feedback}\n"

    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    log.info(f"  ← stop_reason={resp.stop_reason} in={resp.usage.input_tokens} out={resp.usage.output_tokens}")

    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    bilingual = _extract_json(text)
    if "zh-CN" not in bilingual or "en-US" not in bilingual:
        raise ValueError(f"LLM output missing language keys: {list(bilingual.keys())}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bilingual, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    # Per-language files for direct edge-tts feeding
    for lang, entries in bilingual.items():
        per = output_path.parent / f"voiceover_script_{lang}.json"
        # ensure each entry has lang field set
        for e in entries:
            e.setdefault("lang", lang)
        per.write_text(json.dumps(entries, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        log.info(f"  ✓ {per.name}  {len(entries)} cues")

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    log.info(f"done  output={output_path.name} elapsed={elapsed:.1f}s")
    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="agent5_voice",
                 name="voiceover_script", path=str(output_path),
                 elapsed_s=round(elapsed, 1),
                 langs=list(bilingual.keys()))
    return output_path
