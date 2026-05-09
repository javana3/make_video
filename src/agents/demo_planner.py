"""Phase 2B-demo · LLM Demo Planner.

Before recording, this agent:
  1. Opens the running app URL with Playwright (headless).
  2. Captures a DOM/accessibility-tree snapshot (semantic, not just HTML).
  3. Asks the LLM to design a 25-30s demonstration script — a list of
     concrete user actions (navigate / click / type / scroll / hover / wait)
     each with a paired bilingual caption.
  4. Validates selectors against the captured DOM where possible.

Output: `demo_script.json` consumed by `src/tools/demo_executor.py` to
drive the actual recording (with synced caption timing).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from loguru import logger

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger
from ..tools.llm import anthropic_client, model_for


# ---------------------------------------------------------------------------
# DOM snapshot
# ---------------------------------------------------------------------------

def capture_dom_snapshot(url: str,
                          width: int = 1920,
                          height: int = 1080,
                          wait_load_s: float = 3.0,
                          max_chars: int = 12000) -> dict:
    """Open `url` headless and return {a11y_tree, html_tail, screenshot_b64}.

    The accessibility tree is much smaller than raw HTML and tells the LLM
    what's clickable / typeable / readable. Raw HTML tail is a fallback.
    """
    log = agent_logger("demo_planner")
    log.info(f"DOM snapshot: {url}  viewport={width}x{height}")
    from playwright.sync_api import sync_playwright

    out: dict = {"url": url}
    with sync_playwright() as p:
        # Bypass system VPN proxy for localhost (same fix as opendesign_client)
        browser = p.chromium.launch(headless=True, args=["--no-proxy-server"])
        ctx = browser.new_context(viewport={"width": width, "height": height})
        page = ctx.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_timeout(int(wait_load_s * 1000))
            out["title"] = page.title()
            # Accessibility tree (semantic structure)
            try:
                a11y = page.accessibility.snapshot(interesting_only=True) or {}
            except Exception as e:
                log.warning(f"a11y snapshot failed: {e}")
                a11y = {}
            out["a11y_tree"] = _shrink(a11y, max_chars=max_chars)
            # Visible text fallback
            try:
                visible_text = page.evaluate(
                    "() => document.body.innerText"
                ) or ""
            except Exception:
                visible_text = ""
            out["visible_text_tail"] = visible_text[:max_chars]
            # Quick interactable element catalog (selectors LLM can use)
            try:
                catalog = page.evaluate("""() => {
                  const out = [];
                  document.querySelectorAll('button, a, input, textarea, select, [role=button], [role=link], [role=tab]').forEach((el, i) => {
                    if (i > 100) return;
                    const tag = el.tagName.toLowerCase();
                    const role = el.getAttribute('role') || '';
                    const txt = (el.innerText || el.value || el.placeholder || '').trim().slice(0, 80);
                    const id  = el.id || '';
                    const cls = (el.className || '').toString().slice(0, 60);
                    out.push({ i, tag, role, txt, id, cls });
                  });
                  return out;
                }""")
            except Exception:
                catalog = []
            out["interactables"] = catalog[:80]
        finally:
            browser.close()
    log.info(f"snapshot: title={out.get('title')!r} interactables={len(out.get('interactables', []))}")
    return out


def _shrink(obj, max_chars: int = 12000) -> str:
    """Serialize JSON-ish object trimmed to max_chars (a11y trees can be huge)."""
    s = json.dumps(obj, ensure_ascii=False)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 30] + "...<truncated>"


# ---------------------------------------------------------------------------
# LLM planning
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a product demo director. You will be given:
  - The product brief (project_brief.md).
  - A live snapshot of the running app: title, accessibility tree, visible
    text, and a catalog of interactable elements with their tag/role/text/id/class.
  - A target duration for the demo (typically 25-30 seconds).

Your task: design a demonstration script — an ordered list of concrete user
actions that show this product's value to a viewer in the target window.

OUTPUT a strict JSON object inside a ```json fence:
{
  "duration_s": <number>,            // target total duration
  "viewport": {"width": 1920, "height": 1080},
  "steps": [
    {
      "id": "S1",
      "action": "navigate" | "click" | "type" | "scroll" | "hover" | "wait" | "press_key",
      "target": <string>,             // for click/type/hover: a CSS selector OR Playwright "text=..." OR "role=button[name='X']"
      "url": <string>,                // for navigate
      "text": <string>,               // for type
      "key": <string>,                // for press_key (e.g. "Enter")
      "x": <number>,                  // for scroll: pixels
      "wait_after_s": <number>,       // post-action dwell so the viewer SEES it (1.5-3.5 typical)
      "caption_zh": <string>,         // 中文字幕，<= 16 字
      "caption_en": <string>          // English caption, <= 6 words
    }
  ]
}

Rules:
- Only use selectors / text from the provided interactables catalog or the
  visible_text. Do NOT invent selectors. If unsure, fall back to text=...
  with the visible button label.
- First step is ALMOST ALWAYS `navigate` to the original URL.
- Each step's wait_after_s is how long the viewer LOOKS AT THE RESULT before
  the next action — keep it real (clicks: 2-3s, scroll/hover: 1-2s).
- Sum of (action_time≈0.5s + wait_after_s) for all steps ≈ duration_s.
- 5-8 steps is the sweet spot for 25-30s.
- Captions tell the story — they should NOT just describe the click ("点了
  按钮"); they should explain the VALUE ("数据动画立即响应"). Pull copy
  from 独特卖点 in the brief whenever you can.
- If something looks complex (login, sign up, pay) skip it; pick simple
  read-only / showcase paths."""


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict:
    m = _FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))
    return json.loads(text)


@traced_agent("Phase 2B · Demo Planner", phase=2)
def plan_demo(run_dir: Path,
              service_url: str,
              project_brief: str,
              output_path: Path,
              duration_s: float = 25.0) -> Path:
    """Capture DOM snapshot + ask LLM → write demo_script.json."""
    log = agent_logger("demo_planner")
    snapshot = capture_dom_snapshot(service_url)

    interactables_summary = "\n".join(
        f"  [{e['i']:3d}] <{e['tag']}{(' role='+e['role']) if e['role'] else ''}> "
        f"text={e['txt']!r}  id={e['id']!r}  class={e['cls'][:40]!r}"
        for e in snapshot.get("interactables", [])
    ) or "  (none captured)"

    user_msg = (
        f"=== project_brief.md ===\n{project_brief}\n\n"
        f"=== target duration ===\n{duration_s}s\n\n"
        f"=== app snapshot ===\n"
        f"url: {snapshot['url']}\n"
        f"title: {snapshot.get('title')!r}\n\n"
        f"--- visible_text (first ~12000 chars) ---\n"
        f"{snapshot.get('visible_text_tail', '')[:6000]}\n\n"
        f"--- interactables ---\n{interactables_summary}\n\n"
        f"--- a11y_tree (truncated) ---\n{snapshot.get('a11y_tree', '')[:4000]}"
    )

    client = anthropic_client()
    model = model_for("reasoning")
    log.info(f"LLM plan_demo model={model} interactables={len(snapshot.get('interactables', []))}")
    resp = client.messages.create(
        model=model,
        max_tokens=2400,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    plan = _extract_json(text)
    if not isinstance(plan.get("steps"), list) or not plan["steps"]:
        raise ValueError(f"LLM returned no steps: {plan}")

    # Augment with the snapshot for downstream debugging
    plan["_meta"] = {
        "service_url": service_url,
        "snapshot_title": snapshot.get("title"),
        "n_interactables": len(snapshot.get("interactables", [])),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    log.info(f"  ✓ {output_path.name}  steps={len(plan['steps'])}  duration={plan.get('duration_s')}s")

    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="demo_planner",
                 name="demo_script", path=str(output_path),
                 n_steps=len(plan["steps"]),
                 duration_s=plan.get("duration_s"),
                 service_url=service_url)
    return output_path
