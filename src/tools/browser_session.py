"""Long-lived browser session for the demo-driver agent.

A `BrowserSession` wraps Playwright Chromium with `record_video_dir` so the
entire session lifetime is captured natively. The driver agent operates the
page through `goto`, `click`, `fill`, `press`, `scroll`, `wait_for`,
`screenshot`, `visible_text`, `a11y_snapshot`, and decides via `stop()`
when the recording ends.

There is NO duration cap. The session lives until the driver calls stop().
On stop, Playwright finalises the .webm; we transcode to .mp4 with ffmpeg
so it slots into the rest of the pipeline cleanly.
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..observability.audit import traced_step
from ..observability.logger import agent_logger
from .ffbin import ffmpeg, ffprobe
from .shell import run as shell_run


@dataclass
class BrowserSessionResult:
    output_path: str
    duration_s: float
    size_bytes: int
    width: int
    height: int
    n_actions: int
    started_at: str
    stopped_at: str
    final_url: Optional[str]


def _shrink_a11y(node: dict, depth: int = 0, max_depth: int = 6,
                 max_chars: int = 12000, _acc: Optional[list] = None) -> str:
    """Render an accessibility tree as a compact text outline."""
    if _acc is None:
        _acc = []
    if not isinstance(node, dict) or depth > max_depth:
        return ""
    role = node.get("role", "")
    name = (node.get("name") or "").strip()[:80]
    val = (node.get("value") or "")
    if isinstance(val, str):
        val = val.strip()[:80]
    line_parts = [f"  " * depth + role]
    if name:
        line_parts.append(f"name={name!r}")
    if val:
        line_parts.append(f"value={val!r}")
    if node.get("focused"):
        line_parts.append("[focused]")
    if node.get("checked") is not None:
        line_parts.append(f"checked={node['checked']}")
    line = " ".join(line_parts)
    _acc.append(line)
    rendered = "\n".join(_acc)
    if len(rendered) > max_chars:
        return rendered[:max_chars] + "\n[...truncated]"
    for child in node.get("children") or []:
        _shrink_a11y(child, depth + 1, max_depth, max_chars, _acc)
        if sum(len(s) for s in _acc) > max_chars:
            break
    return "\n".join(_acc)[:max_chars]


class BrowserSession:
    """Concurrent browser + native video recording.

    Lifetime:
        s = BrowserSession.start(record_dir=..., viewport=(1920, 1080))
        s.goto("http://localhost:3000")
        s.click("button:has-text('Login')")
        s.fill("input[name=email]", "demo@example.com")
        s.screenshot()  # for LLM vision
        ...
        result = s.stop(output_path=Path("recording.mp4"))
    """

    def __init__(self,
                 record_dir: Path,
                 viewport_w: int = 1920,
                 viewport_h: int = 1080,
                 headless: bool = True,
                 default_timeout_ms: int = 15000):
        self.record_dir = record_dir
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.headless = headless
        self.default_timeout_ms = default_timeout_ms

        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._t0: float = 0.0
        self._stopped_at: Optional[float] = None
        self._n_actions = 0
        self.started_at_iso: Optional[str] = None

        # Observability buffers — populated by page.on(...) listeners so the
        # agent can later query JS console messages and HTTP failures via
        # tools. Bounded to avoid unbounded memory growth on long sessions.
        self._console_log: list[dict] = []
        self._network_failures: list[dict] = []
        self._page_errors: list[dict] = []
        self._max_buffer = 500

        self.log = agent_logger("browser_session")

    # ─── lifecycle ─────────────────────────────────────────────────────
    @classmethod
    def start(cls, record_dir: Path, **kwargs) -> "BrowserSession":
        s = cls(record_dir, **kwargs)
        s._launch()
        return s

    def _launch(self) -> None:
        if self.record_dir.exists():
            shutil.rmtree(self.record_dir)
        self.record_dir.mkdir(parents=True)

        from playwright.sync_api import sync_playwright
        self.log.info(f"launch chromium  viewport={self.viewport_w}x{self.viewport_h}  "
                      f"record_dir={self.record_dir}")
        self._t0 = time.monotonic()
        self.started_at_iso = datetime.now(timezone.utc).isoformat()
        self._pw = sync_playwright().start()
        # --no-proxy-server: bypass system VPN proxy for localhost demo targets
        self._browser = self._pw.chromium.launch(
            headless=self.headless,
            args=["--no-proxy-server"],
        )
        self._context = self._browser.new_context(
            viewport={"width": self.viewport_w, "height": self.viewport_h},
            record_video_dir=str(self.record_dir),
            record_video_size={"width": self.viewport_w, "height": self.viewport_h},
        )
        self._context.set_default_timeout(self.default_timeout_ms)
        self._page = self._context.new_page()
        self._page.on("dialog", lambda d: d.accept())

        # Console messages: type ∈ {log, info, warning, error, debug, ...}.
        def _on_console(msg):
            try:
                entry = {
                    "ts_action": self._n_actions,
                    "ts_iso": datetime.now(timezone.utc).isoformat(),
                    "type": msg.type,
                    "text": msg.text,
                    "location": str(msg.location) if msg.location else None,
                }
            except Exception as e:
                entry = {"ts_action": self._n_actions, "type": "log",
                          "text": f"<console capture failed: {e}>"}
            self._console_log.append(entry)
            if len(self._console_log) > self._max_buffer:
                self._console_log = self._console_log[-self._max_buffer:]
        self._page.on("console", _on_console)

        # Network responses: capture 4xx/5xx so agent can see failed XHR/fetch.
        def _on_response(resp):
            try:
                status = resp.status
                if status < 400:
                    return
                entry = {
                    "ts_action": self._n_actions,
                    "ts_iso": datetime.now(timezone.utc).isoformat(),
                    "url": resp.url,
                    "status": status,
                    "method": resp.request.method,
                    "request_url": resp.request.url,
                }
            except Exception as e:
                entry = {"ts_action": self._n_actions,
                          "url": "?", "status": -1, "method": "?",
                          "error": str(e)}
            self._network_failures.append(entry)
            if len(self._network_failures) > self._max_buffer:
                self._network_failures = self._network_failures[-self._max_buffer:]
        self._page.on("response", _on_response)

        # Uncaught JS exceptions (pageerror event).
        def _on_pageerror(exc):
            entry = {
                "ts_action": self._n_actions,
                "ts_iso": datetime.now(timezone.utc).isoformat(),
                "message": str(exc),
            }
            self._page_errors.append(entry)
            if len(self._page_errors) > self._max_buffer:
                self._page_errors = self._page_errors[-self._max_buffer:]
        self._page.on("pageerror", _on_pageerror)

    # ─── observability getters (called by agent tools) ─────────────────
    def console_log(self, level: Optional[str] = None, since_action: int = 0,
                      limit: int = 100) -> list[dict]:
        """Return buffered console messages, optionally filtered by level.

        level: None (all) | 'error' | 'warning' | 'info' | 'log' | 'debug'.
        since_action: only entries with ts_action >= this (use after a click
                       to see new console output since then).
        """
        out = self._console_log
        if level:
            out = [e for e in out if e.get("type") == level]
        if since_action > 0:
            out = [e for e in out if e.get("ts_action", 0) >= since_action]
        return out[-limit:]

    def network_failures(self, since_action: int = 0,
                            min_status: int = 400,
                            limit: int = 100) -> list[dict]:
        """Return buffered 4xx/5xx responses, optionally filtered."""
        out = [e for e in self._network_failures
               if e.get("status", 0) >= min_status
               and e.get("ts_action", 0) >= since_action]
        return out[-limit:]

    def page_errors(self, since_action: int = 0, limit: int = 50) -> list[dict]:
        """Return uncaught JS exceptions."""
        out = [e for e in self._page_errors
               if e.get("ts_action", 0) >= since_action]
        return out[-limit:]

    @property
    def n_actions(self) -> int:
        return self._n_actions

    # ─── agent-facing API ──────────────────────────────────────────────
    def goto(self, url: str, wait_until: str = "domcontentloaded",
             timeout_ms: Optional[int] = None) -> dict:
        with traced_step("browser.goto", url=url, wait_until=wait_until):
            self._n_actions += 1
            assert self._page is not None
            r = self._page.goto(url, wait_until=wait_until,
                                timeout=timeout_ms or self.default_timeout_ms)
            return {
                "url": self._page.url,
                "status": r.status if r is not None else None,
                "title": self._page.title(),
            }

    def click(self, target: str, by: str = "auto",
              timeout_ms: Optional[int] = None) -> dict:
        """Click by selector / text / role.

        by: 'selector' | 'text' | 'role' | 'auto' (try selector → text).
        target: CSS selector, or visible text, or 'role:button[name=\"Save\"]'.
        """
        with traced_step("browser.click", target=target, by=by):
            self._n_actions += 1
            assert self._page is not None
            timeout = timeout_ms or self.default_timeout_ms
            errors = []
            attempts = []
            if by in ("auto", "selector"):
                attempts.append(("selector", target))
            if by in ("auto", "text"):
                attempts.append(("text", f"text={target}"))
            if by == "role":
                attempts = [("role", target)]
            for kind, sel in attempts:
                try:
                    self._page.click(sel, timeout=timeout)
                    return {"clicked_via": kind, "selector": sel,
                            "url_after": self._page.url}
                except Exception as e:
                    errors.append(f"{kind}({sel!r}): {e}")
            raise RuntimeError("click failed; tried: " + " | ".join(errors))

    def fill(self, selector: str, text: str,
             timeout_ms: Optional[int] = None) -> dict:
        with traced_step("browser.fill", selector=selector, text=text[:200]):
            self._n_actions += 1
            assert self._page is not None
            self._page.fill(selector, text,
                             timeout=timeout_ms or self.default_timeout_ms)
            return {"selector": selector, "filled_chars": len(text)}

    def press(self, key: str, selector: Optional[str] = None) -> dict:
        with traced_step("browser.press", key=key, selector=selector or ""):
            self._n_actions += 1
            assert self._page is not None
            if selector:
                self._page.press(selector, key)
            else:
                self._page.keyboard.press(key)
            return {"key": key}

    def scroll(self, dy: int = 600, dx: int = 0) -> dict:
        with traced_step("browser.scroll", dy=dy, dx=dx):
            self._n_actions += 1
            assert self._page is not None
            self._page.mouse.wheel(dx, dy)
            return {"dy": dy, "dx": dx}

    def hover(self, target: str) -> dict:
        with traced_step("browser.hover", target=target):
            self._n_actions += 1
            assert self._page is not None
            self._page.hover(target,
                              timeout=self.default_timeout_ms)
            return {"target": target}

    def wait_for(self, selector: Optional[str] = None,
                 text: Optional[str] = None,
                 timeout_ms: int = 30000) -> dict:
        """Wait for selector visible OR text visible OR plain delay (selector=text=None)."""
        with traced_step("browser.wait_for",
                          selector=selector or "", text=text or "",
                          timeout_ms=timeout_ms) as span:
            assert self._page is not None
            if selector:
                self._page.wait_for_selector(selector, timeout=timeout_ms)
                span.set_attribute("step.result", "selector_visible")
                return {"matched": "selector", "selector": selector}
            if text:
                # locator with text= — wait for visible
                self._page.locator(f"text={text}").first.wait_for(
                    state="visible", timeout=timeout_ms)
                span.set_attribute("step.result", "text_visible")
                return {"matched": "text", "text": text}
            self._page.wait_for_timeout(timeout_ms)
            span.set_attribute("step.result", "delay")
            return {"matched": "delay", "ms": timeout_ms}

    def screenshot(self, full_page: bool = False) -> bytes:
        with traced_step("browser.screenshot", full_page=full_page):
            assert self._page is not None
            return self._page.screenshot(full_page=full_page, type="png")

    def screenshot_b64(self, full_page: bool = False) -> str:
        return base64.b64encode(self.screenshot(full_page=full_page)).decode()

    def url(self) -> Optional[str]:
        return self._page.url if self._page else None

    def title(self) -> Optional[str]:
        return self._page.title() if self._page else None

    def visible_text(self, max_chars: int = 8000) -> str:
        with traced_step("browser.visible_text", max_chars=max_chars):
            assert self._page is not None
            try:
                t = self._page.evaluate("() => document.body.innerText") or ""
            except Exception:
                t = ""
            return t[:max_chars]

    def a11y_snapshot(self, max_chars: int = 12000) -> str:
        """Accessibility tree via Chrome DevTools Protocol.

        playwright >= 1.50 dropped `page.accessibility`; this uses CDP
        directly so the agent still gets a semantic tree."""
        with traced_step("browser.a11y_snapshot", max_chars=max_chars):
            assert self._page is not None and self._context is not None
            try:
                client = self._context.new_cdp_session(self._page)
                client.send("Accessibility.enable")
                resp = client.send("Accessibility.getFullAXTree") or {}
                nodes = resp.get("nodes") or []
            except Exception as e:
                self.log.warning(f"a11y err: {e}")
                return ""

            # Convert flat node list (CDP) → indented outline
            by_id = {n.get("nodeId"): n for n in nodes}
            roots = [n for n in nodes if not n.get("parentId") or n.get("parentId") not in by_id]

            def fmt_value(v: Any) -> str:
                if isinstance(v, dict):
                    return str(v.get("value", ""))[:80]
                return str(v)[:80]

            lines: list[str] = []
            def walk(node: dict, depth: int) -> None:
                role = fmt_value(node.get("role") or "")
                name = fmt_value(node.get("name") or "")
                value = fmt_value(node.get("value") or "")
                if role in ("InlineTextBox", "StaticText") and not name:
                    return
                bits = ["  " * depth + role]
                if name:
                    bits.append(f"name={name!r}")
                if value:
                    bits.append(f"value={value!r}")
                lines.append(" ".join(bits))
                if sum(len(s) for s in lines) > max_chars:
                    return
                for child_id in node.get("childIds") or []:
                    child = by_id.get(child_id)
                    if child:
                        walk(child, depth + 1)

            for root in roots:
                walk(root, 0)
                if sum(len(s) for s in lines) > max_chars:
                    break
            out = "\n".join(lines)
            return out[:max_chars] + ("\n[...truncated]" if len(out) > max_chars else "")

    def interactables(self, max_items: int = 80) -> list[dict]:
        """Return a catalog of clickable/typeable elements with stable selectors."""
        with traced_step("browser.interactables", max_items=max_items):
            assert self._page is not None
            try:
                items = self._page.evaluate("""(MAX) => {
                  const out = [];
                  const sels = 'button, a, input, textarea, select, [role=button], [role=link], [role=tab]';
                  document.querySelectorAll(sels).forEach((el, i) => {
                    if (i >= MAX) return;
                    const tag = el.tagName.toLowerCase();
                    const role = el.getAttribute('role') || '';
                    const txt = (el.innerText || el.value || el.placeholder || '').trim().slice(0, 80);
                    const id = el.id || '';
                    const name = el.getAttribute('name') || '';
                    const aria = el.getAttribute('aria-label') || '';
                    const cls = (el.className && typeof el.className === 'string') ? el.className.slice(0, 80) : '';
                    out.push({ tag, role, text: txt, id, name, aria, cls });
                  });
                  return out;
                }""", max_items) or []
                return items
            except Exception as e:
                self.log.warning(f"interactables err: {e}")
                return []

    def is_alive(self) -> bool:
        return self._browser is not None and self._browser.is_connected()

    def elapsed_s(self) -> float:
        end = self._stopped_at if self._stopped_at is not None else time.monotonic()
        return end - self._t0

    # ─── shutdown / video finalisation ─────────────────────────────────
    def stop(self, output_path: Path) -> BrowserSessionResult:
        if self._page is None:
            raise RuntimeError("session never started")
        with traced_step("browser.stop", output_path=str(output_path)):
            final_url = None
            try:
                final_url = self._page.url
            except Exception:
                pass
            self._stopped_at = time.monotonic()
            try:
                # close context first → finalises the .webm
                if self._context:
                    self._context.close()
            except Exception as e:
                self.log.warning(f"context close err: {e}")
            try:
                if self._browser:
                    self._browser.close()
            except Exception as e:
                self.log.warning(f"browser close err: {e}")
            try:
                if self._pw:
                    self._pw.stop()
            except Exception as e:
                self.log.warning(f"playwright stop err: {e}")

            return self._transcode_to_mp4(output_path, final_url=final_url)

    def _transcode_to_mp4(self, output_path: Path,
                          final_url: Optional[str]) -> BrowserSessionResult:
        webms = sorted(self.record_dir.glob("*.webm"))
        if not webms:
            raise RuntimeError(f"no .webm produced in {self.record_dir}")
        webm = webms[-1]  # most recent

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with traced_step("browser.ffmpeg_transcode",
                          src=str(webm), dst=str(output_path)):
            cmd = [
                ffmpeg(), "-y", "-i", str(webm),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path),
            ]
            shell_run(cmd, check=True, timeout=900)

        probe = shell_run([ffprobe(), "-v", "quiet", "-print_format", "json",
                            "-show_streams", "-show_format", str(output_path)],
                           check=True)
        pdata = json.loads(probe.stdout)
        v = next((s for s in pdata.get("streams", []) if s.get("codec_type") == "video"), {})
        fmt = pdata.get("format", {})

        try:
            shutil.rmtree(self.record_dir)
        except Exception:
            pass

        return BrowserSessionResult(
            output_path=str(output_path),
            duration_s=float(fmt.get("duration", 0)),
            size_bytes=int(fmt.get("size", 0)),
            width=int(v.get("width", 0)),
            height=int(v.get("height", 0)),
            n_actions=self._n_actions,
            started_at=self.started_at_iso or "",
            stopped_at=datetime.now(timezone.utc).isoformat(),
            final_url=final_url,
        )
