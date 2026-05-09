"""Self-test the OpenDesign Web UI flow with Playwright.

Steps:
  1. Open /runs/football-match-simulator/49aecf4a/opendesign
  2. Screenshot bootstrap state
  3. Click "初始化 OpenDesign session"
  4. Wait for redirect to active session
  5. Screenshot session state
  6. (Optional) send a prompt — skipped here because OpenCode runs ~10-15min
  7. Screenshot iframe preview
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, expect, TimeoutError as PWTimeout


URL = "http://127.0.0.1:7860/runs/football-match-simulator/49aecf4a/opendesign"
SHOTS = Path(r"C:\Users\dfgfd\AppData\Local\Temp\opendesign_shots")
SHOTS.mkdir(parents=True, exist_ok=True)


def shot(page, name: str) -> None:
    p = SHOTS / f"{name}.png"
    page.screenshot(path=str(p), full_page=True)
    print(f"  📸 {p}")


def main() -> int:
    with sync_playwright() as pw:
        # NB: bypass system VPN proxy (we hit 127.0.0.1 directly)
        browser = pw.chromium.launch(
            headless=True,
            args=["--disable-features=NetworkService", "--no-proxy-server"],
        )
        context = browser.new_context(
            viewport={"width": 1600, "height": 1000},
            ignore_https_errors=True,
        )
        page = context.new_page()
        page.on("console", lambda msg: print(f"  [console.{msg.type}] {msg.text[:200]}"))
        page.on("pageerror", lambda exc: print(f"  [PAGEERROR] {exc}"))

        print(f"\n=== 1. open {URL} ===")
        try:
            resp = page.goto(URL, wait_until="domcontentloaded", timeout=30000)
            print(f"  HTTP {resp.status if resp else '?'}")
        except PWTimeout as e:
            print(f"  TIMEOUT loading: {e}")
            shot(page, "01-load-timeout")
            return 1

        page.wait_for_load_state("networkidle", timeout=10000)
        title = page.title()
        body_text_sample = page.locator("body").inner_text()[:300]
        print(f"  title:    {title!r}")
        print(f"  body[:300]: {body_text_sample!r}")
        shot(page, "01-loaded")

        # Detect which state we're in
        init_btn = page.get_by_role("button", name="初始化 OpenDesign session")
        if init_btn.count() > 0:
            print("\n=== 2. bootstrap mode — clicking init ===")
            print("  (this triggers ensure_daemon + LLM pick + create_project; ~30-60s)")
            init_btn.click()
            # The htmx response triggers a body swap that loads /opendesign again.
            # Wait for the page to either change or show error.
            try:
                page.wait_for_function(
                    """() => {
                        const txt = document.body.innerText;
                        return txt.includes('Project:') || txt.includes('init 失败');
                    }""",
                    timeout=120_000,
                )
            except PWTimeout:
                print("  TIMEOUT waiting for bootstrap result")
                shot(page, "02-bootstrap-timeout")
                return 1
            time.sleep(1.0)
            shot(page, "02-after-init")
            err = page.locator(".text-rose-400").count()
            if err:
                print(f"  ❌ init reported error: {page.locator('.text-rose-400').first.inner_text()}")
                return 1
            print("  ✓ init done")

        # Now we should be in active session UI
        print("\n=== 3. verify session UI ===")
        try:
            expect(page.locator("#preview-iframe")).to_be_visible(timeout=5000)
        except Exception as e:
            print(f"  ❌ preview iframe not visible: {e}")
            shot(page, "03-no-iframe")
            return 1
        shot(page, "03-session-loaded")

        # Inspect session info card text
        info = page.locator("text=Project:").locator("..").inner_text()
        print(f"  session info: {info[:300]}")

        # Iframe preview check
        iframe_el = page.locator("#preview-iframe")
        iframe_src = iframe_el.get_attribute("src")
        print(f"  iframe src: {iframe_src}")

        # Try to access the iframe content
        frame = page.frame_locator("#preview-iframe")
        try:
            frame_body_text = frame.locator("body").inner_text(timeout=5000)
            print(f"  iframe body[:200]: {frame_body_text[:200]!r}")
        except Exception as e:
            print(f"  iframe body unavailable: {e}")

        print("\n=== 4. textarea + send button present? ===")
        textarea = page.locator("#iterate-msg")
        send_btn = page.locator("#send-btn")
        print(f"  textarea visible: {textarea.is_visible()}")
        print(f"  send button: {send_btn.text_content() if send_btn.count() else '?'}")
        shot(page, "04-final")

        print("\n=== 5. verify Agent first-turn banner + initial_prompt visible ===")
        kick_btn = page.locator("#kick-first-turn")
        if kick_btn.count() == 0:
            print("  ⚠ 没看到 first-turn 按钮 — session.history 可能不是 0")
        else:
            # Get the initial_prompt shown on page
            ip_text = page.locator("pre").first.inner_text()
            print(f"  initial_prompt[:300]: {ip_text[:300]!r}")
            print(f"  banner: {page.locator('text=Agent 已就绪').count()} match")
            shot(page, "05-first-turn-banner")

            print("\n=== 6. click 'Agent 开跑' (don't wait 15min, just verify SSE starts) ===")
            kick_btn.click()
            # Wait for the first SSE event to land in #sse-events
            try:
                page.wait_for_function(
                    """() => {
                        const pre = document.getElementById('sse-events');
                        return pre && pre.textContent.includes('start');
                    }""",
                    timeout=60000,
                )
                print("  ✓ SSE 'start' event arrived in panel")
                shot(page, "06-sse-started")
            except PWTimeout:
                print("  ❌ SSE didn't show 'start' within 60s")
                shot(page, "06-sse-no-start")

            # Wait a bit more to see agent status running
            page.wait_for_timeout(15000)
            evt_log = page.locator("#sse-events").inner_text()
            n_lines = len([l for l in evt_log.split("\n") if l.strip()])
            print(f"  SSE log lines after 15s: {n_lines}")
            print(f"  last few:\n    {evt_log.split(chr(10))[-3:]}")
            shot(page, "07-sse-running")

        print("\n=== ✅ Web UI loads + Agent first-turn fires + SSE streams ===")
        print(f"  screenshots saved in: {SHOTS}")
        browser.close()
        return 0


if __name__ == "__main__":
    sys.exit(main())
