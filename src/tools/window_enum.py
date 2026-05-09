"""Enumerate visible top-level windows on Windows via ctypes.

No third-party deps (no pywin32). Used by the UI to populate a dropdown so the
user picks the project window for ffmpeg gdigrab capture.
"""
from __future__ import annotations

import ctypes
import os
import re
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Optional


# Junk titles to filter (system-internal, not project windows)
_JUNK_TITLES = {
    "Default IME", "MSCTFIME UI", "Program Manager", "Windows Input Experience",
    "Settings", "", " ",
}
_JUNK_PREFIXES = ("DesktopWindow_", "Windows.UI.Core",)


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    pid: int = 0
    score: int = 0
    score_reasons: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.score_reasons is None:
            self.score_reasons = []


def list_windows(min_title_len: int = 2) -> list[WindowInfo]:
    """Return visible top-level windows with non-empty titles, sorted by title."""
    if os.name != "nt":
        return []

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    EnumWindows = user32.EnumWindows
    GetWindowTextLengthW = user32.GetWindowTextLengthW
    GetWindowTextW = user32.GetWindowTextW
    IsWindowVisible = user32.IsWindowVisible
    GetWindowThreadProcessId = user32.GetWindowThreadProcessId

    EnumWindowsProc = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )

    out: list[WindowInfo] = []

    @EnumWindowsProc
    def callback(hwnd, _lparam):
        try:
            if not IsWindowVisible(hwnd):
                return True
            length = GetWindowTextLengthW(hwnd)
            if length < min_title_len:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            if title in _JUNK_TITLES:
                return True
            if any(title.startswith(p) for p in _JUNK_PREFIXES):
                return True
            pid = wintypes.DWORD()
            GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            out.append(WindowInfo(hwnd=int(hwnd), title=title, pid=int(pid.value)))
        except Exception:
            pass
        return True

    EnumWindows(callback, 0)
    out.sort(key=lambda w: w.title.lower())
    # Deduplicate by title (multiple HWNDs can share title — keep first)
    seen: set[str] = set()
    deduped: list[WindowInfo] = []
    for w in out:
        if w.title in seen:
            continue
        seen.add(w.title)
        deduped.append(w)
    return deduped


def score_window(win: WindowInfo, hints: dict) -> WindowInfo:
    """Score a window by how likely it is the project window.

    hints: {
        "service_urls": ["http://127.0.0.1:5500/index.html", ...],
        "project_name": "football-match-simulator",
    }
    Modifies win.score and win.score_reasons in place.
    """
    title = win.title or ""
    title_lower = title.lower()
    score = 0
    reasons: list[str] = []

    urls = hints.get("service_urls") or []
    project_name = (hints.get("project_name") or "").lower()

    # Penalize the workflow web UI itself. Our run page title contains the
    # unique run_id (8-hex-char), which 99.99% won't naturally appear elsewhere.
    run_id = (hints.get("run_id") or "").lower()
    if run_id and run_id in title_lower:
        score -= 200
        reasons.append(f"title contains our run_id {run_id} — this is the workflow UI itself (-200)")
    if "promo video pipeline" in title_lower or " · phase " in title_lower:
        score -= 100
        reasons.append("looks like the workflow UI (-100)")

    # Strong: title mentions a specific port we're serving
    for url in urls:
        m = re.search(r":(\d+)", url)
        if m:
            port = m.group(1)
            if port in title:
                score += 60
                reasons.append(f"title contains port {port} (+60)")
        if "localhost" in title_lower or "127.0.0.1" in title_lower:
            score += 30
            reasons.append("title contains localhost/127.0.0.1 (+30)")
            break

    # Project name in title
    if project_name and len(project_name) >= 4:
        # Try whole name and individual tokens
        if project_name in title_lower:
            score += 25
            reasons.append(f"title contains project name (+25)")
        else:
            tokens = [t for t in re.split(r"[-_/\s]+", project_name) if len(t) >= 4]
            for tok in tokens:
                if tok in title_lower:
                    score += 8
                    reasons.append(f"title contains '{tok}' (+8)")

    # Browser hints
    for browser in ("chrome", "edge", "firefox", "safari", "brave", "opera"):
        if browser in title_lower:
            score += 5
            reasons.append(f"browser ({browser}) (+5)")
            break

    win.score = score
    win.score_reasons = reasons
    return win


def list_windows_ranked(hints: dict, min_title_len: int = 2) -> list[WindowInfo]:
    """List windows + score each + sort by score desc."""
    wins = list_windows(min_title_len=min_title_len)
    for w in wins:
        score_window(w, hints)
    wins.sort(key=lambda w: (-w.score, w.title.lower()))
    return wins


def get_window_rect(title: str) -> Optional[tuple[int, int, int, int]]:
    """Find a window by exact title and return (left, top, width, height) in
    screen pixels. None if not found or window is minimized.

    Uses DwmGetWindowAttribute(DWMWA_EXTENDED_FRAME_BOUNDS) to get the VISIBLE
    rect, NOT the (larger) GetWindowRect bounds — Windows DWM adds invisible
    margins that, if recorded, leak surrounding desktop / other windows into
    the capture. This returns the precise visible frame.
    """
    if os.name != "nt" or not title:
        return None
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
    FindWindow = user32.FindWindowW
    IsIconic = user32.IsIconic
    GetWindowRect = user32.GetWindowRect
    DwmGetWindowAttribute = dwmapi.DwmGetWindowAttribute

    DWMWA_EXTENDED_FRAME_BOUNDS = 9

    hwnd = FindWindow(None, title)
    if not hwnd:
        return None
    if IsIconic(hwnd):
        return None  # minimized — gdigrab can't capture it

    rect = wintypes.RECT()
    # Try DWM-aware bounds first (excludes invisible Windows-Aero margins).
    hr = DwmGetWindowAttribute(
        hwnd,
        DWMWA_EXTENDED_FRAME_BOUNDS,
        ctypes.byref(rect),
        ctypes.sizeof(rect),
    )
    if hr != 0:
        # Fall back to plain GetWindowRect on pre-Vista or DWM-disabled hosts.
        if not GetWindowRect(hwnd, ctypes.byref(rect)):
            return None

    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None
    # h264 (libx264) requires EVEN dimensions; round down to even.
    width -= width % 2
    height -= height % 2
    return (rect.left, rect.top, width, height)


def set_topmost(title: str, on: bool = True) -> bool:
    """Pin/unpin a window as always-on-top via SetWindowPos.

    Used to protect a recording-target window from being occluded by other
    windows (terminals, IDEs, popups) during a screen capture session.
    """
    if os.name != "nt" or not title:
        return False
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    FindWindow = user32.FindWindowW
    SetWindowPos = user32.SetWindowPos
    HWND_TOPMOST = -1
    HWND_NOTOPMOST = -2
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_SHOWWINDOW = 0x0040
    hwnd = FindWindow(None, title)
    if not hwnd:
        return False
    flag = HWND_TOPMOST if on else HWND_NOTOPMOST
    SetWindowPos(hwnd, flag, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
    return True


def bring_to_foreground(title: str, retries: int = 3) -> bool:
    """Find a window by exact title and force it to the foreground.

    Returns True if a matching window was found and a foreground call was
    issued (no guarantee Windows honored it; SetForegroundWindow has rules).
    """
    if os.name != "nt" or not title:
        return False
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    FindWindow = user32.FindWindowW
    ShowWindow = user32.ShowWindow
    SetForegroundWindow = user32.SetForegroundWindow
    BringWindowToTop = user32.BringWindowToTop
    SwitchToThisWindow = user32.SwitchToThisWindow

    SW_RESTORE = 9

    for _ in range(retries):
        hwnd = FindWindow(None, title)
        if hwnd:
            try:
                ShowWindow(hwnd, SW_RESTORE)
                BringWindowToTop(hwnd)
                # SwitchToThisWindow is a less-restricted API
                SwitchToThisWindow(hwnd, True)
                SetForegroundWindow(hwnd)
            except Exception:
                pass
            return True
        time.sleep(0.2)
    return False
