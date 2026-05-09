"""Locate ffmpeg / ffprobe binaries.

Resolution order:
  1. PATH (`shutil.which`)
  2. winget Gyan.FFmpeg install dir under %LOCALAPPDATA%\\Microsoft\\WinGet\\Packages
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


def _winget_search(name: str) -> Optional[str]:
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if not base.exists():
        return None
    exe = name + (".exe" if os.name == "nt" else "")
    for hit in base.glob(f"Gyan.FFmpeg*/**/bin/{exe}"):
        return str(hit)
    return None


def find(name: str) -> str:
    """Return absolute path to `ffmpeg` or `ffprobe`. Raise if not found."""
    found = shutil.which(name)
    if found:
        return found
    found = _winget_search(name)
    if found:
        return found
    raise FileNotFoundError(
        f"{name} not found in PATH or winget Packages dir. "
        f"Install via: winget install Gyan.FFmpeg"
    )


def ffmpeg() -> str:
    return find("ffmpeg")


def ffprobe() -> str:
    return find("ffprobe")
