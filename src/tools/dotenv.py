"""Minimal .env loader — avoids the python-dotenv dependency."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_dotenv(path: Optional[Path] = None) -> None:
    """Load KEY=VALUE lines from `path` into os.environ. Existing values win."""
    if path is None:
        path = Path.cwd() / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
