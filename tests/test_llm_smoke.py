"""LLM smoke test — verify Ark coding endpoint works via both SDKs.

Tests two paths:
  1. anthropic SDK → Ark Anthropic-compatible endpoint
  2. openai SDK → Ark OpenAI-compatible endpoint (Claude model)

Loads creds from .env at repo root.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_dotenv(path: Path) -> None:
    """Tiny .env loader — no dependency on python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def test_anthropic_sdk() -> bool:
    print("\n--- [1/2] anthropic SDK → Ark Anthropic-compatible endpoint ---")
    from anthropic import Anthropic
    client = Anthropic(
        api_key=os.environ["ARK_KEY_1"],
        base_url=os.environ["ARK_BASE_URL_ANTHROPIC"],
    )
    try:
        resp = client.messages.create(
            model=os.environ.get("LLM_REASONING", "claude-sonnet-4-20250514"),
            max_tokens=60,
            messages=[{"role": "user", "content": "Reply with just: PONG"}],
        )
        text = resp.content[0].text if resp.content else ""
        print(f"   reply: {text!r}")
        print(f"   stop_reason: {resp.stop_reason}")
        print(f"   usage: in={resp.usage.input_tokens} out={resp.usage.output_tokens}")
        ok = "PONG" in text.upper()
        print(f"   {'[OK]' if ok else '[FAIL]'} anthropic SDK")
        return ok
    except Exception as e:
        print(f"   [FAIL] {type(e).__name__}: {e}")
        return False


def test_openai_sdk() -> bool:
    print("\n--- [2/2] openai SDK → Ark OpenAI-compatible endpoint ---")
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["ARK_KEY_1"],
        base_url=os.environ["ARK_BASE_URL_OPENAI"],
    )
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_FAST", "deepseek-v3.2"),
            max_tokens=60,
            messages=[{"role": "user", "content": "Reply with just: PONG"}],
        )
        text = resp.choices[0].message.content or ""
        print(f"   model: {resp.model}")
        print(f"   reply: {text!r}")
        print(f"   usage: prompt={resp.usage.prompt_tokens} comp={resp.usage.completion_tokens}")
        ok = "PONG" in text.upper()
        print(f"   {'[OK]' if ok else '[FAIL]'} openai SDK")
        return ok
    except Exception as e:
        print(f"   [FAIL] {type(e).__name__}: {e}")
        return False


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")

    if not os.environ.get("ARK_KEY_1"):
        print("ERROR: ARK_KEY_1 not set; check .env")
        return 1

    a_ok = test_anthropic_sdk()
    o_ok = test_openai_sdk()

    print(f"\n=== summary ===")
    print(f"  anthropic SDK : {'OK' if a_ok else 'FAIL'}")
    print(f"  openai SDK    : {'OK' if o_ok else 'FAIL'}")

    return 0 if (a_ok and o_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
