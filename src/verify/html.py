"""HTML asset verification — WORKFLOW.md §7.4.

M0 (current): file existence + index.html presence + offline-CDN warning.
M2 (later):    full Playwright headless render + console error scan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from opentelemetry import trace

tracer = trace.get_tracer("video-workflow")


@dataclass
class HtmlVerifyResult:
    ok: bool
    path: Path
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_EXTERNAL_CDN_HINTS = (
    "https://fonts.googleapis.com",
    "https://cdnjs.",
    "https://cdn.jsdelivr.net",
    "https://unpkg.com",
)


def verify_html(html_dir: Path) -> HtmlVerifyResult:
    with tracer.start_as_current_span("verify.html") as span:
        span.set_attribute("path", str(html_dir))
        issues: list[str] = []
        warnings: list[str] = []

        if not html_dir.exists():
            issues.append(f"directory does not exist: {html_dir}")
            result = HtmlVerifyResult(ok=False, path=html_dir, issues=issues)
            span.set_attribute("verify.passed", result.ok)
            span.set_attribute("verify.issues", len(issues))
            return result

        if not html_dir.is_dir():
            issues.append(f"not a directory: {html_dir}")
            result = HtmlVerifyResult(ok=False, path=html_dir, issues=issues)
            span.set_attribute("verify.passed", result.ok)
            span.set_attribute("verify.issues", len(issues))
            return result

        index = html_dir / "index.html"
        if not index.exists():
            issues.append("index.html missing")
        elif index.stat().st_size == 0:
            issues.append("index.html is empty")
        else:
            text = index.read_text(encoding="utf-8", errors="replace")
            if any(hint in text for hint in _EXTERNAL_CDN_HINTS):
                warnings.append(
                    "index.html references external CDN — may fail offline (WORKFLOW §2.1)"
                )

        result = HtmlVerifyResult(
            ok=len(issues) == 0, path=html_dir,
            issues=issues, warnings=warnings,
        )
        span.set_attribute("verify.passed", result.ok)
        span.set_attribute("verify.issues", len(issues))
        span.set_attribute("verify.warnings", len(warnings))
        return result
