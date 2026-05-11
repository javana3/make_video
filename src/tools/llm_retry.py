"""LLM call retry wrapper with exponential backoff + escalation hook.

Approved-by-user state machine: 3 attempts with 1s / 5s / 15s backoff. After
3 failures, the wrapped function raises a RetryExhausted exception so the
caller can escalate to the ErrorAgent.

Tool-call errors (agent's `dispatch(tool_name, args)` raising) are NOT
handled here — those get returned as tool_result text and the LLM agent
self-corrects on the next turn. This wrapper is strictly for the LLM call
itself (network 429 / 5xx / SDK errors that happen before the agent gets
to see anything).
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..observability.error_log import log_error


BACKOFF_SECONDS = [1.0, 5.0, 15.0]  # user-approved schedule: 3 attempts
MAX_ATTEMPTS = len(BACKOFF_SECONDS) + 1  # = 4? No: 3 attempts total. See below.


# Clarification: attempt 1 runs immediately. If it fails → sleep BACKOFF_SECONDS[0]=1s →
# attempt 2 runs. If fail → sleep 5s → attempt 3 runs. If fail → raise RetryExhausted.
# So MAX_ATTEMPTS = 3, BACKOFF_SECONDS has 2 sleeps actually used (1s, 5s).
# Re-aligning: keep 3 attempts, sleep schedule between them = [1, 5] seconds.
# But user said 1/5/15 — that means 4 attempts? Re-reading: "重试3次"
# means 3 retries on top of initial = 4 total. Easier: 3 sleeps = 3 retries.
MAX_ATTEMPTS = 4  # initial + 3 retries
RETRY_SLEEPS = [1.0, 5.0, 15.0]  # sleeps BEFORE retry attempts 2, 3, 4


class RetryExhausted(Exception):
    """Raised when all retries fail. Carries the last underlying exception."""
    def __init__(self, agent: str, step_label: str, last_error: BaseException,
                  error_chain: list[BaseException]):
        self.agent = agent
        self.step_label = step_label
        self.last_error = last_error
        self.error_chain = error_chain
        super().__init__(
            f"[{agent}/{step_label}] all {MAX_ATTEMPTS} attempts failed; "
            f"last: {type(last_error).__name__}: {last_error!s:.200}"
        )


def call_with_retries(fn: Callable[[], Any], *,
                      run_dir: Path,
                      agent: str,
                      step_label: str,
                      context_hint: Optional[dict] = None,
                      log=None) -> Any:
    """Run `fn()` with retries on any Exception.

    Logs every failed attempt to <run_dir>/errors.jsonl. On final exhaustion,
    raises RetryExhausted so the caller can route to ErrorAgent.
    """
    error_chain: list[BaseException] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:
            error_chain.append(e)
            is_final = attempt == MAX_ATTEMPTS
            log_error(
                run_dir, agent=agent, step_label=step_label,
                attempt=attempt, max_attempts=MAX_ATTEMPTS,
                error=e, escalated=is_final, context_hint=context_hint or {},
            )
            if log is not None:
                log.warning(
                    f"[retry {attempt}/{MAX_ATTEMPTS}] {agent}/{step_label} "
                    f"{type(e).__name__}: {str(e)[:200]}"
                )
            if is_final:
                raise RetryExhausted(agent, step_label, e, error_chain)
            sleep_s = RETRY_SLEEPS[attempt - 1] if attempt - 1 < len(RETRY_SLEEPS) else RETRY_SLEEPS[-1]
            if log is not None:
                log.info(f"[retry] backoff {sleep_s}s before attempt {attempt + 1}")
            time.sleep(sleep_s)
    # Unreachable; loop either returns or raises.
    raise RuntimeError("call_with_retries: control should never reach here")
