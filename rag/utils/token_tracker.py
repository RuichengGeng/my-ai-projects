"""
Lightweight token-usage tracker for DeepSeek / OpenAI-compatible API calls.

Usage pattern
-------------
Every agent sets the active tracker at the start of a run via a context
variable so _deepseek() can record without the tracker being threaded through
every function signature:

    from rag.utils.token_tracker import UsageTracker, set_active_tracker

    tracker = UsageTracker()
    with set_active_tracker(tracker):
        ...  # all _deepseek() calls inside here are recorded
    tracker.print_summary()
    tracker.save_jsonl(USAGE_LOG_PATH, metadata={"question": "..."})

Call sites pass a short label so the breakdown shows per-step costs:
    _deepseek([...], label="plan")
"""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Per-call record
# ---------------------------------------------------------------------------

@dataclass
class _CallRecord:
    label: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class UsageTracker:
    """Accumulates token usage and estimates cost for one agent run."""

    # DeepSeek Chat (deepseek-chat = DeepSeek-V3) pricing, USD per million tokens.
    # Update if DeepSeek changes pricing: https://platform.deepseek.com/api-docs/pricing
    PRICE_INPUT_PER_M:  float = 0.27   # cache miss
    PRICE_CACHED_PER_M: float = 0.07   # cache hit
    PRICE_OUTPUT_PER_M: float = 1.10

    def __init__(self) -> None:
        self._calls: list[_CallRecord] = []

    def record(self, usage, label: str = "") -> None:
        """Record token counts from an API response usage object."""
        if usage is None:
            return
        # DeepSeek exposes cache hits as prompt_cache_hit_tokens;
        # newer OpenAI SDK exposes them under usage.prompt_tokens_details.cached_tokens
        cached = (
            getattr(usage, "prompt_cache_hit_tokens", 0)
            or getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0)
            or 0
        )
        self._calls.append(_CallRecord(
            label=label,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=cached,
        ))

    def reset(self) -> None:
        self._calls.clear()

    # ── Aggregates ──────────────────────────────────────────────────────────

    @property
    def total_calls(self) -> int:
        return len(self._calls)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self._calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self._calls)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_cached_tokens(self) -> int:
        return sum(c.cached_tokens for c in self._calls)

    @property
    def estimated_cost_usd(self) -> float:
        uncached = max(0, self.total_prompt_tokens - self.total_cached_tokens)
        return (
            uncached                    * self.PRICE_INPUT_PER_M  / 1_000_000
            + self.total_cached_tokens  * self.PRICE_CACHED_PER_M / 1_000_000
            + self.total_completion_tokens * self.PRICE_OUTPUT_PER_M / 1_000_000
        )

    def breakdown(self) -> dict[str, dict]:
        """Return per-label aggregates: {label: {calls, prompt_tokens, completion_tokens}}."""
        result: dict[str, dict] = {}
        for c in self._calls:
            key = c.label or "unknown"
            bucket = result.setdefault(key, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
            bucket["calls"] += 1
            bucket["prompt_tokens"] += c.prompt_tokens
            bucket["completion_tokens"] += c.completion_tokens
        return result

    # ── Output ──────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        print(f"\n{'─' * 52}")
        print("  Token usage")
        print(f"{'─' * 52}")
        print(f"  LLM calls          : {self.total_calls}")
        print(f"  Prompt tokens      : {self.total_prompt_tokens:>10,}  "
              f"(cached: {self.total_cached_tokens:,})")
        print(f"  Completion tokens  : {self.total_completion_tokens:>10,}")
        print(f"  Total tokens       : {self.total_tokens:>10,}")
        print(f"  Estimated cost     : ${self.estimated_cost_usd:.5f}")
        if self._calls:
            print("  ── by step ─────────────────────────────────────")
            for label, s in self.breakdown().items():
                print(f"    {label:<22} calls={s['calls']}  "
                      f"in={s['prompt_tokens']:,}  out={s['completion_tokens']:,}")
        print(f"{'─' * 52}\n")

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "breakdown": self.breakdown(),
            "calls": [
                {
                    "label": c.label,
                    "prompt_tokens": c.prompt_tokens,
                    "completion_tokens": c.completion_tokens,
                    "cached_tokens": c.cached_tokens,
                }
                for c in self._calls
            ],
        }

    def save_jsonl(self, path: Path, metadata: dict | None = None) -> None:
        """Append one JSON line to the usage log file."""
        record = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **(metadata or {}),
            **self.to_dict(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Context variable — one active tracker per async/thread context
# ---------------------------------------------------------------------------

_active_tracker: ContextVar[UsageTracker | None] = ContextVar(
    "_active_tracker", default=None
)


def get_active_tracker() -> UsageTracker | None:
    return _active_tracker.get()


@contextmanager
def set_active_tracker(tracker: UsageTracker):
    """Context manager that activates a tracker for the duration of the block."""
    token = _active_tracker.set(tracker)
    try:
        yield tracker
    finally:
        _active_tracker.reset(token)
