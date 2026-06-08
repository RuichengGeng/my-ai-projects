"""
Exponential-backoff retry for transient OpenAI / DeepSeek API errors.

Usage:
    from utils.retry import call_with_retry

    resp = call_with_retry(
        lambda: client.chat.completions.create(...),
        retries=3,
        base_delay=2.0,
    )
"""

import time

# OpenAI SDK exception class names that indicate a transient server-side problem.
# We match by name so there is no hard import of the openai package here.
_RETRYABLE_TYPES = frozenset({
    "RateLimitError",         # 429
    "InternalServerError",    # 500
    "APIStatusError",         # catch-all for 5xx variants
    "APIConnectionError",     # network-level failure
    "APITimeoutError",        # request timed out
})


def call_with_retry(fn, *, retries: int = 3, base_delay: float = 2.0):
    """Call fn(); on transient API errors retry with exponential backoff.

    Args:
        fn:          Zero-argument callable that makes the API call.
        retries:     Maximum number of retry attempts (not counting the first try).
        base_delay:  Seconds to wait before the first retry; doubles each round.

    Raises:
        The original exception if all retries are exhausted or if the error is
        not considered transient.
    """
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            exc_name = type(exc).__name__
            status   = getattr(exc, "status_code", None)

            is_transient = exc_name in _RETRYABLE_TYPES or (
                isinstance(status, int) and status >= 500
            )

            if not is_transient or attempt == retries:
                raise

            delay = base_delay * (2 ** attempt)
            print(
                f"  [retry] {exc_name}"
                + (f" (HTTP {status})" if status else "")
                + f"  attempt {attempt + 1}/{retries}  waiting {delay:.0f}s …"
            )
            time.sleep(delay)

    # unreachable — kept for type checkers
    raise RuntimeError("call_with_retry: exhausted retries without raising")
