"""
DeepSeek API client — OpenAI-compatible, zero-dependency beyond `openai`.

Usage:
    client = DeepSeekClient()
    reply = client.chat("What is NVDA's P/E ratio?")
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

# ── Constants ────────────────────────────────────────────────────────────────

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096


# ── Exceptions ───────────────────────────────────────────────────────────────

class DeepSeekError(Exception):
    """Base exception for DeepSeek client errors."""


class DeepSeekAuthError(DeepSeekError):
    """Raised when the API key is missing or invalid."""


class DeepSeekRateLimitError(DeepSeekError):
    """Raised when the API rate limit is hit."""


# ── Client ───────────────────────────────────────────────────────────────────

@dataclass
class DeepSeekConfig:
    """Configuration for the DeepSeek client."""

    api_key: str = ""
    base_url: str = DEEPSEEK_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: int = 60

    def __post_init__(self):
        # Fall back to env var if no key provided
        if not self.api_key:
            self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        # Strip any trailing slash from base_url
        self.base_url = self.base_url.rstrip("/")


class DeepSeekClient:
    """
    Lightweight wrapper around the DeepSeek API.

    Features:
      - Chat completions (streaming + non-streaming)
      - Reasoning-mode support (deepseek-reasoner)
      - Token usage tracking
      - Automatic retry on rate limits (optional)
    """

    def __init__(self, config: Optional[DeepSeekConfig] = None):
        self.config = config or DeepSeekConfig()

        if not self.config.api_key:
            raise DeepSeekAuthError(
                "DeepSeek API key not found. "
                "Set DEEPSEEK_API_KEY env var or pass api_key to DeepSeekConfig."
            )

        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> "ChatResponse":
        """
        Send a chat message and get a response.

        Args:
            message: The user's message.
            system_prompt: Optional system-level instruction.
            model: Override the default model (e.g. "deepseek-reasoner").
            temperature: Override the default temperature.
            max_tokens: Override the default max tokens.
            stream: If True, return a streaming response.

        Returns:
            ChatResponse with the reply text and usage metadata.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        kwargs = dict(
            model=model or self.config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            stream=stream,
        )

        try:
            if stream:
                return self._stream_chat(**kwargs)
            else:
                return self._complete_chat(**kwargs)
        except Exception as e:
            self._raise_wrapped(e)

    def chat_with_history(
        self,
        messages: list[dict],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> "ChatResponse":
        """
        Send a full message history (for multi-turn conversations).

        Each message should be {"role": "system"|"user"|"assistant", "content": "..."}.
        """
        kwargs = dict(
            model=model or self.config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            stream=stream,
        )

        try:
            if stream:
                return self._stream_chat(**kwargs)
            else:
                return self._complete_chat(**kwargs)
        except Exception as e:
            self._raise_wrapped(e)

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Estimate token count for a given text.
        Uses a simple heuristic (4 chars ≈ 1 token) unless a proper tokenizer is available.
        """
        # Rough estimate — replace with tiktoken if needed
        return len(text) // 4

    # ── Internals ────────────────────────────────────────────────────────

    def _complete_chat(self, **kwargs) -> "ChatResponse":
        kwargs["stream"] = False
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        return ChatResponse(
            text=choice.message.content or "",
            model=response.model,
            usage=ChatUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            ),
            finish_reason=choice.finish_reason,
        )

    def _stream_chat(self, **kwargs) -> "ChatResponse":
        kwargs["stream"] = True
        stream = self._client.chat.completions.create(**kwargs)

        collected_chunks = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                collected_chunks.append(chunk.choices[0].delta.content)

        return ChatResponse(
            text="".join(collected_chunks),
            model=kwargs.get("model", self.config.model),
            usage=ChatUsage(),
            finish_reason="stop",
        )

    def _raise_wrapped(self, error: Exception):
        """Map OpenAI errors to DeepSeek-specific errors."""
        import openai

        if isinstance(error, openai.AuthenticationError):
            raise DeepSeekAuthError(str(error)) from error
        if isinstance(error, openai.RateLimitError):
            raise DeepSeekRateLimitError(str(error)) from error
        raise DeepSeekError(str(error)) from error


# ── Response types ───────────────────────────────────────────────────────────

@dataclass
class ChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    text: str = ""
    model: str = ""
    usage: ChatUsage = field(default_factory=ChatUsage)
    finish_reason: Optional[str] = None
