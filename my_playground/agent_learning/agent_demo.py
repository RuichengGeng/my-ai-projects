"""
Minimal single-agent loop, built from scratch with nothing but `requests`
and `pydantic` for structured output.

No LangChain, no Assistants API, no Agent SDK -- just raw HTTP calls to the
Claude Messages API and a hand-written while-loop. The point is to make the
mechanics from our conversation concrete:

    1. Send context (system + tool schemas + messages) to the model.
    2. Model returns text and/or a tool-call request. The model does NOT
       execute anything -- it only ever produces text describing what it
       wants to do.
    3. This code (the "orchestrator") notices the tool-call request, runs
       the real Python function, and gets a real result.
    4. The result is appended to the conversation as a "tool_result", and
       we go back to step 1.
    5. Loop ends when the model responds without asking for a tool.
    6. run_agent() returns a pydantic AgentResult, which can be handed to
       another function directly or serialized with .model_dump_json().

Also demonstrates Anthropic prompt caching: the (system + tools) prefix is
identical every step of the loop, so we mark it `cache_control: ephemeral`
and reuse the server-side KV cache across turns.

Setup:
    pip install requests pydantic
    export ANTHROPIC_API_KEY=sk-ant-...
    python agent_demo.py
"""

from __future__ import annotations

import ast
import operator
import os
import json
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field

API_URL = "https://api.anthropic.com/v1/messages"
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-5"   # swap for whatever current model string you have access to
MAX_LOOP_STEPS = 8          # a hard safety cap -- always bound your loops

SYSTEM_PROMPT = (
    "You are a small, careful agent. Use the provided tools whenever a "
    "question requires real computation or lookup. When you are fully done, "
    "reply with a final natural-language answer."
)


# ---------------------------------------------------------------------------
# 1. Tools. Each tool is just a normal Python function plus a JSON schema
#    describing it, so the model knows it exists and how to call it.
# ---------------------------------------------------------------------------

def calculator(expression: str) -> float:
    """Safely evaluate basic arithmetic (+ - * / ** parens), no eval()."""
    _OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {expression}")

    return _eval(ast.parse(expression, mode="eval").body)


def get_weather(city: str) -> dict:
    """Mocked weather lookup -- stands in for a real API call."""
    fake_data = {
        "tokyo": {"temp_f": 84, "conditions": "humid, partly cloudy"},
        "london": {"temp_f": 61, "conditions": "light rain"},
    }
    return fake_data.get(
        city.lower(),
        {"temp_f": 70, "conditions": "unknown city, made this up"},
    )


TOOL_FUNCTIONS = {
    "calculator": lambda input_: calculator(input_["expression"]),
    "get_weather": lambda input_: get_weather(input_["city"]),
}

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression, e.g. '47 * 89'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city (demo data, not real).",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
]


# ---------------------------------------------------------------------------
# 2. Pydantic models for the structured result. This is what run_agent()
#    returns, so downstream code can consume it as typed objects or JSON.
# ---------------------------------------------------------------------------

class ToolCallRecord(BaseModel):
    step: int
    tool: str
    input: dict[str, Any]
    output: Any


class UsageRecord(BaseModel):
    """Token accounting. Cache-hit fields show prompt caching in action."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class AgentResult(BaseModel):
    status: Literal["ok", "max_steps_reached"] = "ok"
    final_text: str | None = None
    steps: int = 0
    tool_trace: list[ToolCallRecord] = Field(default_factory=list)
    usage: UsageRecord = Field(default_factory=UsageRecord)
    # Full transcript, kept so callers can debug or replay. Omit from dumps
    # by using .model_dump(exclude={"messages"}) if you don't want it.
    messages: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 3. Raw API call. One HTTP request in, one JSON response out. The (system
#    + tools) prefix is marked `cache_control: ephemeral` so Anthropic can
#    reuse the KV cache for it across every step of the loop.
# ---------------------------------------------------------------------------

def call_claude(messages: list[dict]) -> dict:
    # cache_control on the last block of `system` and the last tool schema
    # tells the server "everything up to and including this is cacheable".
    system_blocks = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    tools = [dict(t) for t in TOOL_SCHEMAS]
    tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}

    response = requests.post(
        API_URL,
        headers={
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": 1024,
            "system": system_blocks,
            "tools": tools,
            "messages": messages,
        },
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# 4. Orchestrator loop. Returns an AgentResult; no printing required.
# ---------------------------------------------------------------------------

def run_agent(user_goal: str, verbose: bool = False) -> AgentResult:
    messages: list[dict] = [{"role": "user", "content": user_goal}]
    result = AgentResult(messages=messages)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    for step in range(1, MAX_LOOP_STEPS + 1):
        _log(f"\n--- step {step} ---")
        api_result = call_claude(messages)
        content_blocks = api_result["content"]
        messages.append({"role": "assistant", "content": content_blocks})

        # Accumulate token usage across the loop.
        usage = api_result.get("usage", {})
        result.usage.input_tokens += usage.get("input_tokens", 0)
        result.usage.output_tokens += usage.get("output_tokens", 0)
        result.usage.cache_creation_input_tokens += usage.get(
            "cache_creation_input_tokens", 0
        )
        result.usage.cache_read_input_tokens += usage.get(
            "cache_read_input_tokens", 0
        )

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in content_blocks:
            if block["type"] == "text":
                text_parts.append(block["text"])
                _log(f"[model says]: {block['text']}")
            elif block["type"] == "tool_use":
                tool_calls.append(block)
                _log(f"[model requests tool]: {block['name']}({block['input']})")

        # Terminal turn: model produced a final answer, no tool_use requested.
        if api_result["stop_reason"] != "tool_use":
            result.status = "ok"
            result.final_text = "\n".join(text_parts).strip() or None
            result.steps = step
            return result

        # Execute each requested tool, feed real results back into the loop.
        tool_result_blocks = []
        for call in tool_calls:
            output = TOOL_FUNCTIONS[call["name"]](call["input"])
            _log(f"[tool executed]: {call['name']} -> {output}")
            result.tool_trace.append(
                ToolCallRecord(
                    step=step,
                    tool=call["name"],
                    input=call["input"],
                    output=output,
                )
            )
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": json.dumps(output),
                }
            )

        messages.append({"role": "user", "content": tool_result_blocks})

    result.status = "max_steps_reached"
    result.steps = MAX_LOOP_STEPS
    return result


# ---------------------------------------------------------------------------
# 5. Example downstream consumer of the structured result.
# ---------------------------------------------------------------------------

def summarize(result: AgentResult) -> dict:
    """Toy example of another function eating the agent's structured output."""
    return {
        "answer": result.final_text,
        "tool_count": len(result.tool_trace),
        "tokens_in": result.usage.input_tokens,
        "tokens_out": result.usage.output_tokens,
        "cache_read": result.usage.cache_read_input_tokens,
    }


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("Set ANTHROPIC_API_KEY first.")

    result = run_agent(
        "What's 47 * 89? Also check the weather in Tokyo. "
        "Then tell me which number is bigger: your calculation result "
        "or the Tokyo temperature in Fahrenheit.",
        verbose=True,
    )

    print("\n=== AgentResult as JSON ===")
    # exclude the full transcript for readability; drop `exclude=` to see it all.
    print(result.model_dump_json(indent=2, exclude={"messages"}))

    print("\n=== downstream summary ===")
    print(json.dumps(summarize(result), indent=2))
