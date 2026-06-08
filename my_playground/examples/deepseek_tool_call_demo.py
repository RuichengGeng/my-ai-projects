"""
DeepSeek tool-calling demo.

This script shows the same core loop used by TradingAgents analyst nodes:

1. Send the user question plus tool schemas to the LLM.
2. The LLM replies with tool_calls instead of a final answer.
3. Python executes those local functions.
4. Tool results are appended to the message history.
5. The LLM sees the tool results and writes the final answer.

Run:
    export DEEPSEEK_API_KEY="sk-..."
    make run CMD="python my_playground/examples/deepseek_tool_call_demo.py"

Try a custom question:
    make run CMD='python my_playground/examples/deepseek_tool_call_demo.py --question "Use tools to compare NVDA and AAPL P/E ratios."'
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


# Local fake market data keeps the demo deterministic. The LLM cannot see this
# data unless it asks for a tool call and we send the result back.
MOCK_MARKET_DATA = {
    "NVDA": {"price": 125.20, "eps_ttm": 2.13, "currency": "USD"},
    "AAPL": {"price": 190.40, "eps_ttm": 6.43, "currency": "USD"},
    "TSLA": {"price": 177.80, "eps_ttm": 3.10, "currency": "USD"},
}


def get_mock_market_data(ticker: str) -> dict[str, Any]:
    """Return mock price and EPS data for a ticker."""
    symbol = ticker.upper()
    if symbol not in MOCK_MARKET_DATA:
        return {
            "ticker": symbol,
            "error": f"No mock data for {symbol}. Try NVDA, AAPL, or TSLA.",
        }
    return {"ticker": symbol, **MOCK_MARKET_DATA[symbol]}


def calculate_pe_ratio(price: float, eps_ttm: float) -> dict[str, Any]:
    """Calculate trailing price-to-earnings ratio."""
    if eps_ttm == 0:
        return {"error": "EPS is zero, so P/E ratio is undefined."}
    return {"pe_ratio": round(price / eps_ttm, 2)}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_mock_market_data",
            "description": "Get mock price, trailing EPS, and currency for a stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, for example NVDA or AAPL.",
                    }
                },
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_pe_ratio",
            "description": "Calculate trailing P/E ratio from price and trailing EPS.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "Current stock price."},
                    "eps_ttm": {
                        "type": "number",
                        "description": "Trailing twelve month earnings per share.",
                    },
                },
                "required": ["price", "eps_ttm"],
                "additionalProperties": False,
            },
        },
    },
]


TOOL_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "get_mock_market_data": get_mock_market_data,
    "calculate_pe_ratio": calculate_pe_ratio,
}


def load_repo_env() -> None:
    """Load simple KEY=VALUE pairs from the repo .env when the IDE has not."""
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def run_tool_call_demo(question: str, model: str) -> None:
    load_repo_env()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY. Export it before running this demo.")

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a finance assistant. Use the provided tools whenever "
                "you need market data or calculations. Do not invent prices or EPS."
            ),
        },
        {"role": "user", "content": question},
    ]

    print("\nUSER QUESTION")
    print(question)

    for turn in range(1, 6):
        print(f"\n--- LLM CALL {turn} ---")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            temperature=0,
        )
        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if assistant_message.content:
            print("Assistant content:")
            print(assistant_message.content)

        if not tool_calls:
            print("\nFINAL ANSWER")
            print(assistant_message.content or "")
            return

        print("Tool calls requested by the LLM:")
        for call in tool_calls:
            print(f"- {call.function.name}({call.function.arguments})")

        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [call.model_dump() for call in tool_calls],
            }
        )

        for call in tool_calls:
            tool_name = call.function.name
            tool_args = json.loads(call.function.arguments or "{}")
            tool_func = TOOL_DISPATCH[tool_name]
            tool_result = tool_func(**tool_args)

            print(f"Python executed {tool_name}:")
            print(json.dumps(tool_result, indent=2))

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": tool_name,
                    "content": json.dumps(tool_result),
                }
            )

    raise SystemExit("Stopped after 5 LLM calls to avoid an infinite tool loop.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek tool-calling demo.")
    parser.add_argument(
        "--question",
        default=(
            "Use tools to find NVDA's mock trailing P/E ratio. "
            "Show the price, EPS, and final P/E calculation."
        ),
    )
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tool_call_demo(args.question, args.model)
