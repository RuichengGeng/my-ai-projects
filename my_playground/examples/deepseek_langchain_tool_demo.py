"""
DeepSeek tool-calling demo with LangChain / LangGraph.

Compare this with deepseek_tool_call_demo.py:

- Raw API demo: you write the OpenAI-compatible tool JSON schemas yourself.
- LangChain demo: @tool creates schemas from Python functions, and
  llm.bind_tools(tools) sends those schemas to the model.
- LangGraph agent mode: the framework also runs the tool loop for you.

Run manual mode:
    .venv/bin/python my_playground/examples/deepseek_langchain_tool_demo.py

Run LangGraph agent mode:
    .venv/bin/python my_playground/examples/deepseek_langchain_tool_demo.py --mode agent
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


MOCK_MARKET_DATA = {
    "NVDA": {"price": 125.20, "eps_ttm": 2.13, "currency": "USD"},
    "AAPL": {"price": 190.40, "eps_ttm": 6.43, "currency": "USD"},
    "TSLA": {"price": 177.80, "eps_ttm": 3.10, "currency": "USD"},
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


@tool
def get_mock_market_data(ticker: str) -> dict[str, Any]:
    """Get mock price, trailing EPS, and currency for a stock ticker."""
    symbol = ticker.upper()
    if symbol not in MOCK_MARKET_DATA:
        return {
            "ticker": symbol,
            "error": f"No mock data for {symbol}. Try NVDA, AAPL, or TSLA.",
        }
    return {"ticker": symbol, **MOCK_MARKET_DATA[symbol]}


@tool
def calculate_pe_ratio(price: float, eps_ttm: float) -> dict[str, Any]:
    """Calculate trailing price-to-earnings ratio from price and trailing EPS."""
    if eps_ttm == 0:
        return {"error": "EPS is zero, so P/E ratio is undefined."}
    return {"pe_ratio": round(price / eps_ttm, 2)}


TOOLS = [get_mock_market_data, calculate_pe_ratio]
TOOLS_BY_NAME = {tool_.name: tool_ for tool_ in TOOLS}


def create_deepseek_llm(model: str) -> ChatOpenAI:
    load_repo_env()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY. Add it to .env or export it.")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
        temperature=0,
    )


def run_manual_langchain_loop(question: str, model: str) -> None:
    """Show LangChain's bind_tools, while still executing the loop ourselves."""
    llm = create_deepseek_llm(model)
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [
        SystemMessage(
            content=(
                "You are a finance assistant. Use the provided tools whenever "
                "you need market data or calculations. Do not invent prices or EPS."
            )
        ),
        HumanMessage(content=question),
    ]

    print("\nMODE")
    print("manual LangChain loop")
    print("\nUSER QUESTION")
    print(question)

    for turn in range(1, 6):
        print(f"\n--- LLM CALL {turn} ---")
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        if ai_message.content:
            print("Assistant content:")
            print(ai_message.content)

        if not ai_message.tool_calls:
            print("\nFINAL ANSWER")
            print(ai_message.content)
            return

        print("Tool calls requested by the LLM:")
        for call in ai_message.tool_calls:
            print(f"- {call['name']}({json.dumps(call['args'])})")

        for call in ai_message.tool_calls:
            selected_tool = TOOLS_BY_NAME[call["name"]]
            tool_result = selected_tool.invoke(call["args"])

            print(f"LangChain executed {call['name']}:")
            print(json.dumps(tool_result, indent=2))

            messages.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )

    raise SystemExit("Stopped after 5 LLM calls to avoid an infinite tool loop.")


def run_langgraph_agent(question: str, model: str) -> None:
    """Let LangGraph run the tool-calling loop for us."""
    llm = create_deepseek_llm(model)
    agent = create_react_agent(llm, TOOLS)

    messages = [
        SystemMessage(
            content=(
                "You are a finance assistant. Use the provided tools whenever "
                "you need market data or calculations. Do not invent prices or EPS."
            )
        ),
        HumanMessage(content=question),
    ]

    print("\nMODE")
    print("LangGraph prebuilt ReAct agent")
    print("\nUSER QUESTION")
    print(question)

    final_state = agent.invoke({"messages": messages})

    print("\nMESSAGES PRODUCED BY THE AGENT")
    for message in final_state["messages"]:
        message_type = message.__class__.__name__
        print(f"\n[{message_type}]")
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print(f"tool_call: {call['name']}({json.dumps(call['args'])})")
        print(message.content)

    print("\nFINAL ANSWER")
    print(final_state["messages"][-1].content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek LangChain tool demo.")
    parser.add_argument(
        "--question",
        default=(
            "Use tools to find NVDA's mock trailing P/E ratio. "
            "Show the price, EPS, and final P/E calculation."
        ),
    )
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--mode",
        choices=["manual", "agent"],
        default="agent",
        help="agent shows bind_tools; agent lets LangGraph run the loop.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "agent":
        run_langgraph_agent(args.question, args.model)
    else:
        run_manual_langchain_loop(args.question, args.model)
