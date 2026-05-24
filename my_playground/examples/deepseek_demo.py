"""
DeepSeek client demo.

Run:
    export DEEPSEEK_API_KEY="sk-..."
    make run CMD="python my-playground/examples/deepseek_demo.py"
"""

from my_playground.deepseek_client import DeepSeekClient, DeepSeekConfig

# ── Option 1: use DEEPSEEK_API_KEY env var ────────────────────
client = DeepSeekClient()

# ── Option 2: pass key explicitly ─────────────────────────────
# config = DeepSeekConfig(api_key="sk-...", model="deepseek-reasoner")
# client = DeepSeekClient(config)

# ── Basic chat ────────────────────────────────────────────────
response = client.chat(
    "What is the price-to-earnings ratio and how is it used in stock analysis?",
    system_prompt="You are a financial analyst assistant.",
)

print(f"Model:     {response.model}")
print(f"Tokens:    {response.usage.total_tokens}")
print(f"Response:\n{response.text}")
