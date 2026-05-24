"""Verify all submodules and core dependencies import correctly."""
import sys

errors = []

def try_import(name, import_stmt=None):
    try:
        if import_stmt:
            exec(import_stmt)
        else:
            __import__(name)
        print(f"  ✅ {name}")
    except Exception as e:
        errors.append(f"  ❌ {name}: {e}")
        print(f"  ❌ {name}: {e}")


if __name__ == "__main__":
    print("── Submodule imports ──")
    try_import("Kronos", "from model.kronos import Kronos, KronosTokenizer")
    try_import("tradingagents", "import tradingagents")

    print("\n── Core dependencies ──")
    try_import("torch")
    try_import("pandas")
    try_import("numpy")
    try_import("langgraph")
    try_import("backtrader")
    try_import("yfinance")
    try_import("langchain_core")
    try_import("matplotlib")

    print(f"\n── Versions ──")
    import torch; print(f"  Torch:     {torch.__version__}")
    import pandas; print(f"  Pandas:    {pandas.__version__}")
    import numpy; print(f"  NumPy:     {numpy.__version__}")

    if errors:
        print(f"\n❌ {len(errors)} error(s) found")
        sys.exit(1)
    else:
        print("\n✅ All imports OK")
