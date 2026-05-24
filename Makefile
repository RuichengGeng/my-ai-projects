.PHONY: run shell sync clean test-imports

# ─── Environment ───────────────────────────────────────────────
.venv:
	uv sync

sync: .venv
	uv sync

clean:
	rm -rf .venv *.egg-info dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

# ─── Commands (sets PYTHONPATH so Kronos resolves) ─────────────
run:
	PYTHONPATH=kronos uv run $(CMD)

shell:
	PYTHONPATH=kronos uv run python

# ─── Verify both projects import cleanly ───────────────────────
test-imports:
	PYTHONPATH=kronos uv run python scripts/check_imports.py
