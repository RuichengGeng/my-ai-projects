ROOT := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
.PHONY: run shell sync clean test-imports setup-pth

# ─── Environment ───────────────────────────────────────────────
.venv: setup-pth

sync:
	uv sync
	$(MAKE) setup-pth

setup-pth:
	@echo "Setting up submodule path hooks..."
	@SITE_PKG=$$(uv run python -c "import site; print(site.getsitepackages()[0])"); \
	echo "$(ROOT)/kronos" > "$$SITE_PKG/kronos.pth"; \
	echo "$(ROOT)/ta-agents" > "$$SITE_PKG/ta-agents.pth"; \
	echo "  ✓ kronos.pth"
	@echo "  ✓ ta-agents.pth"

clean:
	rm -rf .venv *.egg-info dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

# ─── Commands (PYTHONPATH as fallback) ─────────────────────────
run:
	PYTHONPATH=kronos uv run $(CMD)

shell:
	PYTHONPATH=kronos uv run python

# ─── Verify both projects import cleanly ───────────────────────
test-imports:
	uv run python scripts/check_imports.py
