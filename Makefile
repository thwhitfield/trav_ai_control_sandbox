.PHONY: help install upgrade lint format clean view

help:
	@echo "Available targets:"
	@echo "  install  - Install dependencies (incl. dev group)"
	@echo "  upgrade  - Upgrade all dependencies to latest and re-sync"
	@echo "  lint     - Run ruff linter"
	@echo "  format   - Format code with ruff"
	@echo "  clean    - Remove caches"
	@echo "  view     - Open the Inspect log viewer"

install:
	uv sync

upgrade:
	uv lock --upgrade
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

view:
	uv run inspect view
