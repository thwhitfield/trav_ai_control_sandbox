.PHONY: install dev clean test lint format check run help

help:
	@echo "Available targets:"
	@echo "  install  - Install project dependencies"
	@echo "  dev      - Install with dev dependencies"
	@echo "  clean    - Remove build artifacts and caches"
	@echo "  test     - Run tests"
	@echo "  lint     - Run linter"
	@echo "  format   - Format code"
	@echo "  check    - Run linter and type checks"
	@echo "  run      - Run the application"

install:
	uv sync

dev:
	uv sync --all-extras

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

check: lint
	uv run mypy .

