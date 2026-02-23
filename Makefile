.PHONY: install install-dev test test-cov lint format clean help \
       docker docker-run docker-up docker-down docker-shell \
       uv-sync uv-sync-all uv-lock uv-test uv-lint uv-format uv-run

# Default target
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ===========================================================================
# pip-based targets (classic)
# ===========================================================================

install:  ## [pip] Install the package in editable mode
	pip install -e .

install-dev:  ## [pip] Install with all dev dependencies
	pip install -e ".[dev,all]"
	pre-commit install

test:  ## Run unit tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage report
	pytest tests/ -v --cov=vembed --cov-report=term-missing --cov-report=html
	@echo "HTML coverage report: htmlcov/index.html"

lint:  ## Run all linters
	ruff check vembed tests
	black --check --line-length=100 vembed tests
	isort --check-only --profile black --line-length=100 vembed tests

format:  ## Auto-format code
	black --line-length=100 vembed tests examples
	isort --profile black --line-length=100 vembed tests examples
	ruff check --fix vembed tests examples

typecheck:  ## Run mypy type checking
	mypy vembed/trainer.py vembed/inference.py vembed/model/modeling.py --ignore-missing-imports

clean:  ## Remove build artifacts
	rm -rf build dist *.egg-info htmlcov .coverage coverage.xml .mypy_cache .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

docker:  ## Build Docker image
	docker build -t vembed-factory .

docker-run:  ## Run Docker container with GPU
	docker run --gpus all -it -v $(PWD)/data:/app/data vembed-factory bash

docker-up:  ## Start container via docker compose (detached)
	docker compose up -d

docker-down:  ## Stop and remove docker compose containers
	docker compose down

docker-shell:  ## Enter running docker compose container
	docker compose exec vembed bash

benchmark:  ## Run benchmark suite
	python examples/benchmark/run.py

demo:  ## Launch Gradio demo app
	python examples/app.py

release-check:  ## Verify package builds correctly
	pip install build twine
	python -m build
	twine check dist/*

# ===========================================================================
# uv-based targets (recommended)
# ===========================================================================

uv-sync:  ## [uv] Sync project virtualenv (install deps + dev group)
	uv sync

uv-sync-all:  ## [uv] Sync with all optional extras
	uv sync --all-extras

uv-lock:  ## [uv] Regenerate uv.lock lockfile
	uv lock

uv-test:  ## [uv] Run tests via uv
	uv run pytest tests/ -v

uv-test-cov:  ## [uv] Run tests with coverage via uv
	uv run pytest tests/ -v --cov=vembed --cov-report=term-missing --cov-report=html

uv-lint:  ## [uv] Run all linters via uv
	uv run ruff check vembed tests
	uv run black --check --line-length=100 vembed tests
	uv run isort --check-only --profile black --line-length=100 vembed tests

uv-format:  ## [uv] Auto-format code via uv
	uv run black --line-length=100 vembed tests examples
	uv run isort --profile black --line-length=100 vembed tests examples
	uv run ruff check --fix vembed tests examples

uv-typecheck:  ## [uv] Run mypy type checking via uv
	uv run mypy vembed/trainer.py vembed/inference.py vembed/model/modeling.py --ignore-missing-imports

uv-run:  ## [uv] Run arbitrary command (usage: make uv-run CMD="python train.py")
	uv run $(CMD)

uv-build:  ## [uv] Build package
	uv build

uv-clean:  ## [uv] Clean all artifacts including .venv
	rm -rf build dist *.egg-info htmlcov .coverage coverage.xml .mypy_cache .pytest_cache .venv
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
