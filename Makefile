.PHONY: install test lint format run-cli run-ui clean

install: ## Install dev + training extras (editable)
	pip install -e ".[dev,training]"

test: ## Run the full test suite
	pytest --verbose

lint: ## Static analysis: ruff check + mypy
	ruff check .
	mypy src

format: ## Auto-format: ruff (replaces black + isort)
	ruff format .
	ruff check --fix .

run-cli: ## Launch headless inference pipeline
	python scripts/run_realtime.py --config configs

run-ui: ## Launch PySide6 GUI
	python scripts/run_ui.py

clean: ## Remove Python caches and build artifacts
	find . -type d \( -name __pycache__ -o -name "*.egg-info" -o -name .mypy_cache -o -name .ruff_cache \) -exec rm -rf {} +

# Credits: https://www.kdnuggets.com/the-case-for-makefiles-in-python-projects-and-how-to-get-started
