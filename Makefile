.PHONY: install install-dev install-build test lint typecheck format run-cli run-ui build build-onefile clean

install: ## Install UI extras for end-user use (editable)
	pip install -e ".[ui]"

install-dev: ## Install dev + training extras (editable)
	pip install -e ".[dev,training]"

install-build: ## Install UI + PyInstaller extras (replaces opencv-headless)
	pip uninstall -y opencv-python-headless 2>/dev/null || true
	pip install -e ".[ui,build]"

test: ## Run the non-training test suite
	pytest -m "not training"

test-all: ## Run the full test suite (requires training extras)
	pytest --verbose

lint: ## Static analysis: ruff check + mypy
	ruff check .
	mypy src

typecheck: ## Type-check only
	mypy src

format: ## Auto-format: ruff (replaces black + isort)
	ruff format .
	ruff check --fix .

run-cli: ## Launch headless inference pipeline
	python scripts/run_realtime.py

run-ui: ## Launch PySide6 GUI
	python scripts/run_ui.py

# ── Application bundling ──────────────────────────────────────────────────────

build: install-build ## Build standalone app bundle via PyInstaller
	@echo "Building Maestro standalone application..."
	pyinstaller --noconfirm maestro.spec
	@echo ""
	@if [ "$$(uname)" = "Darwin" ]; then \
	    echo "✓ macOS bundle: dist/Maestro.app"; \
	    echo "  Launch: open dist/Maestro.app"; \
	else \
	    echo "✓ Executable: dist/Maestro/"; \
	fi

build-onefile: install-build ## Build a single-file executable (slower startup)
	pyinstaller --noconfirm --onefile --windowed --name Maestro \
	    src/maestro/app/ui/app_ui.py
	@echo "✓ Output: dist/Maestro"

generate-icon: ## Re-generate icon assets (PNG / ICNS / ICO) from tools/generate_icon.py
	python tools/generate_icon.py

clean: ## Remove Python caches, build artifacts, and icon scratch files
	rm -rf build/ dist/
	find . -type d \( -name __pycache__ -o -name "*.egg-info" -o -name .mypy_cache -o -name .ruff_cache \) -exec rm -rf {} +
	rm -f docs/assets/icon_preview.png docs/assets/icon.iconset

# Credits: https://www.kdnuggets.com/the-case-for-makefiles-in-python-projects-and-how-to-get-started
