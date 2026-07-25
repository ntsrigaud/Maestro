.PHONY: help setup-build-env test test-all lint typecheck format run-ui build build-onefile generate-icon clean

help: ## Show this help message
	@echo "Maestro — available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	    | sort \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup-build-env: ## Install PyInstaller build dependencies (editable)
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

run-ui: ## Launch PySide6 GUI
	python scripts/run_ui.py

# ── Application bundling ──────────────────────────────────────────────────────

build: clean setup-build-env ## Build standalone app bundle via PyInstaller
	@echo "Downloading offline model from Hugging Face Hub..."
	python scripts/download_bundled_model.py
	@echo "Building Maestro standalone application..."
	pyinstaller --clean --noconfirm maestro.spec
	@echo ""
	@if [ "$$(uname)" = "Darwin" ]; then \
	    echo "Code signing application..."; \
	    codesign --force --deep --sign - dist/Maestro.app; \
	    echo "Creating DMG archive..."; \
	    rm -f dist/*.dmg; \
	    create-dmg --overwrite --no-code-sign dist/Maestro.app dist/ || true; \
	    mv dist/Maestro\ *.dmg dist/Maestro-macos.dmg; \
	    echo "Cleaning up intermediate build directory..."; \
	    rm -rf dist/Maestro dist/Maestro.app; \
	    echo "✓ macOS bundle: dist/Maestro-macos.dmg"; \
	else \
	    echo "✓ Executable: dist/Maestro/"; \
	fi

build-onefile: setup-build-env ## Build a single-file executable (slower startup)
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
