# Semantic HGR System - Production Makefile
# =========================================
# Standardized workflows for development, testing, and production operations
#
# Usage:
#   make install          - Install package in editable mode
#   make dev-install      - Install with development dependencies
#   make clip-check       - Validate CLIP integration
#   make test             - Run full test suite
#   make calibration      - Run calibration pipeline
#   make demo             - Launch real-time demo
#   make clean            - Remove temporary artifacts

.PHONY: help install dev-install clip-check calibration-interactive calibration-evaluate realtime-demo test clean

# Default target - show help
help:
	@echo ""
	@echo "Semantic HGR System - Make targets"
	@echo "===================================="
	@echo ""
	@echo "Installation:"
	@echo "  make install           Install package (production dependencies)"
	@echo "  make dev-install       Install with dev dependencies (pytest, black, etc.)"
	@echo ""
	@echo "Validation:"
	@echo "  make clip-check        Verify CLIP integration (embedding dims, probabilities)"
	@echo "  make test              Run full pytest suite (54 tests)"
	@echo ""
	@echo "Production Tools:"
	@echo "  make calibration-prep  Prepare calibration dataset (synthetic)"
	@echo "  make calibration-run   Run calibration study"
	@echo "  make calibration       Full calibration pipeline (prep + run)"
	@echo "  make overlap-analysis  Analyze intent semantic overlap"
	@echo "  make demo              Launch real-time gesture demo"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean             Remove temporary files and caches"
	@echo "  make clean-all         Deep clean (including pip cache)"
	@echo ""

# ============================================================================
# INSTALLATION TARGETS
# ============================================================================

install:
	@echo "Installing semantic-hgr package..."
	pip install -e .
	@echo "✓ Installation complete"
	@echo "  Available commands: prepare-calibration-dataset, calibration-study, realtime-demo"

dev-install:
	@echo "Installing semantic-hgr with development dependencies..."
	pip install -e ".[dev]"
	@echo "✓ Development installation complete"
	@echo "  Dev tools: pytest, black, flake8, mypy"

# ============================================================================
# VALIDATION TARGETS
# ============================================================================

clip-check:
	@echo ""
	@echo "======================================================================="
	@echo "CLIP INTEGRATION VALIDATION"
	@echo "======================================================================="
	@echo ""
	@echo "Testing CLIP encoder initialization..."
	@python -c "from semantic_hgr.representation.clip_encoder import CLIPEncoder; \
	            encoder = CLIPEncoder(); \
	            print(f'✓ CLIPEncoder initialized: {encoder.embedding_dim}-dim embeddings')"
	@echo ""
	@echo "Testing embedding generation..."
	@python -c "import numpy as np; \
	            from semantic_hgr.representation.clip_encoder import CLIPEncoder; \
	            encoder = CLIPEncoder(); \
	            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8); \
	            emb = encoder.encode_image(img); \
	            assert emb.shape == (512,), f'Expected (512,), got {emb.shape}'; \
	            assert np.abs(np.linalg.norm(emb) - 1.0) < 1e-5, 'Embedding not normalized'; \
	            print(f'✓ Embedding shape: {emb.shape}'); \
	            print(f'✓ L2 norm: {np.linalg.norm(emb):.6f} (expected: 1.0)')"
	@echo ""
	@echo "Testing semantic intent extraction..."
	@python -c "import numpy as np; \
	            from semantic_hgr.representation.clip_encoder import CLIPEncoder; \
	            from semantic_hgr.semantics.clip_intent import SemanticIntentExtractor; \
	            encoder = CLIPEncoder(); \
	            extractor = SemanticIntentExtractor(clip_model=encoder.model, device=encoder.device); \
	            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8); \
	            emb = encoder.encode_image(img); \
	            results = extractor.extract_intent(emb, top_k=5); \
	            assert len(results) == 5, f'Expected 5 results, got {len(results)}'; \
	            assert all(0 <= conf <= 1.0 for _, conf in results), 'Confidence not in [0,1]'; \
	            total_conf = sum(conf for _, conf in results); \
	            print(f'✓ Top-5 intents extracted'); \
	            print(f'✓ Confidence range: [{results[-1][1]:.3f}, {results[0][1]:.3f}]'); \
	            print(f'✓ Top intent: {results[0][0]} ({results[0][1]:.3f})')"
	@echo ""
	@echo "Testing probability normalization..."
	@python -c "import numpy as np; \
	            from semantic_hgr.representation.clip_encoder import CLIPEncoder; \
	            from semantic_hgr.semantics.clip_intent import SemanticIntentExtractor; \
	            encoder = CLIPEncoder(); \
	            extractor = SemanticIntentExtractor(clip_model=encoder.model, device=encoder.device); \
	            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8); \
	            emb = encoder.encode_image(img); \
	            results = extractor.extract_intent(emb, top_k=19, min_confidence=0.0); \
	            probs = [conf for _, conf in results]; \
	            total = sum(probs); \
	            assert abs(total - 1.0) < 1e-5, f'Probabilities sum to {total:.6f}, expected 1.0'; \
	            print(f'✓ Probability sum: {total:.6f} (expected: 1.0)')"
	@echo ""
	@echo "✓ All CLIP validation checks passed"
	@echo "======================================================================="

test:
	@echo "Running pytest test suite..."
	pytest tests/ -v --tb=short
	@echo "✓ Test suite complete"

# ============================================================================
# PRODUCTION CALIBRATION WORKFLOW
# ============================================================================

calibration-prep:
	@echo "Preparing synthetic calibration dataset..."
	@mkdir -p data/calibration
	prepare-calibration-dataset \
	    --synthetic \
	    --output data/calibration/validation_gestures_synthetic.json \
	    --samples 100
	@echo "✓ Synthetic dataset created: data/calibration/validation_gestures_synthetic.json"

calibration-run:
	@echo "Running calibration study..."
	@mkdir -p results/calibration
	calibration-study \
	    --dataset data/calibration/validation_gestures_synthetic.json \
	    --output results/calibration/calibration_report.json \
	    --num-bins 10 \
	    --plot results/calibration/reliability_diagram.png
	@echo "✓ Calibration report: results/calibration/calibration_report.json"

calibration: calibration-prep calibration-run
	@echo ""
	@echo "✓ Full calibration pipeline complete"
	@echo "  Dataset: data/calibration/validation_gestures_synthetic.json"
	@echo "  Report:  results/calibration/calibration_report.json"

overlap-analysis:
	@echo "Running intent overlap analysis..."
	@mkdir -p results/overlap
	intent-overlap-analysis \
	    --dataset data/calibration/validation_gestures_synthetic.json \
	    --output results/overlap/overlap_analysis.json \
	    --plot-similarity results/overlap/similarity_matrix.png \
	    --plot-confusion results/overlap/confusion_matrix.png \
	    --similarity-threshold 0.85
	@echo "✓ Overlap analysis: results/overlap/overlap_analysis.json"

# ============================================================================
# REAL-TIME DEMO
# ============================================================================

demo:
	@echo "Launching real-time gesture recognition demo..."
	@echo "Controls:"
	@echo "  Q / ESC - Quit"
	@echo "  P       - Pause/unpause"
	@echo "  S       - Save frame"
	@echo "  T       - Toggle performance overlay"
	@echo ""
	realtime-demo

# ============================================================================
# VALIDATION SCRIPTS (INDIVIDUAL)
# ============================================================================

validate-input:
	@echo "Validating input layer (camera + hand detection)..."
	validate-input-layer

validate-hand:
	@echo "Validating hand detection..."
	validate-hand-detection

validate-clip:
	@echo "Validating CLIP integration..."
	validate-clip-integration

validate-all: validate-input validate-hand validate-clip
	@echo "✓ All validation checks complete"

# ============================================================================
# CLEANUP TARGETS
# ============================================================================

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	rm -rf .coverage htmlcov/
	rm -f /tmp/test_calibration.json /tmp/calibration_report.json
	@echo "✓ Cleanup complete"

clean-all: clean
	@echo "Deep cleaning (pip cache)..."
	pip cache purge
	@echo "✓ Deep clean complete"

# ============================================================================
# DEVELOPMENT HELPERS
# ============================================================================

format:
	@echo "Formatting code with black..."
	black src/ tests/
	@echo "✓ Code formatted"

lint:
	@echo "Linting code with flake8..."
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	@echo "✓ Lint checks complete"

typecheck:
	@echo "Type checking with mypy..."
	mypy src/semantic_hgr --ignore-missing-imports
	@echo "✓ Type checks complete"

# ============================================================================
# END OF MAKEFILE
# ============================================================================
