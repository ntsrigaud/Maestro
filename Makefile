# Semantic HGR System — Architecture A Makefile
# ================================================
# Dual-Stream Supervised Gesture Recognition
# GCN (MediaPipe 21-keypoint graph) + ViT-B/16 + Cross-Attention Fusion
#
# Phases:
#   1. extract-keypoints    Offline MediaPipe landmark extraction
#   2. validate-keypoints   Pre-training gate (Section 2.6)
#   3. train-gcn            Phase 1: GCN standalone (target >=65% val Top-1)
#   4. train-rgb            Phase 2: ViT-B/16 baseline (target >=88% val Top-1)
#   5. train-dual           Phase 3+4: Cross-attention fusion + end-to-end tune
#   6. eval                 Full evaluation (Top-K, ECE, McNemar's, confusion)

.PHONY: help install dev-install test test-fast \
        validate-input validate-hand validate-keypoints extract-keypoints \
        train-gcn train-rgb train-dual eval \
        download-hagrid \
        clean clean-all format lint typecheck

# ============================================================================
# DEFAULT
# ============================================================================

help:
	@echo ""
	@echo "Semantic HGR — Architecture A"
	@echo "================================"
	@echo ""
	@echo "Installation:"
	@echo "  make install              Install package (production)"
	@echo "  make dev-install          Install with dev dependencies"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make download-hagrid      Download HaGRID 8-class subset"
	@echo "  make extract-keypoints    Offline MediaPipe keypoint extraction"
	@echo "  make validate-keypoints   Pre-training validation gate (Sec 2.6)"
	@echo ""
	@echo "Training (4-phase Arch A protocol):"
	@echo "  make train-gcn            Phase 1  GCN geometry stream"
	@echo "  make train-rgb            Phase 2  ViT-B/16 RGB stream"
	@echo "  make train-dual           Phase 3+4 Dual-stream fusion"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval                 Top-K, ECE, McNemar, confusion matrix"
	@echo "  make test                 Unit test suite"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format | lint | typecheck"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean | clean-all"
	@echo ""

# ============================================================================
# INSTALLATION
# ============================================================================

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

# ============================================================================
# DATA PIPELINE
# ============================================================================

download-hagrid:
	python scripts/download_hagrid_subset.py

extract-keypoints:
	@echo "Extracting MediaPipe 21-keypoint landmarks from HaGRID subset..."
	python -m semantic_hgr.data.extract_keypoints \
		--input-dir data/hagrid_subset \
		--output-dir data/keypoints \
		--split-dir data/splits
	@echo "Keypoint extraction complete -> data/keypoints/"

validate-keypoints:
	@echo "Running pre-training validation checklist (Architecture A Section 2.6)..."
	python -m semantic_hgr.scripts.validate_hand_detection \
		--data-dir data/hagrid_subset \
		--keypoint-dir data/keypoints
	@echo "Validation complete"

validate-input: validate-keypoints

validate-hand:
	python -m semantic_hgr.scripts.validate_hand_detection

# ============================================================================
# TRAINING  — Architecture A 4-phase protocol
# ============================================================================
# Phase 1: GCN standalone
#   Input : (21, 3) keypoint graph per sample
#   Output: 8-class logits
#   Gate  : >=65% val Top-1 before proceeding

train-gcn:
	@echo "Phase 1: Training GCN geometry stream..."
	python -m semantic_hgr.training.train_gcn --config configs/arch_a_gcn.yaml
	@echo "GCN training complete. Check checkpoints/gcn/best.pt"

# Phase 2: ViT-B/16 RGB baseline
#   Input : 224x224 RGB crops via center_crop_frame + ImageNet normalisation
#   Output: 8-class logits
#   Gate  : >=88% val Top-1 before proceeding

train-rgb:
	@echo "Phase 2: Training ViT-B/16 RGB stream..."
	python -m semantic_hgr.training.train_rgb --config configs/arch_a_rgb.yaml
	@echo "RGB training complete. Check checkpoints/rgb/best.pt"

# Phase 3+4: Cross-attention fusion, then end-to-end fine-tune
#   Requires Phase 1 and Phase 2 checkpoints

train-dual:
	@test -f checkpoints/gcn/best.pt || \
		(echo "ERROR: checkpoints/gcn/best.pt not found. Run 'make train-gcn' first." && exit 1)
	@test -f checkpoints/rgb/best.pt || \
		(echo "ERROR: checkpoints/rgb/best.pt not found. Run 'make train-rgb' first." && exit 1)
	@echo "Phase 3+4: Training dual-stream fusion model..."
	python -m semantic_hgr.training.train_dual \
		--config configs/arch_a_dual.yaml \
		--gcn-ckpt checkpoints/gcn/best.pt \
		--rgb-ckpt checkpoints/rgb/best.pt
	@echo "Dual-stream training complete. Check checkpoints/dual/best.pt"

# ============================================================================
# EVALUATION
# ============================================================================

eval:
	@echo "Running full evaluation suite..."
	python -m semantic_hgr.evaluation.evaluate \
		--config configs/arch_a_dual.yaml \
		--checkpoint checkpoints/dual/best.pt \
		--output-dir results/
	@echo "Evaluation complete. Results in results/"

# ============================================================================
# TESTS
# ============================================================================

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -x --tb=short

# ============================================================================
# CODE QUALITY
# ============================================================================

format:
	ruff format src/ scripts/ tests/

lint:
	ruff check src/ scripts/ tests/

typecheck:
	mypy src/semantic_hgr/ --ignore-missing-imports

# ============================================================================
# CLEAN
# ============================================================================

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	@echo "Cleaned temporary files"

clean-all: clean
	pip cache purge 2>/dev/null || true
	@echo "Deep clean complete"
