#!/usr/bin/env python3
"""
CLIP Semantic Intent Calibration Study

Purpose: Evaluate whether probability scores are well-calibrated and reliable
         for deployment in production semantic gesture recognition system.

Calibration Metrics:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- Reliability Diagram (Confidence vs Accuracy)

Methodology:
- Collect labeled validation dataset
- Bin predictions by confidence
- Compute calibration gaps
- Apply temperature scaling if miscalibrated
- Generate comprehensive calibration report

Usage:
    python scripts/calibration_study.py --dataset data/validation_gestures.json \
                                        --output results/calibration_report.json
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings

from semantic_hgr.representation.clip_encoder import CLIPEncoder
from semantic_hgr.semantics.clip_intent import SemanticIntentExtractor


@dataclass
class CalibrationMetrics:
    """Container for calibration metrics."""
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    average_confidence: float
    average_accuracy: float
    total_samples: int
    num_bins: int
    bin_statistics: List[Dict]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ValidationSample:
    """Single validation sample with ground truth."""
    gesture_embedding: np.ndarray
    true_intent: str
    predicted_intent: str
    confidence: float
    is_correct: bool
    top_k_predictions: List[Tuple[str, float]]


class CalibrationAnalyzer:
    """Analyze probability calibration for semantic intent extraction."""
    
    def __init__(self, intent_extractor: SemanticIntentExtractor, num_bins: int = 10):
        """
        Initialize calibration analyzer.
        
        Args:
            intent_extractor: Semantic intent extraction model
            num_bins: Number of confidence bins for calibration analysis
        """
        self.extractor = intent_extractor
        self.num_bins = num_bins
        self.samples: List[ValidationSample] = []
        
    def add_sample(self, gesture_embedding: np.ndarray, true_intent: str) -> ValidationSample:
        """
        Add validation sample and run inference.
        
        Args:
            gesture_embedding: CLIP gesture embedding (512-dim)
            true_intent: Ground truth intent label
            
        Returns:
            ValidationSample with predictions
        """
        # Extract intent with probabilities
        results = self.extractor.extract_intent(gesture_embedding, top_k=5, min_confidence=0.0)
        
        if not results:
            # No predictions (should not happen with min_confidence=0.0)
            warnings.warn(f"No predictions for sample with true intent: {true_intent}")
            predicted_intent = "UNKNOWN"
            confidence = 0.0
            top_k = []
            is_correct = False
        else:
            # Results are tuples: (intent_name, confidence)
            predicted_intent = results[0][0]
            confidence = results[0][1]
            top_k = [(intent, conf) for intent, conf in results]
            is_correct = (predicted_intent == true_intent)
        
        sample = ValidationSample(
            gesture_embedding=gesture_embedding,
            true_intent=true_intent,
            predicted_intent=predicted_intent,
            confidence=confidence,
            is_correct=is_correct,
            top_k_predictions=top_k
        )
        
        self.samples.append(sample)
        return sample
    
    def compute_calibration_metrics(self) -> CalibrationMetrics:
        """
        Compute comprehensive calibration metrics.
        
        Returns:
            CalibrationMetrics object with all metrics
        """
        if not self.samples:
            raise ValueError("No samples added. Call add_sample() first.")
        
        # Extract confidences and correctness
        confidences = np.array([s.confidence for s in self.samples])
        correct = np.array([s.is_correct for s in self.samples])
        
        # Compute Brier Score
        # BS = (1/n) Σ (p - y)² where p=confidence, y=1 if correct else 0
        brier_score = np.mean((confidences - correct.astype(float)) ** 2)
        
        # Bin samples by confidence
        bin_edges = np.linspace(0, 1, self.num_bins + 1)
        bin_indices = np.digitize(confidences, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        
        # Compute per-bin statistics
        bin_stats = []
        ece = 0.0
        mce = 0.0
        
        for bin_idx in range(self.num_bins):
            mask = bin_indices == bin_idx
            bin_size = mask.sum()
            
            if bin_size == 0:
                bin_stats.append({
                    'bin_index': bin_idx,
                    'bin_range': (bin_edges[bin_idx], bin_edges[bin_idx + 1]),
                    'num_samples': 0,
                    'average_confidence': 0.0,
                    'accuracy': 0.0,
                    'calibration_error': 0.0
                })
                continue
            
            bin_confidences = confidences[mask]
            bin_correct = correct[mask]
            
            avg_confidence = bin_confidences.mean()
            accuracy = bin_correct.mean()
            calibration_error = abs(avg_confidence - accuracy)
            
            # Weighted contribution to ECE
            weight = bin_size / len(self.samples)
            ece += weight * calibration_error
            
            # Track maximum calibration error
            mce = max(mce, calibration_error)
            
            bin_stats.append({
                'bin_index': bin_idx,
                'bin_range': (float(bin_edges[bin_idx]), float(bin_edges[bin_idx + 1])),
                'num_samples': int(bin_size),
                'average_confidence': float(avg_confidence),
                'accuracy': float(accuracy),
                'calibration_error': float(calibration_error)
            })
        
        return CalibrationMetrics(
            expected_calibration_error=float(ece),
            maximum_calibration_error=float(mce),
            brier_score=float(brier_score),
            average_confidence=float(confidences.mean()),
            average_accuracy=float(correct.mean()),
            total_samples=len(self.samples),
            num_bins=self.num_bins,
            bin_statistics=bin_stats
        )
    
    def plot_reliability_diagram(self, save_path: Optional[Path] = None):
        """
        Generate reliability diagram (confidence vs accuracy).
        
        Args:
            save_path: Optional path to save plot
        """
        metrics = self.compute_calibration_metrics()
        
        # Extract bin statistics
        bin_centers = []
        accuracies = []
        confidences = []
        sample_counts = []
        
        for stat in metrics.bin_statistics:
            if stat['num_samples'] > 0:
                bin_center = (stat['bin_range'][0] + stat['bin_range'][1]) / 2
                bin_centers.append(bin_center)
                accuracies.append(stat['accuracy'])
                confidences.append(stat['average_confidence'])
                sample_counts.append(stat['num_samples'])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Reliability Diagram
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        ax1.plot(confidences, accuracies, 'o-', markersize=8, linewidth=2, 
                 label='Model Calibration', color='#2E86AB')
        
        # Add gap lines
        for conf, acc in zip(confidences, accuracies):
            ax1.plot([conf, conf], [conf, acc], 'r--', alpha=0.5, linewidth=1)
        
        ax1.set_xlabel('Confidence', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Add ECE annotation
        ax1.text(0.05, 0.95, f'ECE: {metrics.expected_calibration_error:.4f}\n'
                              f'MCE: {metrics.maximum_calibration_error:.4f}',
                 transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', 
                 facecolor='wheat', alpha=0.5))
        
        # Plot 2: Sample Distribution
        ax2.bar(bin_centers, sample_counts, width=0.08, alpha=0.7, color='#A23B72')
        ax2.set_xlabel('Confidence Bin', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax2.set_title('Sample Distribution Across Confidence Bins', 
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved reliability diagram: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, output_path: Path):
        """
        Generate comprehensive calibration report.
        
        Args:
            output_path: Path to save JSON report
        """
        metrics = self.compute_calibration_metrics()
        
        # Determine calibration status
        if metrics.expected_calibration_error < 0.05:
            calibration_status = "WELL-CALIBRATED"
            recommendation = "Deploy with current probability scores"
        elif metrics.expected_calibration_error < 0.10:
            calibration_status = "ACCEPTABLE"
            recommendation = "Consider temperature scaling for improved calibration"
        else:
            calibration_status = "MISCALIBRATED"
            recommendation = "Apply temperature scaling before deployment"
        
        # Determine confidence bias
        confidence_gap = metrics.average_confidence - metrics.average_accuracy
        if confidence_gap > 0.05:
            bias = "OVERCONFIDENT"
            bias_explanation = "Model predictions are more confident than accurate"
        elif confidence_gap < -0.05:
            bias = "UNDERCONFIDENT"
            bias_explanation = "Model predictions are less confident than accuracy warrants"
        else:
            bias = "NEUTRAL"
            bias_explanation = "Model confidence aligns with accuracy"
        
        report = {
            "calibration_study": {
                "date": "2026-02-11",
                "model": "CLIP ViT-B/32 + Semantic Intent Extractor",
                "dataset_size": metrics.total_samples,
                "num_bins": metrics.num_bins
            },
            "metrics": metrics.to_dict(),
            "analysis": {
                "calibration_status": calibration_status,
                "confidence_bias": bias,
                "bias_explanation": bias_explanation,
                "recommendation": recommendation
            },
            "interpretation": {
                "expected_calibration_error": {
                    "value": metrics.expected_calibration_error,
                    "interpretation": self._interpret_ece(metrics.expected_calibration_error)
                },
                "brier_score": {
                    "value": metrics.brier_score,
                    "interpretation": self._interpret_brier(metrics.brier_score)
                },
                "confidence_gap": {
                    "value": confidence_gap,
                    "interpretation": f"Average confidence - accuracy = {confidence_gap:.4f}"
                }
            },
            "bin_details": metrics.bin_statistics
        }
        
        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Calibration report saved: {output_path}")
        return report
    
    @staticmethod
    def _interpret_ece(ece: float) -> str:
        """Interpret ECE value."""
        if ece < 0.05:
            return "Excellent calibration - probabilities are highly reliable"
        elif ece < 0.10:
            return "Good calibration - probabilities are reasonably reliable"
        elif ece < 0.15:
            return "Moderate calibration - some probability unreliability"
        else:
            return "Poor calibration - probabilities should not be trusted"
    
    @staticmethod
    def _interpret_brier(brier: float) -> str:
        """Interpret Brier score."""
        if brier < 0.10:
            return "Excellent probability accuracy"
        elif brier < 0.20:
            return "Good probability accuracy"
        elif brier < 0.30:
            return "Moderate probability accuracy"
        else:
            return "Poor probability accuracy"
    
    def print_summary(self):
        """Print calibration summary to console."""
        metrics = self.compute_calibration_metrics()
        
        print("\n" + "="*70)
        print("CALIBRATION STUDY SUMMARY")
        print("="*70)
        print(f"\nDataset: {metrics.total_samples} samples")
        print(f"Bins: {metrics.num_bins}")
        
        print(f"\n{'METRIC':<35} {'VALUE':>15}")
        print("-"*70)
        print(f"{'Expected Calibration Error (ECE)':<35} {metrics.expected_calibration_error:>15.4f}")
        print(f"{'Maximum Calibration Error (MCE)':<35} {metrics.maximum_calibration_error:>15.4f}")
        print(f"{'Brier Score':<35} {metrics.brier_score:>15.4f}")
        print(f"{'Average Confidence':<35} {metrics.average_confidence:>15.4f}")
        print(f"{'Average Accuracy':<35} {metrics.average_accuracy:>15.4f}")
        print(f"{'Confidence Gap':<35} {(metrics.average_confidence - metrics.average_accuracy):>15.4f}")
        
        print(f"\n{'BIN':<8} {'RANGE':<15} {'SAMPLES':<10} {'CONF':<10} {'ACC':<10} {'GAP':<10}")
        print("-"*70)
        for stat in metrics.bin_statistics:
            if stat['num_samples'] > 0:
                bin_range = f"{stat['bin_range'][0]:.2f}-{stat['bin_range'][1]:.2f}"
                print(f"{stat['bin_index']:<8} {bin_range:<15} {stat['num_samples']:<10} "
                      f"{stat['average_confidence']:<10.3f} {stat['accuracy']:<10.3f} "
                      f"{stat['calibration_error']:<10.3f}")
        
        print("="*70)


class TemperatureScaler:
    """Apply temperature scaling for post-hoc calibration."""
    
    def __init__(self, intent_extractor: SemanticIntentExtractor):
        """
        Initialize temperature scaler.
        
        Args:
            intent_extractor: Semantic intent extraction model
        """
        self.extractor = intent_extractor
        self.optimal_temperature: Optional[float] = None
    
    def find_optimal_temperature(self, 
                                 validation_samples: List[Tuple[np.ndarray, str]],
                                 temperature_range: Tuple[float, float] = (0.01, 5.0),
                                 num_trials: int = 50) -> float:
        """
        Find optimal temperature that minimizes ECE on validation set.
        
        Args:
            validation_samples: List of (embedding, true_intent) tuples
            temperature_range: (min, max) temperature to search
            num_trials: Number of temperatures to evaluate
            
        Returns:
            Optimal temperature value
        """
        print(f"\nSearching for optimal temperature...")
        print(f"Range: {temperature_range}, Trials: {num_trials}")
        
        temperatures = np.linspace(temperature_range[0], temperature_range[1], num_trials)
        best_temp = self.extractor.temperature
        best_ece = float('inf')
        
        original_temp = self.extractor.temperature
        
        for temp in temperatures:
            # Temporarily set temperature
            self.extractor.temperature = temp
            
            # Evaluate calibration
            analyzer = CalibrationAnalyzer(self.extractor, num_bins=10)
            for embedding, true_intent in validation_samples:
                analyzer.add_sample(embedding, true_intent)
            
            metrics = analyzer.compute_calibration_metrics()
            ece = metrics.expected_calibration_error
            
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
            
            if (temperatures.tolist().index(temp) + 1) % 10 == 0:
                print(f"  Progress: {temperatures.tolist().index(temp) + 1}/{num_trials} "
                      f"(Current best: T={best_temp:.3f}, ECE={best_ece:.4f})")
        
        # Restore original temperature
        self.extractor.temperature = original_temp
        self.optimal_temperature = best_temp
        
        print(f"\n✓ Optimal temperature found: {best_temp:.4f}")
        print(f"  ECE improvement: {best_ece:.4f}")
        
        return best_temp
    
    def apply_temperature(self, temperature: float):
        """Apply temperature scaling to extractor."""
        self.extractor.temperature = temperature
        print(f"✓ Applied temperature: {temperature:.4f}")


def load_validation_dataset(dataset_path: Path) -> List[Tuple[np.ndarray, str]]:
    """
    Load validation dataset from JSON file.
    
    Expected format:
    {
        "samples": [
            {
                "gesture_embedding": [512-dim array],
                "intent": "scroll_up"
            },
            ...
        ]
    }
    
    Args:
        dataset_path: Path to validation dataset JSON
        
    Returns:
        List of (embedding, intent) tuples
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    samples = []
    for sample in data['samples']:
        embedding = np.array(sample['gesture_embedding'], dtype=np.float32)
        intent = sample['intent']
        samples.append((embedding, intent))
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="CLIP Semantic Intent Calibration Study"
    )
    parser.add_argument('--dataset', type=Path, required=True,
                        help='Path to validation dataset JSON')
    parser.add_argument('--output', type=Path, required=True,
                        help='Path to save calibration report JSON')
    parser.add_argument('--plot', type=Path, default=None,
                        help='Path to save reliability diagram')
    parser.add_argument('--num-bins', type=int, default=10,
                        help='Number of confidence bins')
    parser.add_argument('--apply-temperature', action='store_true',
                        help='Find and apply optimal temperature scaling')
    parser.add_argument('--temperature-trials', type=int, default=50,
                        help='Number of temperature values to try')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CLIP SEMANTIC INTENT CALIBRATION STUDY")
    print("="*70)
    
    # Load validation dataset
    print(f"\nLoading validation dataset: {args.dataset}")
    validation_samples = load_validation_dataset(args.dataset)
    print(f"✓ Loaded {len(validation_samples)} samples")
    
    # Initialize CLIP encoder and semantic intent extractor
    print("\nInitializing CLIP semantic intent extractor...")
    encoder = CLIPEncoder(
        model_name="ViT-B-32",
        pretrained="openai",
        device="cpu"  # Change to "cuda" if GPU available
    )
    extractor = SemanticIntentExtractor(
        clip_model=encoder.model,
        device=encoder.device
    )
    print(f"✓ Model initialized")
    print(f"  Available intents: {len(extractor.intent_names)}")
    print(f"  Current temperature: {extractor.temperature}")
    
    # Run calibration analysis
    print(f"\nRunning calibration analysis ({args.num_bins} bins)...")
    analyzer = CalibrationAnalyzer(extractor, num_bins=args.num_bins)
    
    for i, (embedding, true_intent) in enumerate(validation_samples):
        analyzer.add_sample(embedding, true_intent)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(validation_samples)} samples")
    
    print(f"✓ Analysis complete")
    
    # Compute and display metrics
    analyzer.print_summary()
    
    # Generate reliability diagram
    if args.plot:
        print(f"\nGenerating reliability diagram...")
        analyzer.plot_reliability_diagram(save_path=args.plot)
    
    # Apply temperature scaling if requested
    if args.apply_temperature:
        print("\n" + "="*70)
        print("TEMPERATURE SCALING")
        print("="*70)
        
        scaler = TemperatureScaler(extractor)
        optimal_temp = scaler.find_optimal_temperature(
            validation_samples,
            temperature_range=(0.01, 5.0),
            num_trials=args.temperature_trials
        )
        
        # Re-run analysis with optimal temperature
        scaler.apply_temperature(optimal_temp)
        
        print(f"\nRe-running calibration analysis with T={optimal_temp:.4f}...")
        analyzer_scaled = CalibrationAnalyzer(extractor, num_bins=args.num_bins)
        for embedding, true_intent in validation_samples:
            analyzer_scaled.add_sample(embedding, true_intent)
        
        analyzer_scaled.print_summary()
        
        if args.plot:
            scaled_plot_path = args.plot.parent / f"{args.plot.stem}_scaled{args.plot.suffix}"
            analyzer_scaled.plot_reliability_diagram(save_path=scaled_plot_path)
    
    # Generate comprehensive report
    print("\n" + "="*70)
    print("GENERATING REPORT")
    print("="*70)
    
    report = analyzer.generate_report(args.output)
    
    print("\n✓ Calibration study complete")
    print(f"  Status: {report['analysis']['calibration_status']}")
    print(f"  Recommendation: {report['analysis']['recommendation']}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
