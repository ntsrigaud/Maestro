#!/usr/bin/env python3
"""
CLIP Improvements Validation Script

Validates the effectiveness of applied fixes:
1. CLIP logit_scale application
2. Structured prompt templates
3. Temperature scaling

Compares baseline vs improved configurations with comprehensive metrics.

Usage:
    python scripts/validate_clip_improvements.py
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from representation import CLIPEncoder
from semantics import SemanticIntentExtractor, SEMANTIC_INTENTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


class CLIPImprovementValidator:
    """Validates improvements to semantic confidence extraction."""
    
    def __init__(self):
        self.encoder = None
        self.test_samples = 100
        
    def initialize_encoder(self):
        """Initialize CLIP encoder (shared across configurations)."""
        logger.info("Initializing CLIP encoder...")
        self.encoder = CLIPEncoder(model_name='ViT-B-32', pretrained='openai', device='cpu')
        logger.info(f"✅ Encoder ready: {self.encoder.model_name}, embedding_dim={self.encoder.embedding_dim}")
    
    def generate_test_embeddings(self, n_samples: int) -> List[np.ndarray]:
        """Generate test gesture embeddings."""
        logger.info(f"Generating {n_samples} test embeddings...")
        embeddings = []
        
        for i in range(n_samples):
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            embedding = self.encoder.encode_image(test_image)
            embeddings.append(embedding)
        
        logger.info(f"✅ Generated {len(embeddings)} test embeddings")
        return embeddings
    
    def evaluate_configuration(
        self,
        config_name: str,
        use_prompt_template: bool,
        temperature: float,
        apply_logit_scale: bool,
        test_embeddings: List[np.ndarray]
    ) -> Dict:
        """Evaluate semantic extraction with specific configuration."""
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATING: {config_name}")
        logger.info(f"{'='*80}")
        logger.info(f"  prompt_template: {use_prompt_template}")
        logger.info(f"  temperature: {temperature}")
        logger.info(f"  apply_logit_scale: {apply_logit_scale}")
        
        # Initialize extractor with configuration
        extractor = SemanticIntentExtractor(
            clip_model=self.encoder.model,
            device=self.encoder.device,
            intent_vocabulary=SEMANTIC_INTENTS,
            use_prompt_template=use_prompt_template,
            temperature=temperature,
            apply_logit_scale=apply_logit_scale
        )
        
        # Evaluate metrics
        top1_confidences = []
        top1_vs_top2_margins = []
        all_confidences = []
        entropies = []
        latencies_ms = []
        
        for embedding in test_embeddings:
            start_time = time.time()
            
            # Extract all intents to compute full distribution
            intents = extractor.extract_intent(embedding, top_k=len(SEMANTIC_INTENTS), min_confidence=-np.inf)
            
            latency_ms = (time.time() - start_time) * 1000
            latencies_ms.append(latency_ms)
            
            # Extract confidences
            confidences = np.array([conf for _, conf in intents])
            all_confidences.extend(confidences.tolist())
            
            # Top-1 confidence
            top1_conf = confidences[0]
            top1_confidences.append(top1_conf)
            
            # Top-1 vs Top-2 margin
            if len(confidences) >= 2:
                margin = confidences[0] - confidences[1]
                top1_vs_top2_margins.append(margin)
            
            # Compute entropy (using softmax to convert to probabilities)
            # Only if confidences are in reasonable range
            if confidences.max() < 100:  # Avoid overflow
                exp_confs = np.exp(confidences)
                probs = exp_confs / exp_confs.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)
        
        top1_confidences = np.array(top1_confidences)
        top1_vs_top2_margins = np.array(top1_vs_top2_margins)
        all_confidences = np.array(all_confidences)
        latencies_ms = np.array(latencies_ms)
        
        # Compute metrics
        results = {
            'config_name': config_name,
            'top1_mean': float(top1_confidences.mean()),
            'top1_std': float(top1_confidences.std()),
            'top1_min': float(top1_confidences.min()),
            'top1_max': float(top1_confidences.max()),
            'margin_mean': float(top1_vs_top2_margins.mean()),
            'margin_std': float(top1_vs_top2_margins.std()),
            'all_conf_mean': float(all_confidences.mean()),
            'all_conf_std': float(all_confidences.std()),
            'latency_mean_ms': float(latencies_ms.mean()),
            'latency_std_ms': float(latencies_ms.std()),
            'samples_above_0.8': int(np.sum(top1_confidences >= 0.8)),
            'samples_above_0.9': int(np.sum(top1_confidences >= 0.9)),
        }
        
        if entropies:
            entropies = np.array(entropies)
            results['entropy_mean'] = float(entropies.mean())
            results['entropy_normalized'] = float(entropies.mean() / np.log(len(SEMANTIC_INTENTS)))
        else:
            results['entropy_mean'] = None
            results['entropy_normalized'] = None
        
        # Log results
        logger.info(f"\n📊 RESULTS:")
        logger.info(f"  Top-1 Confidence:")
        logger.info(f"    Mean: {results['top1_mean']:.4f} (Target: ≥0.80)")
        logger.info(f"    Std:  {results['top1_std']:.4f}")
        logger.info(f"    Min:  {results['top1_min']:.4f}")
        logger.info(f"    Max:  {results['top1_max']:.4f}")
        logger.info(f"  Top-1 vs Top-2 Margin:")
        logger.info(f"    Mean: {results['margin_mean']:.4f}")
        logger.info(f"    Std:  {results['margin_std']:.4f}")
        logger.info(f"  Latency:")
        logger.info(f"    Mean: {results['latency_mean_ms']:.2f} ms")
        logger.info(f"    Std:  {results['latency_std_ms']:.2f} ms")
        logger.info(f"  Success Rate:")
        logger.info(f"    Samples ≥0.8: {results['samples_above_0.8']}/{len(test_embeddings)} ({100*results['samples_above_0.8']/len(test_embeddings):.1f}%)")
        logger.info(f"    Samples ≥0.9: {results['samples_above_0.9']}/{len(test_embeddings)} ({100*results['samples_above_0.9']/len(test_embeddings):.1f}%)")
        
        if results['entropy_normalized'] is not None:
            logger.info(f"  Entropy:")
            logger.info(f"    Normalized: {results['entropy_normalized']:.4f} (lower = more confident)")
        
        # Verdict
        if results['top1_mean'] >= 0.80:
            logger.info(f"\n✅ SUCCESS: Confidence target met!")
        else:
            gap = 0.80 - results['top1_mean']
            logger.info(f"\n⚠️  Gap to target: {gap:.4f}")
        
        return results
    
    def compare_configurations(self, all_results: List[Dict]):
        """Generate comparison report across configurations."""
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPARATIVE ANALYSIS")
        logger.info(f"{'='*80}")
        
        # Find baseline and best
        baseline = all_results[0]
        best = max(all_results, key=lambda x: x['top1_mean'])
        
        logger.info(f"\n📊 CONFIGURATION RANKING (by Top-1 Confidence):\n")
        
        sorted_results = sorted(all_results, key=lambda x: x['top1_mean'], reverse=True)
        for i, result in enumerate(sorted_results, 1):
            improvement = result['top1_mean'] - baseline['top1_mean']
            logger.info(
                f"  {i}. {result['config_name']:30s}: {result['top1_mean']:.4f} "
                f"(+{improvement:.4f} vs baseline, {result['latency_mean_ms']:.1f}ms)"
            )
        
        logger.info(f"\n📈 IMPROVEMENT BREAKDOWN:\n")
        
        # Baseline to best
        top1_improvement = best['top1_mean'] - baseline['top1_mean']
        margin_improvement = best['margin_mean'] - baseline['margin_mean']
        
        logger.info(f"  Baseline → Best Configuration:")
        logger.info(f"    Top-1 Confidence: +{top1_improvement:.4f} ({100*top1_improvement/baseline['top1_mean']:.1f}% increase)")
        logger.info(f"    Top-1 Margin:     +{margin_improvement:.4f}")
        logger.info(f"    Configuration:    {best['config_name']}")
        
        # Component contributions
        logger.info(f"\n🔧 COMPONENT CONTRIBUTIONS:\n")
        
        # Find configuration with only logit_scale
        logit_only = next((r for r in all_results if 
                          r['config_name'] == 'Baseline + Logit Scale'), None)
        if logit_only:
            contrib = logit_only['top1_mean'] - baseline['top1_mean']
            logger.info(f"  Logit Scale:        +{contrib:.4f}")
        
        # Find configuration with logit + prompt
        logit_prompt = next((r for r in all_results if 
                            'Logit + Prompt' in r['config_name']), None)
        if logit_prompt and logit_only:
            contrib = logit_prompt['top1_mean'] - logit_only['top1_mean']
            logger.info(f"  Prompt Template:    +{contrib:.4f}")
        
        # Full configuration
        full_config = next((r for r in all_results if 
                           'Full Fix' in r['config_name']), None)
        if full_config and logit_prompt:
            contrib = full_config['top1_mean'] - logit_prompt['top1_mean']
            logger.info(f"  Temperature Tuning: +{contrib:.4f}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"RECOMMENDATION")
        logger.info(f"{'='*80}\n")
        
        if best['top1_mean'] >= 0.80:
            logger.info(f"✅ Use: {best['config_name']}")
            logger.info(f"   Confidence: {best['top1_mean']:.4f} (meets ≥0.80 target)")
            logger.info(f"   Latency: {best['latency_mean_ms']:.1f} ms")
            logger.info(f"   Success rate: {100*best['samples_above_0.8']/self.test_samples:.1f}%")
        else:
            logger.info(f"⚠️  Best configuration: {best['config_name']}")
            logger.info(f"   Confidence: {best['top1_mean']:.4f}")
            logger.info(f"   Gap to target: {0.80 - best['top1_mean']:.4f}")
            logger.info(f"\n   RECOMMENDATION: Fine-tuning CLIP on gesture dataset may be required")
            logger.info(f"   to achieve ≥0.80 confidence with current intent vocabulary.")
        
        return best
    
    def run_full_validation(self):
        """Run comprehensive validation across all configurations."""
        logger.info("="*80)
        logger.info("CLIP IMPROVEMENTS VALIDATION")
        logger.info("="*80)
        
        # Initialize encoder
        self.initialize_encoder()
        
        # Generate test embeddings (shared across configs)
        test_embeddings = self.generate_test_embeddings(self.test_samples)
        
        # Test configurations
        configs = [
            {
                'name': 'Baseline (Original)',
                'use_prompt_template': False,
                'temperature': 1.0,  # No temperature scaling
                'apply_logit_scale': False
            },
            {
                'name': 'Baseline + Logit Scale',
                'use_prompt_template': False,
                'temperature': 1.0,
                'apply_logit_scale': True
            },
            {
                'name': 'Logit + Prompt Template',
                'use_prompt_template': True,
                'temperature': 1.0,
                'apply_logit_scale': True
            },
            {
                'name': 'Full Fix (temp=0.07)',
                'use_prompt_template': True,
                'temperature': 0.07,
                'apply_logit_scale': True
            },
            {
                'name': 'Full Fix (temp=0.05)',
                'use_prompt_template': True,
                'temperature': 0.05,
                'apply_logit_scale': True
            },
            {
                'name': 'Full Fix (temp=0.10)',
                'use_prompt_template': True,
                'temperature': 0.10,
                'apply_logit_scale': True
            },
        ]
        
        all_results = []
        
        for config in configs:
            results = self.evaluate_configuration(
                config_name=config['name'],
                use_prompt_template=config['use_prompt_template'],
                temperature=config['temperature'],
                apply_logit_scale=config['apply_logit_scale'],
                test_embeddings=test_embeddings
            )
            all_results.append(results)
            
            # Brief pause
            time.sleep(0.5)
        
        # Comparative analysis
        best_config = self.compare_configurations(all_results)
        
        return all_results, best_config


def main():
    """Main entry point."""
    validator = CLIPImprovementValidator()
    all_results, best_config = validator.run_full_validation()
    
    logger.info("\n✅ Validation complete.")
    return all_results, best_config


if __name__ == '__main__':
    main()
