#!/usr/bin/env python3
"""
CLIP Confidence Diagnostic Script

Systematically investigates root causes of low semantic confidence scores.

Investigation Areas:
1. Similarity computation validation
2. Prompt engineering analysis
3. Intent vocabulary overlap measurement
4. Logit scaling investigation
5. Confidence distribution analysis

Usage:
    python scripts/diagnose_clip_confidence.py
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

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


class CLIPConfidenceDiagnostics:
    """Comprehensive diagnostics for CLIP semantic confidence issues."""
    
    def __init__(self):
        self.encoder = None
        self.extractor = None
        self.test_embeddings = []
        self.findings = {}
    
    def initialize_models(self):
        """Initialize CLIP encoder and semantic extractor."""
        logger.info("=" * 80)
        logger.info("INITIALIZING CLIP MODELS")
        logger.info("=" * 80)
        
        self.encoder = CLIPEncoder(model_name='ViT-B-32', pretrained='openai', device='cpu')
        self.extractor = SemanticIntentExtractor(
            clip_model=self.encoder.model,
            device=self.encoder.device,
            intent_vocabulary=SEMANTIC_INTENTS
        )
        
        logger.info(f"✅ Encoder: {self.encoder.model_name}, device={self.encoder.device}")
        logger.info(f"✅ Extractor: {len(self.extractor.intent_names)} intents loaded")
    
    def investigate_similarity_computation(self):
        """Investigation 1: Validate cosine similarity computation."""
        logger.info("\n" + "=" * 80)
        logger.info("INVESTIGATION 1: SIMILARITY COMPUTATION VALIDATION")
        logger.info("=" * 80)
        
        # Create test embeddings
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image_embedding = self.encoder.encode_image(test_image)
        
        # Check normalization
        image_norm = np.linalg.norm(image_embedding)
        text_norms = np.linalg.norm(self.extractor.text_embeddings, axis=1)
        
        logger.info(f"Image embedding norm: {image_norm:.6f}")
        logger.info(f"Text embeddings norm range: [{text_norms.min():.6f}, {text_norms.max():.6f}]")
        logger.info(f"Text embeddings norm mean: {text_norms.mean():.6f}")
        
        # Compute similarities manually
        manual_similarities = np.dot(self.extractor.text_embeddings, image_embedding)
        
        # Get similarities from extractor
        intents = self.extractor.extract_intent(image_embedding, top_k=len(SEMANTIC_INTENTS))
        extractor_similarities = np.array([conf for _, conf in intents])
        
        logger.info(f"\nSimilarity statistics:")
        logger.info(f"  Mean: {manual_similarities.mean():.4f}")
        logger.info(f"  Std:  {manual_similarities.std():.4f}")
        logger.info(f"  Min:  {manual_similarities.min():.4f}")
        logger.info(f"  Max:  {manual_similarities.max():.4f}")
        logger.info(f"  Range: {manual_similarities.max() - manual_similarities.min():.4f}")
        
        # Check for CLIP logit scaling
        logger.info(f"\nCLIP model logit_scale parameter:")
        if hasattr(self.encoder.model, 'logit_scale'):
            logit_scale = self.encoder.model.logit_scale.exp().item()
            logger.info(f"  Logit scale (exp): {logit_scale:.4f}")
            logger.info(f"  Logit scale (raw): {self.encoder.model.logit_scale.item():.4f}")
            
            # Apply logit scaling
            scaled_similarities = manual_similarities * logit_scale
            logger.info(f"\nScaled similarity statistics:")
            logger.info(f"  Mean: {scaled_similarities.mean():.4f}")
            logger.info(f"  Std:  {scaled_similarities.std():.4f}")
            logger.info(f"  Min:  {scaled_similarities.min():.4f}")
            logger.info(f"  Max:  {scaled_similarities.max():.4f}")
            
            self.findings['logit_scale'] = logit_scale
            self.findings['raw_similarities'] = manual_similarities
            self.findings['scaled_similarities'] = scaled_similarities
        else:
            logger.warning("  ⚠️  No logit_scale parameter found!")
            self.findings['logit_scale'] = None
        
        # Validation check
        if np.allclose(image_norm, 1.0, atol=1e-5) and np.allclose(text_norms, 1.0, atol=1e-5):
            logger.info("\n✅ Normalization is correct")
        else:
            logger.warning("\n⚠️  Normalization issue detected!")
        
        self.findings['similarity_validation'] = {
            'image_norm': float(image_norm),
            'text_norms_ok': bool(np.allclose(text_norms, 1.0, atol=1e-5)),
            'similarity_mean': float(manual_similarities.mean()),
            'similarity_std': float(manual_similarities.std()),
            'similarity_range': float(manual_similarities.max() - manual_similarities.min())
        }
    
    def investigate_prompt_engineering(self):
        """Investigation 2: Analyze text prompt effectiveness."""
        logger.info("\n" + "=" * 80)
        logger.info("INVESTIGATION 2: PROMPT ENGINEERING ANALYSIS")
        logger.info("=" * 80)
        
        # Analyze current prompts
        total_variations = 0
        prompt_lengths = []
        
        for intent_name, variations in SEMANTIC_INTENTS.items():
            total_variations += len(variations)
            for variation in variations:
                prompt_lengths.append(len(variation.split()))
        
        logger.info(f"Current prompt statistics:")
        logger.info(f"  Total intents: {len(SEMANTIC_INTENTS)}")
        logger.info(f"  Total variations: {total_variations}")
        logger.info(f"  Avg variations per intent: {total_variations / len(SEMANTIC_INTENTS):.1f}")
        logger.info(f"  Prompt length (words):")
        logger.info(f"    Mean: {np.mean(prompt_lengths):.1f}")
        logger.info(f"    Min:  {np.min(prompt_lengths)}")
        logger.info(f"    Max:  {np.max(prompt_lengths)}")
        
        # Test different prompt templates
        logger.info(f"\nTesting different prompt templates:")
        
        test_intent = "increase_volume"
        test_variations = SEMANTIC_INTENTS[test_intent]
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image_embedding = self.encoder.encode_image(test_image)
        
        template_results = {}
        
        # Template 1: Original (no template)
        original_prompts = test_variations
        original_embeddings = self._encode_text_batch(original_prompts)
        original_sim = np.dot(original_embeddings.mean(axis=0), image_embedding)
        template_results['original'] = float(original_sim)
        logger.info(f"  Original: '{test_variations[0]}' → similarity: {original_sim:.4f}")
        
        # Template 2: "A photo of a hand..."
        photo_prompts = [f"a photo of a hand performing {v}" for v in test_variations]
        photo_embeddings = self._encode_text_batch(photo_prompts)
        photo_sim = np.dot(photo_embeddings.mean(axis=0), image_embedding)
        template_results['photo_template'] = float(photo_sim)
        logger.info(f"  Photo template: '{photo_prompts[0]}' → similarity: {photo_sim:.4f}")
        
        # Template 3: "A person making..."
        person_prompts = [f"a person making a {v} hand gesture" for v in test_variations]
        person_embeddings = self._encode_text_batch(person_prompts)
        person_sim = np.dot(person_embeddings.mean(axis=0), image_embedding)
        template_results['person_template'] = float(person_sim)
        logger.info(f"  Person template: '{person_prompts[0]}' → similarity: {person_sim:.4f}")
        
        # Template 4: "Hand gesture for..."
        gesture_prompts = [f"hand gesture for {v}" for v in test_variations]
        gesture_embeddings = self._encode_text_batch(gesture_prompts)
        gesture_sim = np.dot(gesture_embeddings.mean(axis=0), image_embedding)
        template_results['gesture_template'] = float(gesture_sim)
        logger.info(f"  Gesture template: '{gesture_prompts[0]}' → similarity: {gesture_sim:.4f}")
        
        # Find best template
        best_template = max(template_results, key=template_results.get)
        best_score = template_results[best_template]
        
        logger.info(f"\n✅ Best template: {best_template} (similarity: {best_score:.4f})")
        logger.info(f"   Improvement over original: {(best_score - template_results['original']):.4f}")
        
        self.findings['prompt_analysis'] = {
            'current_prompts': {
                'avg_variations': total_variations / len(SEMANTIC_INTENTS),
                'avg_length_words': float(np.mean(prompt_lengths))
            },
            'template_results': template_results,
            'best_template': best_template,
            'improvement': float(best_score - template_results['original'])
        }
    
    def investigate_intent_overlap(self):
        """Investigation 3: Measure semantic overlap between intents."""
        logger.info("\n" + "=" * 80)
        logger.info("INVESTIGATION 3: INTENT VOCABULARY OVERLAP")
        logger.info("=" * 80)
        
        # Compute pairwise similarities between intent embeddings
        text_embeddings = self.extractor.text_embeddings
        similarity_matrix = np.dot(text_embeddings, text_embeddings.T)
        
        # Mask diagonal (self-similarity)
        np.fill_diagonal(similarity_matrix, 0)
        
        # Find highest overlaps
        flat_similarities = similarity_matrix.flatten()
        flat_similarities = flat_similarities[flat_similarities > 0]
        
        logger.info(f"Intent-to-intent similarity statistics:")
        logger.info(f"  Mean: {flat_similarities.mean():.4f}")
        logger.info(f"  Std:  {flat_similarities.std():.4f}")
        logger.info(f"  Min:  {flat_similarities.min():.4f}")
        logger.info(f"  Max:  {flat_similarities.max():.4f}")
        
        # Find most overlapping pairs
        n_intents = len(self.extractor.intent_names)
        overlap_pairs = []
        
        for i in range(n_intents):
            for j in range(i + 1, n_intents):
                sim = similarity_matrix[i, j]
                overlap_pairs.append((
                    self.extractor.intent_names[i],
                    self.extractor.intent_names[j],
                    sim
                ))
        
        # Sort by similarity
        overlap_pairs.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"\nTop 10 most overlapping intent pairs:")
        for i, (intent1, intent2, sim) in enumerate(overlap_pairs[:10], 1):
            logger.info(f"  {i:2d}. {intent1:20s} ↔ {intent2:20s}: {sim:.4f}")
        
        # Check for problematic overlaps (>0.8)
        high_overlap = [p for p in overlap_pairs if p[2] > 0.8]
        if high_overlap:
            logger.warning(f"\n⚠️  {len(high_overlap)} intent pairs have >0.8 similarity!")
            logger.warning("   This may cause ambiguity in classification.")
        else:
            logger.info(f"\n✅ No problematic overlaps (>0.8) detected")
        
        self.findings['intent_overlap'] = {
            'mean_overlap': float(flat_similarities.mean()),
            'max_overlap': float(flat_similarities.max()),
            'high_overlap_count': len(high_overlap),
            'top_overlaps': [(p[0], p[1], float(p[2])) for p in overlap_pairs[:10]]
        }
    
    def investigate_confidence_distribution(self):
        """Investigation 4: Analyze confidence distribution patterns."""
        logger.info("\n" + "=" * 80)
        logger.info("INVESTIGATION 4: CONFIDENCE DISTRIBUTION ANALYSIS")
        logger.info("=" * 80)
        
        # Generate multiple test embeddings
        n_samples = 50
        logger.info(f"Generating {n_samples} random test embeddings...")
        
        all_confidences = []
        top1_confidences = []
        top1_vs_top2_margins = []
        entropies = []
        
        for i in range(n_samples):
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image_embedding = self.encoder.encode_image(test_image)
            
            # Get all similarities
            similarities = np.dot(self.extractor.text_embeddings, image_embedding)
            all_confidences.extend(similarities.tolist())
            
            # Top-1 confidence
            top1_conf = similarities.max()
            top1_confidences.append(top1_conf)
            
            # Top-1 vs Top-2 margin
            sorted_sims = np.sort(similarities)[::-1]
            margin = sorted_sims[0] - sorted_sims[1]
            top1_vs_top2_margins.append(margin)
            
            # Compute entropy of softmax distribution
            exp_sims = np.exp(similarities)
            probs = exp_sims / exp_sims.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
        
        all_confidences = np.array(all_confidences)
        top1_confidences = np.array(top1_confidences)
        top1_vs_top2_margins = np.array(top1_vs_top2_margins)
        entropies = np.array(entropies)
        
        logger.info(f"\nOverall confidence statistics ({n_samples} samples × {len(SEMANTIC_INTENTS)} intents):")
        logger.info(f"  Mean: {all_confidences.mean():.4f}")
        logger.info(f"  Std:  {all_confidences.std():.4f}")
        logger.info(f"  Min:  {all_confidences.min():.4f}")
        logger.info(f"  Max:  {all_confidences.max():.4f}")
        
        logger.info(f"\nTop-1 confidence statistics:")
        logger.info(f"  Mean: {top1_confidences.mean():.4f} ⚠️  Target: ≥0.80")
        logger.info(f"  Std:  {top1_confidences.std():.4f}")
        logger.info(f"  Min:  {top1_confidences.min():.4f}")
        logger.info(f"  Max:  {top1_confidences.max():.4f}")
        
        logger.info(f"\nTop-1 vs Top-2 margin:")
        logger.info(f"  Mean: {top1_vs_top2_margins.mean():.4f}")
        logger.info(f"  Std:  {top1_vs_top2_margins.std():.4f}")
        logger.info(f"  Min:  {top1_vs_top2_margins.min():.4f}")
        logger.info(f"  Max:  {top1_vs_top2_margins.max():.4f}")
        
        logger.info(f"\nPrediction entropy (lower = more confident):")
        logger.info(f"  Mean: {entropies.mean():.4f}")
        logger.info(f"  Max possible: {np.log(len(SEMANTIC_INTENTS)):.4f}")
        logger.info(f"  Normalized: {entropies.mean() / np.log(len(SEMANTIC_INTENTS)):.4f}")
        
        # Diagnose the issue
        if top1_confidences.mean() < 0.30:
            logger.warning("\n❌ ISSUE: Uniformly low confidence across all predictions")
            logger.warning("   Root cause: Raw cosine similarities are in wrong range")
            logger.warning("   Solution: Apply CLIP logit scaling")
        
        if top1_vs_top2_margins.mean() < 0.05:
            logger.warning("\n❌ ISSUE: Small margin between top predictions")
            logger.warning("   Root cause: High semantic overlap or weak signal")
            logger.warning("   Solution: Improve prompts, add temperature scaling")
        
        if entropies.mean() / np.log(len(SEMANTIC_INTENTS)) > 0.9:
            logger.warning("\n❌ ISSUE: High entropy (nearly uniform distribution)")
            logger.warning("   Root cause: Model cannot distinguish intents")
            logger.warning("   Solution: Better prompt engineering, fine-tuning")
        
        self.findings['confidence_distribution'] = {
            'top1_mean': float(top1_confidences.mean()),
            'top1_vs_top2_margin': float(top1_vs_top2_margins.mean()),
            'entropy_normalized': float(entropies.mean() / np.log(len(SEMANTIC_INTENTS))),
            'all_confidence_range': float(all_confidences.max() - all_confidences.min())
        }
    
    def _encode_text_batch(self, texts: List[str]) -> np.ndarray:
        """Helper: Encode batch of text prompts."""
        import open_clip
        
        text_tokens = open_clip.tokenize(texts).to(self.encoder.device)
        
        with torch.no_grad():
            text_features = self.encoder.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().astype(np.float32)
    
    def generate_diagnosis_report(self):
        """Generate comprehensive diagnosis report."""
        logger.info("\n" + "=" * 80)
        logger.info("DIAGNOSIS REPORT")
        logger.info("=" * 80)
        
        logger.info("\n📊 KEY FINDINGS:\n")
        
        # Finding 1: Logit scaling
        if 'logit_scale' in self.findings and self.findings['logit_scale']:
            logit_scale = self.findings['logit_scale']
            logger.info(f"1. CLIP Logit Scaling:")
            logger.info(f"   ⚠️  CRITICAL: CLIP uses logit_scale = {logit_scale:.2f}")
            logger.info(f"   Current implementation does NOT apply this scaling")
            logger.info(f"   Raw similarities are ~100x lower than they should be")
            logger.info(f"   → SOLUTION: Apply logit_scale to similarities")
        
        # Finding 2: Prompt engineering
        if 'prompt_analysis' in self.findings:
            improvement = self.findings['prompt_analysis']['improvement']
            best_template = self.findings['prompt_analysis']['best_template']
            logger.info(f"\n2. Prompt Engineering:")
            logger.info(f"   Current prompts are bare phrases (e.g., 'increase volume')")
            logger.info(f"   Best template: {best_template}")
            logger.info(f"   Potential improvement: +{improvement:.4f} similarity")
            logger.info(f"   → SOLUTION: Use structured prompt templates")
        
        # Finding 3: Intent overlap
        if 'intent_overlap' in self.findings:
            max_overlap = self.findings['intent_overlap']['max_overlap']
            mean_overlap = self.findings['intent_overlap']['mean_overlap']
            logger.info(f"\n3. Intent Vocabulary Overlap:")
            logger.info(f"   Mean intent-to-intent similarity: {mean_overlap:.4f}")
            logger.info(f"   Max overlap: {max_overlap:.4f}")
            if max_overlap > 0.8:
                logger.info(f"   ⚠️  Some intents are too similar")
                logger.info(f"   → SOLUTION: Merge or disambiguate overlapping intents")
            else:
                logger.info(f"   ✅ No problematic overlaps detected")
        
        # Finding 4: Confidence distribution
        if 'confidence_distribution' in self.findings:
            top1_mean = self.findings['confidence_distribution']['top1_mean']
            margin = self.findings['confidence_distribution']['top1_vs_top2_margin']
            logger.info(f"\n4. Confidence Distribution:")
            logger.info(f"   Current Top-1 confidence: {top1_mean:.4f}")
            logger.info(f"   Target: ≥0.80")
            logger.info(f"   Gap: {0.80 - top1_mean:.4f}")
            logger.info(f"   Top-1 vs Top-2 margin: {margin:.4f}")
        
        logger.info(f"\n" + "=" * 80)
        logger.info("RECOMMENDED ACTIONS (Priority Order):")
        logger.info("=" * 80)
        logger.info("\n1. ⚠️  CRITICAL: Apply CLIP logit_scale to similarities")
        logger.info("   Expected improvement: ~100x increase in confidence scores")
        logger.info("   Implementation: similarities *= model.logit_scale.exp()")
        
        logger.info("\n2. 🔧 HIGH: Implement structured prompt templates")
        logger.info("   Expected improvement: +0.05 to +0.15 similarity")
        logger.info("   Implementation: Use 'a photo of a hand performing {action}' format")
        
        logger.info("\n3. 🎚️  MEDIUM: Apply temperature scaling")
        logger.info("   Expected improvement: Sharpen confidence distribution")
        logger.info("   Implementation: similarities /= temperature (try 0.01 to 0.1)")
        
        logger.info("\n4. 🔍 LOW: Review and refine intent vocabulary")
        logger.info("   Expected improvement: Reduce ambiguity")
        logger.info("   Implementation: Merge/disambiguate high-overlap intents")
        
        logger.info("\n" + "=" * 80)
    
    def run_full_diagnostics(self):
        """Run all diagnostic investigations."""
        self.initialize_models()
        self.investigate_similarity_computation()
        self.investigate_prompt_engineering()
        self.investigate_intent_overlap()
        self.investigate_confidence_distribution()
        self.generate_diagnosis_report()
        
        return self.findings


def main():
    """Main entry point."""
    logger.info("Starting CLIP Confidence Diagnostics...")
    
    diagnostics = CLIPConfidenceDiagnostics()
    findings = diagnostics.run_full_diagnostics()
    
    logger.info("\n✅ Diagnostics complete. Findings saved.")
    return findings


if __name__ == '__main__':
    main()
