"""
Enhanced Unit Tests for Semantic Intent Extractor - Probability Validation

Tests validate corrected probability pipeline:
- Softmax normalization
- Probability constraints [0, 1]
- Sum of probabilities ≈ 1.0
- Monotonic behavior
- Numerical stability
"""

import unittest
import numpy as np
import torch
from unittest.mock import MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semantics.clip_intent import SemanticIntentExtractor


class TestProbabilityConstraints(unittest.TestCase):
    """Test probability validity after softmax correction."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_vocabulary = {
            'action_a': ['do action a'],
            'action_b': ['do action b'],
            'action_c': ['do action c'],
            'action_d': ['do action d'],
            'action_e': ['do action e']
        }
        cls.embedding_dim = 512
        
    def setUp(self):
        """Create extractor with mock CLIP model."""
        # Mock CLIP model
        self.mock_clip_model = MagicMock()
        
        def mock_encode_text(tokens):
            batch_size = tokens.shape[0]
            embeddings = torch.randn(batch_size, self.embedding_dim)
            return embeddings
        
        self.mock_clip_model.encode_text = mock_encode_text
        
        # Mock logit_scale (must return tensor with .item() method)
        mock_logit_scale_tensor = torch.tensor(np.log(100.0))
        mock_logit_scale = MagicMock()
        mock_logit_scale.exp.return_value = mock_logit_scale_tensor
        self.mock_clip_model.logit_scale = mock_logit_scale
        
        # Initialize extractor
        self.extractor = SemanticIntentExtractor(
            clip_model=self.mock_clip_model,
            device='cpu',
            intent_vocabulary=self.test_vocabulary,
            use_prompt_template=True,
            temperature=0.07,
            apply_logit_scale=True
        )
    
    def test_probabilities_in_valid_range(self):
        """Test all probabilities are in [0, 1]."""
        gesture_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        gesture_embedding /= np.linalg.norm(gesture_embedding)
        
        intents = self.extractor.extract_intent(
            gesture_embedding, 
            top_k=len(self.test_vocabulary)  # Get all intents
        )
        
        for intent_name, probability in intents:
            self.assertGreaterEqual(
                probability, 0.0,
                f"Probability for {intent_name} is negative: {probability}"
            )
            self.assertLessEqual(
                probability, 1.0,
                f"Probability for {intent_name} exceeds 1.0: {probability}"
            )
    
    def test_probabilities_sum_to_one(self):
        """Test all probabilities sum to approximately 1.0."""
        gesture_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        gesture_embedding /= np.linalg.norm(gesture_embedding)
        
        # Get ALL intents (no filtering)
        intents = self.extractor.extract_intent(
            gesture_embedding, 
            top_k=len(self.test_vocabulary),
            min_confidence=0.0
        )
        
        total_probability = sum(prob for _, prob in intents)
        
        self.assertAlmostEqual(
            total_probability, 1.0, places=4,
            msg=f"Probabilities sum to {total_probability}, not 1.0"
        )
    
    def test_no_nan_or_inf(self):
        """Test no NaN or Inf in probabilities."""
        gesture_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        gesture_embedding /= np.linalg.norm(gesture_embedding)
        
        intents = self.extractor.extract_intent(gesture_embedding, top_k=5)
        
        for intent_name, probability in intents:
            self.assertTrue(
                np.isfinite(probability),
                f"Probability for {intent_name} is NaN or Inf: {probability}"
            )
    
    def test_batch_probabilities_sum_to_one(self):
        """Test batch probabilities sum to 1.0 for each sample."""
        batch_size = 3
        gesture_embeddings = np.random.randn(batch_size, self.embedding_dim).astype(np.float32)
        gesture_embeddings /= np.linalg.norm(gesture_embeddings, axis=1, keepdims=True)
        
        batch_intents = self.extractor.extract_batch(
            gesture_embeddings,
            top_k=len(self.test_vocabulary),
            min_confidence=0.0
        )
        
        for i, intents in enumerate(batch_intents):
            total_prob = sum(prob for _, prob in intents)
            self.assertAlmostEqual(
                total_prob, 1.0, places=4,
                msg=f"Sample {i}: probabilities sum to {total_prob}, not 1.0"
            )


class TestMonotonicBehavior(unittest.TestCase):
    """Test monotonic properties of probability computation."""
    
    def setUp(self):
        """Create extractor with controlled embeddings."""
        self.embedding_dim = 512
        
        # Create vocabulary with controlled text embeddings
        self.test_vocabulary = {
            'high_similarity': ['specific unique phrase alpha'],
            'low_similarity': ['completely different beta']
        }
        
        # Mock CLIP model with controlled outputs
        self.mock_clip_model = MagicMock()
        
        # Create two very different text embeddings
        # High similarity: will be similar to gesture
        self.text_emb_high = np.zeros(self.embedding_dim, dtype=np.float32)
        self.text_emb_high[:256] = 1.0  # First half is 1
        self.text_emb_high /= np.linalg.norm(self.text_emb_high)
        
        # Low similarity: completely different direction
        self.text_emb_low = np.zeros(self.embedding_dim, dtype=np.float32)
        self.text_emb_low[256:] = 1.0  # Second half is 1 (orthogonal)
        self.text_emb_low /= np.linalg.norm(self.text_emb_low)
        
        # Verify they're orthogonal
        assert abs(np.dot(self.text_emb_high, self.text_emb_low)) < 1e-6, "Embeddings not orthogonal"
        
        # Track which embeddings have been requested
        self.embedding_call_count = 0
        
        def mock_encode_text(tokens):
            batch_size = tokens.shape[0]
            # Return different embeddings for each intent
            embeddings = []
            for i in range(batch_size):
                if i == 0:  # First intent (high_similarity)
                    embeddings.append(self.text_emb_high)
                else:  # Second intent (low_similarity)
                    embeddings.append(self.text_emb_low)
            return torch.from_numpy(np.stack(embeddings))
        
        self.mock_clip_model.encode_text = mock_encode_text
        
        # Mock logit_scale (must return tensor with .item() method)
        mock_logit_scale_tensor = torch.tensor(np.log(100.0))
        mock_logit_scale = MagicMock()
        mock_logit_scale.exp.return_value = mock_logit_scale_tensor
        self.mock_clip_model.logit_scale = mock_logit_scale
        
        self.extractor = SemanticIntentExtractor(
            clip_model=self.mock_clip_model,
            device='cpu',
            intent_vocabulary=self.test_vocabulary,
            temperature=0.07,
            apply_logit_scale=True
        )
        
        # CRITICAL: Override the text embeddings directly to ensure controlled test
        self.extractor.text_embeddings = np.stack([self.text_emb_high, self.text_emb_low])
        self.extractor.intent_names = ['high_similarity', 'low_similarity']
    
    def test_higher_similarity_gives_higher_probability(self):
        """Test that higher cosine similarity produces higher probability."""
        # Create gesture embedding more similar to high_similarity
        gesture_embedding = self.text_emb_high + 0.1 * np.random.randn(self.embedding_dim).astype(np.float32)
        gesture_embedding /= np.linalg.norm(gesture_embedding)
        
        intents = self.extractor.extract_intent(gesture_embedding, top_k=2)
        intent_dict = {name: prob for name, prob in intents}
        
        # High similarity intent should have higher probability
        self.assertGreater(
            intent_dict['high_similarity'],
            intent_dict['low_similarity'],
            "Higher similarity did not produce higher probability"
        )
    
    def test_temperature_sharpens_distribution(self):
        """Test lower temperature produces sharper distribution."""
        gesture_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        gesture_embedding /= np.linalg.norm(gesture_embedding)
        
        # High temperature (smoother distribution)
        extractor_high_temp = SemanticIntentExtractor(
            clip_model=self.mock_clip_model,
            device='cpu',
            intent_vocabulary=self.test_vocabulary,
            temperature=1.0,  # High temperature
            apply_logit_scale=True
        )
        # Override embeddings for controlled test
        extractor_high_temp.text_embeddings = np.stack([self.text_emb_high, self.text_emb_low])
        extractor_high_temp.intent_names = ['high_similarity', 'low_similarity']
        
        intents_high = extractor_high_temp.extract_intent(gesture_embedding, top_k=2)
        probs_high = [prob for _, prob in intents_high]
        margin_high = probs_high[0] - probs_high[1]
        
        # Low temperature (sharper distribution)
        extractor_low_temp = SemanticIntentExtractor(
            clip_model=self.mock_clip_model,
            device='cpu',
            intent_vocabulary=self.test_vocabulary,
            temperature=0.05,  # Low temperature
            apply_logit_scale=True
        )
        # Override embeddings for controlled test
        extractor_low_temp.text_embeddings = np.stack([self.text_emb_high, self.text_emb_low])
        extractor_low_temp.intent_names = ['high_similarity', 'low_similarity']
        
        intents_low = extractor_low_temp.extract_intent(gesture_embedding, top_k=2)
        probs_low = [prob for _, prob in intents_low]
        margin_low = probs_low[0] - probs_low[1]
        
        # Lower temperature should produce larger margin (sharper)
        self.assertGreater(
            margin_low, margin_high,
            f"Lower temperature did not sharpen distribution: "
            f"margin_low={margin_low:.4f}, margin_high={margin_high:.4f}"
        )


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability under extreme conditions."""
    
    def setUp(self):
        """Create extractor."""
        self.embedding_dim = 512
        self.test_vocabulary = {
            'action_a': ['do action a'],
            'action_b': ['do action b']
        }
        
        # Mock CLIP model
        self.mock_clip_model = MagicMock()
        
        def mock_encode_text(tokens):
            batch_size = tokens.shape[0]
            return torch.randn(batch_size, self.embedding_dim)
        
        self.mock_clip_model.encode_text = mock_encode_text
        
        # Mock logit_scale (must return tensor with .item() method)
        mock_logit_scale_tensor = torch.tensor(np.log(100.0))
        mock_logit_scale = MagicMock()
        mock_logit_scale.exp.return_value = mock_logit_scale_tensor
        self.mock_clip_model.logit_scale = mock_logit_scale
        
        self.extractor = SemanticIntentExtractor(
            clip_model=self.mock_clip_model,
            device='cpu',
            intent_vocabulary=self.test_vocabulary,
            temperature=0.07,
            apply_logit_scale=True
        )
    
    def test_no_overflow_with_extreme_logits(self):
        """Test no overflow when logits are very large."""
        # Create embedding that will produce large logits
        gesture_embedding = np.ones(self.embedding_dim, dtype=np.float32)
        gesture_embedding /= np.linalg.norm(gesture_embedding)
        
        # Should not raise overflow error
        try:
            intents = self.extractor.extract_intent(gesture_embedding, top_k=2)
            
            # Probabilities should still be valid
            for _, prob in intents:
                self.assertTrue(np.isfinite(prob))
                self.assertGreaterEqual(prob, 0.0)
                self.assertLessEqual(prob, 1.0)
        except (OverflowError, FloatingPointError) as e:
            self.fail(f"Overflow occurred with extreme logits: {e}")
    
    def test_deterministic_under_fixed_seed(self):
        """Test outputs are deterministic under fixed random seed."""
        np.random.seed(42)
        gesture_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        gesture_embedding /= np.linalg.norm(gesture_embedding)
        
        intents1 = self.extractor.extract_intent(gesture_embedding, top_k=2)
        
        # Reset seed and rerun
        np.random.seed(42)
        gesture_embedding2 = np.random.randn(self.embedding_dim).astype(np.float32)
        gesture_embedding2 /= np.linalg.norm(gesture_embedding2)
        
        intents2 = self.extractor.extract_intent(gesture_embedding2, top_k=2)
        
        # Should be identical
        for (name1, prob1), (name2, prob2) in zip(intents1, intents2):
            self.assertEqual(name1, name2)
            self.assertAlmostEqual(prob1, prob2, places=6)


class TestSoftmaxCorrection(unittest.TestCase):
    """Test that softmax correction was properly applied."""
    
    def setUp(self):
        """Create extractor."""
        self.embedding_dim = 512
        self.test_vocabulary = {
            f'action_{i}': [f'do action {i}'] 
            for i in range(10)  # 10 intents
        }
        
        # Mock CLIP model
        self.mock_clip_model = MagicMock()
        
        def mock_encode_text(tokens):
            batch_size = tokens.shape[0]
            return torch.randn(batch_size, self.embedding_dim)
        
        self.mock_clip_model.encode_text = mock_encode_text
        
        # Mock logit_scale (must return tensor with .item() method)
        mock_logit_scale_tensor = torch.tensor(np.log(100.0))
        mock_logit_scale = MagicMock()
        mock_logit_scale.exp.return_value = mock_logit_scale_tensor
        self.mock_clip_model.logit_scale = mock_logit_scale
        
        self.extractor = SemanticIntentExtractor(
            clip_model=self.mock_clip_model,
            device='cpu',
            intent_vocabulary=self.test_vocabulary,
            temperature=0.07,
            apply_logit_scale=True
        )
    
    def test_all_probabilities_accessible(self):
        """Test we can retrieve all probabilities that sum to 1.0."""
        gesture_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        gesture_embedding /= np.linalg.norm(gesture_embedding)
        
        # Get ALL intents
        all_intents = self.extractor.extract_intent(
            gesture_embedding,
            top_k=len(self.test_vocabulary),
            min_confidence=0.0
        )
        
        # Should return all intents
        self.assertEqual(len(all_intents), len(self.test_vocabulary))
        
        # Sum should be 1.0
        total = sum(prob for _, prob in all_intents)
        self.assertAlmostEqual(total, 1.0, places=4)
    
    def test_probabilities_not_raw_logits(self):
        """Test probabilities are bounded, proving they're not raw logits."""
        gesture_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        gesture_embedding /= np.linalg.norm(gesture_embedding)
        
        intents = self.extractor.extract_intent(gesture_embedding, top_k=5)
        
        for name, prob in intents:
            # Raw logits (cosine_sim * 100 / 0.07) would be ~1400x larger
            # Probabilities must be ≤ 1.0
            self.assertLessEqual(
                prob, 1.0,
                f"Probability {prob} exceeds 1.0 - appears to be raw logit!"
            )


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
