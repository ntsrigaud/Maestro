"""
Unit Tests for Semantic Intent Extractor (Layer 3)

Tests validate:
- Intent vocabulary initialization
- Embedding similarity computation
- Top-K ranking and filtering
- Batch processing
- Deterministic behavior
"""

import unittest
import numpy as np
import torch
from unittest.mock import MagicMock

from semantic_hgr.semantics.clip_intent import SemanticIntentExtractor, SEMANTIC_INTENTS


class TestSemanticIntentExtractor(unittest.TestCase):
    """Test suite for SemanticIntentExtractor."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures shared across tests."""
        # Create minimal test vocabulary
        cls.test_vocabulary = {
            'action_a': ['do action a', 'perform a', 'execute a'],
            'action_b': ['do action b', 'perform b', 'execute b'],
            'action_c': ['do action c', 'perform c', 'execute c']
        }
        cls.test_device = 'cpu'
        cls.embedding_dim = 512
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create mock CLIP model
        self.mock_clip_model = MagicMock()
        
        # Mock encode_text to return deterministic embeddings
        def mock_encode_text(tokens):
            # Return random but deterministic embeddings based on input shape
            batch_size = tokens.shape[0]
            embeddings = torch.randn(batch_size, self.embedding_dim)
            return embeddings
        
        self.mock_clip_model.encode_text = mock_encode_text
        
        # Mock logit_scale (must return tensor with .item() method)
        mock_logit_scale_tensor = torch.tensor(np.log(100.0))
        mock_logit_scale = MagicMock()
        mock_logit_scale.exp.return_value = mock_logit_scale_tensor
        self.mock_clip_model.logit_scale = mock_logit_scale
        
        # Initialize extractor with mock model
        self.extractor = SemanticIntentExtractor(
            clip_model=self.mock_clip_model,
            device=self.test_device,
            intent_vocabulary=self.test_vocabulary
        )
    
    def test_initialization_success(self):
        """Test successful extractor initialization."""
        self.assertIsNotNone(self.extractor)
        self.assertEqual(self.extractor.device, self.test_device)
        self.assertEqual(len(self.extractor.intent_names), len(self.test_vocabulary))
        self.assertIsNotNone(self.extractor.text_embeddings)
        self.assertEqual(
            self.extractor.text_embeddings.shape,
            (len(self.test_vocabulary), self.embedding_dim)
        )
    
    def test_initialization_empty_vocabulary_error(self):
        """Test error handling for empty vocabulary."""
        with self.assertRaises(ValueError) as context:
            SemanticIntentExtractor(
                clip_model=self.mock_clip_model,
                device=self.test_device,
                intent_vocabulary={}
            )
        
        self.assertIn("cannot be empty", str(context.exception))
    
    def test_get_intent_names(self):
        """Test retrieval of intent names."""
        intent_names = self.extractor.get_intent_names()
        self.assertIsInstance(intent_names, list)
        self.assertEqual(len(intent_names), len(self.test_vocabulary))
        self.assertEqual(set(intent_names), set(self.test_vocabulary.keys()))
    
    def test_get_intent_count(self):
        """Test intent count retrieval."""
        count = self.extractor.get_intent_count()
        self.assertEqual(count, len(self.test_vocabulary))
        self.assertIsInstance(count, int)
    
    def test_get_config(self):
        """Test configuration retrieval."""
        config = self.extractor.get_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config['device'], self.test_device)
        self.assertEqual(config['intent_count'], len(self.test_vocabulary))
        self.assertEqual(config['embedding_dim'], self.embedding_dim)
        self.assertIn('intent_names', config)
    
    def test_extract_intent_valid_embedding(self):
        """Test intent extraction with valid embedding."""
        # Create normalized random embedding
        gesture_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        gesture_embedding /= np.linalg.norm(gesture_embedding)
        
        intents = self.extractor.extract_intent(gesture_embedding, top_k=2)
        
        # Validate output
        self.assertIsInstance(intents, list)
        self.assertLessEqual(len(intents), 2)  # At most top_k
        
        # Check structure
        for intent_name, confidence in intents:
            self.assertIsInstance(intent_name, str)
            self.assertIsInstance(confidence, float)
            self.assertIn(intent_name, self.test_vocabulary.keys())
            self.assertGreaterEqual(confidence, -1.0)
            self.assertLessEqual(confidence, 1.0)
        
        # Check sorted descending
        if len(intents) > 1:
            for i in range(len(intents) - 1):
                self.assertGreaterEqual(intents[i][1], intents[i+1][1])
    
    def test_extract_intent_top_k_parameter(self):
        """Test top_k parameter controls output size."""
        gesture_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        gesture_embedding /= np.linalg.norm(gesture_embedding)
        
        # Test different top_k values
        for k in [1, 2, 3]:
            intents = self.extractor.extract_intent(gesture_embedding, top_k=k)
            self.assertLessEqual(len(intents), k)
    
    def test_extract_intent_min_confidence_filter(self):
        """Test min_confidence filtering."""
        gesture_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        gesture_embedding /= np.linalg.norm(gesture_embedding)
        
        # Extract with high confidence threshold
        intents = self.extractor.extract_intent(
            gesture_embedding,
            top_k=10,
            min_confidence=0.9
        )
        
        # All returned intents should meet threshold
        for _, confidence in intents:
            self.assertGreaterEqual(confidence, 0.9)
    
    def test_extract_intent_deterministic(self):
        """Test that extraction is deterministic."""
        gesture_embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        gesture_embedding /= np.linalg.norm(gesture_embedding)
        
        intents1 = self.extractor.extract_intent(gesture_embedding, top_k=3)
        intents2 = self.extractor.extract_intent(gesture_embedding, top_k=3)
        
        # Should be identical
        self.assertEqual(len(intents1), len(intents2))
        for (name1, conf1), (name2, conf2) in zip(intents1, intents2):
            self.assertEqual(name1, name2)
            self.assertAlmostEqual(conf1, conf2, places=6)
    
    def test_extract_intent_invalid_not_numpy(self):
        """Test error handling for non-NumPy input."""
        invalid_input = [0.1] * self.embedding_dim
        
        with self.assertRaises(ValueError) as context:
            self.extractor.extract_intent(invalid_input)
        
        self.assertIn("Expected NumPy array", str(context.exception))
    
    def test_extract_intent_invalid_shape_2d(self):
        """Test error handling for 2D embedding."""
        invalid_embedding = np.random.randn(1, self.embedding_dim).astype(np.float32)
        
        with self.assertRaises(ValueError) as context:
            self.extractor.extract_intent(invalid_embedding)
        
        self.assertIn("Expected 1D embedding", str(context.exception))
    
    def test_extract_intent_invalid_dimension(self):
        """Test error handling for wrong embedding dimension."""
        wrong_dim = 256  # Not matching expected 512
        invalid_embedding = np.random.randn(wrong_dim).astype(np.float32)
        
        with self.assertRaises(ValueError) as context:
            self.extractor.extract_intent(invalid_embedding)
        
        self.assertIn("dimension mismatch", str(context.exception))
    
    def test_extract_intent_unnormalized_embedding(self):
        """Test automatic normalization of unnormalized embedding."""
        # Create unnormalized embedding
        gesture_embedding = np.random.randn(self.embedding_dim).astype(np.float32) * 10
        
        # Should still work (with warning)
        intents = self.extractor.extract_intent(gesture_embedding, top_k=2)
        
        # Should return valid results
        self.assertIsInstance(intents, list)
        self.assertGreater(len(intents), 0)
    
    def test_extract_batch_valid(self):
        """Test batch intent extraction with valid embeddings."""
        batch_size = 3
        gesture_embeddings = np.random.randn(batch_size, self.embedding_dim).astype(np.float32)
        
        # Normalize each embedding
        norms = np.linalg.norm(gesture_embeddings, axis=1, keepdims=True)
        gesture_embeddings /= norms
        
        batch_intents = self.extractor.extract_batch(gesture_embeddings, top_k=2)
        
        # Validate output
        self.assertIsInstance(batch_intents, list)
        self.assertEqual(len(batch_intents), batch_size)
        
        # Check each result
        for intents in batch_intents:
            self.assertIsInstance(intents, list)
            self.assertLessEqual(len(intents), 2)
            
            for intent_name, confidence in intents:
                self.assertIsInstance(intent_name, str)
                self.assertIsInstance(confidence, float)
    
    def test_extract_batch_deterministic(self):
        """Test that batch extraction is deterministic."""
        gesture_embeddings = np.random.randn(2, self.embedding_dim).astype(np.float32)
        gesture_embeddings /= np.linalg.norm(gesture_embeddings, axis=1, keepdims=True)
        
        batch_intents1 = self.extractor.extract_batch(gesture_embeddings, top_k=3)
        batch_intents2 = self.extractor.extract_batch(gesture_embeddings, top_k=3)
        
        # Should be identical
        self.assertEqual(len(batch_intents1), len(batch_intents2))
        for intents1, intents2 in zip(batch_intents1, batch_intents2):
            self.assertEqual(len(intents1), len(intents2))
            for (name1, conf1), (name2, conf2) in zip(intents1, intents2):
                self.assertEqual(name1, name2)
                self.assertAlmostEqual(conf1, conf2, places=6)
    
    def test_extract_batch_invalid_shape_1d(self):
        """Test error handling for 1D input (missing batch dimension)."""
        invalid_embeddings = np.random.randn(self.embedding_dim).astype(np.float32)
        
        with self.assertRaises(ValueError) as context:
            self.extractor.extract_batch(invalid_embeddings)
        
        self.assertIn("Expected 2D batch", str(context.exception))
    
    def test_extract_batch_invalid_dimension(self):
        """Test error handling for wrong embedding dimension in batch."""
        wrong_dim = 256
        invalid_embeddings = np.random.randn(2, wrong_dim).astype(np.float32)
        
        with self.assertRaises(ValueError) as context:
            self.extractor.extract_batch(invalid_embeddings)
        
        self.assertIn("dimension mismatch", str(context.exception))
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.extractor)
        self.assertIn('SemanticIntentExtractor', repr_str)
        self.assertIn(str(len(self.test_vocabulary)), repr_str)
        self.assertIn(self.test_device, repr_str)


class TestSemanticIntentsVocabulary(unittest.TestCase):
    """Test the default SEMANTIC_INTENTS vocabulary."""
    
    def test_vocabulary_not_empty(self):
        """Test that default vocabulary is not empty."""
        self.assertGreater(len(SEMANTIC_INTENTS), 0)
    
    def test_vocabulary_structure(self):
        """Test vocabulary has correct structure."""
        for intent_name, descriptions in SEMANTIC_INTENTS.items():
            self.assertIsInstance(intent_name, str)
            self.assertIsInstance(descriptions, list)
            self.assertGreater(len(descriptions), 0)
            
            for desc in descriptions:
                self.assertIsInstance(desc, str)
                self.assertGreater(len(desc), 0)
    
    def test_vocabulary_has_key_intents(self):
        """Test vocabulary contains expected key intents."""
        expected_intents = [
            'scroll_up', 'scroll_down',
            'increase_volume', 'decrease_volume',
            'select', 'cancel', 'confirm'
        ]
        
        for intent in expected_intents:
            self.assertIn(intent, SEMANTIC_INTENTS)
    
    def test_vocabulary_no_duplicate_names(self):
        """Test vocabulary has no duplicate intent names."""
        intent_names = list(SEMANTIC_INTENTS.keys())
        self.assertEqual(len(intent_names), len(set(intent_names)))


if __name__ == '__main__':
    unittest.main()
