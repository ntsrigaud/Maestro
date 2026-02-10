"""
Unit Tests for CLIP Encoder (Layer 2)

Tests validate:
- Model initialization and lifecycle
- Input validation and error handling
- Embedding generation correctness
- Deterministic behavior
- No external dependencies (no network access)
"""

import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from representation.clip_encoder import CLIPEncoder


class TestCLIPEncoder(unittest.TestCase):
    """Test suite for CLIPEncoder."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests."""
        cls.test_device = 'cpu'  # Always use CPU for deterministic tests
        cls.test_model_name = 'ViT-B-32'
        cls.expected_embedding_dim = 512  # For ViT-B/32
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Initialize encoder once per test
        self.encoder = CLIPEncoder(
            model_name=self.test_model_name,
            pretrained='openai',
            device=self.test_device
        )
    
    def test_initialization_success(self):
        """Test successful encoder initialization."""
        self.assertIsNotNone(self.encoder)
        self.assertEqual(self.encoder.model_name, self.test_model_name)
        self.assertEqual(self.encoder.device, self.test_device)
        self.assertEqual(self.encoder.embedding_dim, self.expected_embedding_dim)
        self.assertIsNotNone(self.encoder.model)
        self.assertIsNotNone(self.encoder.preprocess)
    
    def test_get_embedding_dim(self):
        """Test embedding dimension retrieval."""
        dim = self.encoder.get_embedding_dim()
        self.assertEqual(dim, self.expected_embedding_dim)
        self.assertIsInstance(dim, int)
    
    def test_get_config(self):
        """Test configuration retrieval."""
        config = self.encoder.get_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config['model_name'], self.test_model_name)
        self.assertEqual(config['device'], self.test_device)
        self.assertEqual(config['embedding_dim'], self.expected_embedding_dim)
        self.assertIn('pretrained', config)
    
    def test_encode_image_valid_uint8(self):
        """Test encoding valid uint8 RGB image."""
        # Create dummy RGB image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        embedding = self.encoder.encode_image(image)
        
        # Validate output
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (self.expected_embedding_dim,))
        self.assertEqual(embedding.dtype, np.float32)
        
        # Check normalization (L2 norm should be ~1.0)
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)
        
        # Check no NaN or Inf
        self.assertTrue(np.isfinite(embedding).all())
    
    def test_encode_image_valid_float32(self):
        """Test encoding valid float32 RGB image in [0, 1] range."""
        # Create dummy RGB image
        image = np.random.rand(224, 224, 3).astype(np.float32)
        
        embedding = self.encoder.encode_image(image)
        
        # Validate output
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (self.expected_embedding_dim,))
        self.assertEqual(embedding.dtype, np.float32)
        
        # Check normalization
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_encode_image_deterministic(self):
        """Test that encoding same image produces same embedding."""
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        embedding1 = self.encoder.encode_image(image)
        embedding2 = self.encoder.encode_image(image)
        
        # Should be identical (deterministic)
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=6)
    
    def test_encode_image_different_inputs(self):
        """Test that different images produce different embeddings."""
        image1 = np.zeros((224, 224, 3), dtype=np.uint8)  # Black
        image2 = np.full((224, 224, 3), 255, dtype=np.uint8)  # White
        
        embedding1 = self.encoder.encode_image(image1)
        embedding2 = self.encoder.encode_image(image2)
        
        # Embeddings should be different
        self.assertFalse(np.allclose(embedding1, embedding2))
    
    def test_encode_image_invalid_not_numpy(self):
        """Test error handling for non-NumPy input."""
        invalid_input = [[1, 2, 3]]  # List instead of array
        
        with self.assertRaises(ValueError) as context:
            self.encoder.encode_image(invalid_input)
        
        self.assertIn("Expected NumPy array", str(context.exception))
    
    def test_encode_image_invalid_shape_2d(self):
        """Test error handling for 2D image (missing channel dimension)."""
        invalid_image = np.zeros((224, 224), dtype=np.uint8)
        
        with self.assertRaises(ValueError) as context:
            self.encoder.encode_image(invalid_image)
        
        self.assertIn("Expected (H, W, 3)", str(context.exception))
    
    def test_encode_image_invalid_shape_4_channels(self):
        """Test error handling for RGBA image (4 channels)."""
        invalid_image = np.zeros((224, 224, 4), dtype=np.uint8)
        
        with self.assertRaises(ValueError) as context:
            self.encoder.encode_image(invalid_image)
        
        self.assertIn("Expected (H, W, 3)", str(context.exception))
    
    def test_encode_image_invalid_dtype(self):
        """Test error handling for unsupported dtype."""
        invalid_image = np.zeros((224, 224, 3), dtype=np.int16)
        
        with self.assertRaises(ValueError) as context:
            self.encoder.encode_image(invalid_image)
        
        self.assertIn("Unsupported dtype", str(context.exception))
    
    def test_encode_batch_valid(self):
        """Test batch encoding with valid inputs."""
        batch_size = 3
        images = np.random.randint(0, 255, (batch_size, 224, 224, 3), dtype=np.uint8)
        
        embeddings = self.encoder.encode_batch(images)
        
        # Validate output
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (batch_size, self.expected_embedding_dim))
        self.assertEqual(embeddings.dtype, np.float32)
        
        # Check each embedding is normalized
        norms = np.linalg.norm(embeddings, axis=1)
        for norm in norms:
            self.assertAlmostEqual(norm, 1.0, places=5)
        
        # Check no NaN or Inf
        self.assertTrue(np.isfinite(embeddings).all())
    
    def test_encode_batch_deterministic(self):
        """Test that batch encoding is deterministic."""
        images = np.random.randint(0, 255, (2, 224, 224, 3), dtype=np.uint8)
        
        embeddings1 = self.encoder.encode_batch(images)
        embeddings2 = self.encoder.encode_batch(images)
        
        np.testing.assert_array_almost_equal(embeddings1, embeddings2, decimal=6)
    
    def test_encode_batch_invalid_shape_3d(self):
        """Test error handling for 3D input (missing batch dimension)."""
        invalid_images = np.zeros((224, 224, 3), dtype=np.uint8)
        
        with self.assertRaises(ValueError) as context:
            self.encoder.encode_batch(invalid_images)
        
        self.assertIn("Expected (N, H, W, 3)", str(context.exception))
    
    def test_encode_batch_consistency_with_single(self):
        """Test that batch encoding matches single image encoding."""
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Single encoding
        single_embedding = self.encoder.encode_image(image)
        
        # Batch encoding with single image
        batch_embeddings = self.encoder.encode_batch(image[np.newaxis, ...])
        
        # Should match
        np.testing.assert_array_almost_equal(
            single_embedding,
            batch_embeddings[0],
            decimal=5
        )
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.encoder)
        self.assertIn('CLIPEncoder', repr_str)
        self.assertIn(self.test_model_name, repr_str)
        self.assertIn(self.test_device, repr_str)
        self.assertIn(str(self.expected_embedding_dim), repr_str)


class TestCLIPEncoderEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = CLIPEncoder(model_name='ViT-B-32', device='cpu')
    
    def test_encode_image_min_size(self):
        """Test encoding very small image (will be resized by preprocessing)."""
        small_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        # Should still work (preprocessing resizes)
        embedding = self.encoder.encode_image(small_image)
        self.assertEqual(embedding.shape, (512,))
    
    def test_encode_image_large_size(self):
        """Test encoding large image (will be resized by preprocessing)."""
        large_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        # Should still work (preprocessing resizes)
        embedding = self.encoder.encode_image(large_image)
        self.assertEqual(embedding.shape, (512,))
    
    def test_encode_image_non_square(self):
        """Test encoding non-square image."""
        rect_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        # Should work (preprocessing handles aspect ratio)
        embedding = self.encoder.encode_image(rect_image)
        self.assertEqual(embedding.shape, (512,))
    
    def test_encode_batch_single_image(self):
        """Test batch encoding with batch size 1."""
        images = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
        
        embeddings = self.encoder.encode_batch(images)
        self.assertEqual(embeddings.shape, (1, 512))
    
    def test_encode_image_all_black(self):
        """Test encoding all-black image."""
        black_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        embedding = self.encoder.encode_image(black_image)
        
        # Should produce valid embedding
        self.assertTrue(np.isfinite(embedding).all())
        self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)
    
    def test_encode_image_all_white(self):
        """Test encoding all-white image."""
        white_image = np.full((224, 224, 3), 255, dtype=np.uint8)
        
        embedding = self.encoder.encode_image(white_image)
        
        # Should produce valid embedding
        self.assertTrue(np.isfinite(embedding).all())
        self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
