"""
LSTM model wrapper for real-time gesture recognition inference.

This module loads the pre-trained LSTM model (best_lstm_model79.h5)
and provides a simple interface for dynamic gesture prediction.
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LSTMGestureModel:
    """
    Wrapper for pre-trained LSTM gesture recognition model.

    Expects input sequences of shape (batch_size, sequence_length, feature_dim)
    where features are extracted from ViT (typically 768-dim).
    """

    def __init__(
        self,
        model_path: str = "notebooks/best_lstm_model79.h5",
        num_classes: int = 10,
        sequence_length: int = 16,
        feature_dim: int = 768,
    ):
        """
        Initialize LSTM model.

        Args:
            model_path: Path to trained LSTM model (best_lstm_model79.h5)
            num_classes: Number of gesture classes
            sequence_length: Expected sequence length for LSTM
            feature_dim: Feature dimension (ViT output = 768)

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model loading fails
        """
        self.model_path = Path(model_path)
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim

        if not self.model_path.exists():
            raise FileNotFoundError(f"LSTM model not found: {self.model_path}")

        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
            logger.info(f"✓ Loaded LSTM model from {self.model_path}")
            logger.info(f"  Model shape: input {self.model.input_shape}")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            raise ValueError(f"Invalid LSTM model: {e}")

    def predict(
        self,
        feature_sequence: np.ndarray,
        return_top_k: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict gesture class from feature sequence.

        Args:
            feature_sequence: Sequence of shape (sequence_length, feature_dim)
                             or (batch_size, sequence_length, feature_dim)
            return_top_k: Return top-k predictions

        Returns:
            (class_indices, confidences) - Top-k predictions sorted by confidence

        Example:
            >>> model = LSTMGestureModel()
            >>> features = np.random.randn(16, 768)  # (T, 768)
            >>> classes, confidences = model.predict(features)
            >>> print(f"Predicted: class {classes[0]} with confidence {confidences[0]:.2f}")
        """
        # Handle single sequence input
        if feature_sequence.ndim == 2:
            feature_sequence = np.expand_dims(feature_sequence, 0)

        # Validate shape
        if feature_sequence.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, "
                f"got {feature_sequence.shape[1]}"
            )

        if feature_sequence.shape[2] != self.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.feature_dim}, "
                f"got {feature_sequence.shape[2]}"
            )

        # Run inference
        predictions = self.model.predict(feature_sequence, verbose=0)

        # Get top-k per batch
        batch_classes = np.argsort(-predictions, axis=1)[:, :return_top_k]
        batch_confidences = np.take_along_axis(predictions, batch_classes, axis=1)

        return batch_classes[0], batch_confidences[0]

    def predict_single(
        self,
        feature_sequence: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Predict single gesture class (convenience method).

        Args:
            feature_sequence: Sequence of shape (sequence_length, feature_dim)

        Returns:
            (class_id, confidence)
        """
        classes, confidences = self.predict(feature_sequence, return_top_k=1)
        return int(classes[0]), float(confidences[0])

    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return "\n".join(summary_lines)


if __name__ == "__main__":
    # Test the model
    model = LSTMGestureModel()
    print(model.get_model_summary())

    # Test prediction with dummy data
    dummy_sequence = np.random.randn(16, 768).astype(np.float32)
    class_id, confidence = model.predict_single(dummy_sequence)
    print(f"\nTest prediction: class {class_id}, confidence {confidence:.4f}")
