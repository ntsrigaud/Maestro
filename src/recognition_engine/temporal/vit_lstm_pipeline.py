import torch
import torch.nn as nn
from .vit_feature_extractor import ViTFeatureExtractor


from pathlib import Path

class ViTFrameFeatureExtractor:
    """
    Extracts features from a sequence of frames using a fine-tuned ViT model.
    """
    def __init__(self, checkpoint_path=None, num_classes=18, device='cpu'):
        self.device = device
        if checkpoint_path is None:
            # Default to user's fine-tuned model in checkpoints/best_model.pth
            checkpoint_path = str(Path(__file__).parent.parent.parent.parent / "checkpoints" / "best_model.pth")
        self.vit = ViTFeatureExtractor(checkpoint_path=checkpoint_path, num_classes=num_classes, device=device)


    def extract_features(self, frames):
        """
        Args:
            frames (List[PIL.Image] or np.ndarray): List or batch of frames.
        Returns:
            torch.Tensor: Feature tensor of shape (seq_len, feature_dim)
        """
        features = []
        for frame in frames:
            feat = self.vit.extract_feature(frame)
            features.append(feat)
        return torch.stack(features, dim=0)

    def extract_feature_single(self, frame):
        """
        Extract feature from a single frame for real-time use.
        Args:
            frame (PIL.Image or np.ndarray): Single frame.
        Returns:
            torch.Tensor: Feature vector.
        """
        return self.vit.extract_feature(frame)
