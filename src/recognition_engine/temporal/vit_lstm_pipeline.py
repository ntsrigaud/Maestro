import torch
import torch.nn as nn
from .vit_feature_extractor import ViTFeatureExtractor

class ViTFrameFeatureExtractor:
    """
    Extracts features from a sequence of frames using a ViT model.
    """
    def __init__(self, vit_model_name='vit_base_patch16_224', device='cpu'):
        self.device = device
        self.vit = ViTFeatureExtractor(model_name=vit_model_name, device=device)

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
