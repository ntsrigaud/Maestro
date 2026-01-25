import torch
import torch.nn as nn
import timm


class ViTFeatureExtractor(nn.Module):
    """
    Uses the fine-tuned HaGRID ViT and outputs embeddings (not class logits).
    """

    def __init__(self, checkpoint_path, num_classes, device):
        super().__init__()
        self.device = device

        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=num_classes
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if "model_state_dict" in checkpoint:
            self.vit.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.vit.load_state_dict(checkpoint)

        # Remove classifier head → feature extractor
        self.vit.reset_classifier(0)

        # Freeze ViT
        for p in self.vit.parameters():
            p.requires_grad = False

        self.vit.to(device)
        self.vit.eval()

    def forward(self, x):
        """
        Input:  (B, 3, 224, 224)
        Output: (B, 768)
        """
        with torch.no_grad():
            return self.vit(x)
