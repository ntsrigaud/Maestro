import torch
from recognition_engine.temporal.vit_lstm_pipeline import ViTFrameFeatureExtractor
from recognition_engine.temporal.lstm_model import BiLSTMAttention

class DynamicGestureRecognizer:
    """
    Pipeline for dynamic gesture recognition using ViT + LSTM.
    """
    def __init__(self, vit_model_name, lstm_params, gesture_labels, device='cpu'):
        self.device = device
        self.feature_extractor = ViTFrameFeatureExtractor(vit_model_name, device)
        self.lstm = BiLSTMAttention(**lstm_params).to(device)
        self.labels = gesture_labels

    def predict(self, frames):
        """
        Args:
            frames: List of frames (PIL.Image or np.ndarray)
        Returns:
            dict: { 'gesture': str, 'confidence': float, 'description': str }
        """
        with torch.no_grad():
            feats = self.feature_extractor.extract_features(frames).unsqueeze(0).to(self.device)
            logits, attn = self.lstm(feats)
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            gesture = self.labels[pred.item()]
            description = self._describe_gesture(gesture, attn.squeeze(0).cpu().numpy())
            return {'gesture': gesture, 'confidence': conf.item(), 'description': description}

    def _describe_gesture(self, gesture, attn_weights):
        # Placeholder: Map gesture and attention to a semantic description
        return f"Detected gesture: {gesture} (dynamic evolution captured)"
