
import torch
from collections import deque, Counter
from recognition_engine.temporal.vit_lstm_pipeline import ViTFrameFeatureExtractor


class DynamicGestureRecognizer:
    """
    Buffer-based dynamic gesture recognizer using ViT predictions over time.
    LSTM is optional and not required for buffer-based dynamic meaning.
    """
    def __init__(self, vit_model_name, gesture_labels, buffer_size=16, device='cpu'):
        self.device = device
        self.feature_extractor = ViTFrameFeatureExtractor(vit_model_name, device)
        self.labels = gesture_labels
        self.pred_buffer = deque(maxlen=buffer_size)


    def add_prediction(self, gesture):
        self.pred_buffer.append(gesture)

    def infer_dynamic_meaning(self):
        if not self.pred_buffer:
            return "No gesture detected"
        counts = Counter(self.pred_buffer)
        most_common, freq = counts.most_common(1)[0]
        if freq > len(self.pred_buffer) * 0.7:
            return f"Holding: {most_common}"
        if len(set(self.pred_buffer)) > 1:
            if self.pred_buffer[0] != self.pred_buffer[-1]:
                return f"Transition: {self.pred_buffer[0]} → {self.pred_buffer[-1]}"
            else:
                return f"Changing gesture"
        return f"Gesture: {most_common}"

    def _describe_gesture(self, gesture, attn_weights):
        # Placeholder: Map gesture and attention to a semantic description
        return f"Detected gesture: {gesture} (dynamic evolution captured)"
