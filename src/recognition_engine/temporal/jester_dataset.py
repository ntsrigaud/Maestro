import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class JesterGestureSequenceDataset(Dataset):
    """
    Loads sequences of frames from the Jester dataset for dynamic gesture recognition.
    Each sample is a sequence of frames (as PIL Images) and a gesture label.
    """
    def __init__(self, root_dir, gesture_labels, sequence_length=16, transform=None):
        self.root_dir = root_dir
        self.gesture_labels = gesture_labels
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.samples = self._gather_samples()

    def _gather_samples(self):
        samples = []
        for label in self.gesture_labels:
            label_dir = os.path.join(self.root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for seq in os.listdir(label_dir):
                seq_dir = os.path.join(label_dir, seq)
                if os.path.isdir(seq_dir):
                    frame_files = sorted([f for f in os.listdir(seq_dir) if f.endswith(('.jpg', '.png'))])
                    if len(frame_files) >= self.sequence_length:
                        samples.append((seq_dir, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_dir, label = self.samples[idx]
        frame_files = sorted([f for f in os.listdir(seq_dir) if f.endswith(('.jpg', '.png'))])[:self.sequence_length]
        frames = [self.transform(Image.open(os.path.join(seq_dir, f)).convert('RGB')) for f in frame_files]
        label_idx = self.gesture_labels.index(label)
        return torch.stack(frames), label_idx
