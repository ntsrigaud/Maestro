import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from recognition_engine.temporal.dynamic_gesture_recognizer import DynamicGestureRecognizer
from torchvision import transforms
from PIL import Image
import os

class GestureSequenceDataset(Dataset):
    def __init__(self, root_dir, gesture_labels, max_frames=16):
        self.samples = []
        self.labels = gesture_labels
        self.max_frames = max_frames
        for label in gesture_labels:
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for seq in os.listdir(label_dir):
                seq_dir = os.path.join(label_dir, seq)
                if os.path.isdir(seq_dir):
                    self.samples.append((seq_dir, label))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_dir, label = self.samples[idx]
        files = sorted([f for f in os.listdir(seq_dir) if f.endswith(('.jpg', '.png'))])[:self.max_frames]
        frames = [self.transform(Image.open(os.path.join(seq_dir, f)).convert('RGB')) for f in files]
        frames = [transforms.ToPILImage()(f) for f in frames]
        label_idx = self.labels.index(label)
        return frames, label_idx

def collate_fn(batch):
    frames_batch, labels = zip(*batch)
    return list(frames_batch), torch.tensor(labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--vit_model', type=str, default='vit_base_patch16_224')
    parser.add_argument('--lstm_weights', type=str, required=False)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    gesture_labels = ['swipe_left', 'swipe_right', 'zoom', 'pinch', 'wave']  # Example
    lstm_params = dict(input_dim=768, hidden_dim=256, num_layers=2, num_classes=len(gesture_labels))
    recognizer = DynamicGestureRecognizer(args.vit_model, lstm_params, gesture_labels, device=args.device)
    if args.lstm_weights:
        recognizer.lstm.load_state_dict(torch.load(args.lstm_weights, map_location=args.device))
    recognizer.lstm.train()

    dataset = GestureSequenceDataset(args.data_dir, gesture_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(recognizer.lstm.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total_loss = 0
        for frames_batch, labels in loader:
            feats_batch = []
            for frames in frames_batch:
                feats = recognizer.feature_extractor.extract_features(frames)
                feats_batch.append(feats)
            feats_batch = torch.stack(feats_batch).to(args.device)
            labels = labels.to(args.device)
            logits, _ = recognizer.lstm(feats_batch)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(loader):.4f}")
    torch.save(recognizer.lstm.state_dict(), 'lstm_vit_best.pth')

if __name__ == '__main__':
    main()
