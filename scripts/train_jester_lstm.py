import torch
from torch.utils.data import DataLoader
from recognition_engine.temporal.jester_dataset import JesterGestureSequenceDataset
from recognition_engine.temporal.vit_lstm_pipeline import ViTFrameFeatureExtractor
from recognition_engine.temporal.lstm_model import BiLSTMAttention

# Example gesture labels for Jester (replace with actual)
gesture_labels = [
    'Swiping Left', 'Swiping Right', 'Swiping Down', 'Swiping Up',
    'Pushing Hand Away', 'Pulling Hand In', 'Rolling Hand Forward', 'Rolling Hand Backward',
    # ... add all Jester classes ...
]

def collate_fn(batch):
    frames_batch, labels = zip(*batch)
    return list(frames_batch), torch.tensor(labels)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    dataset = JesterGestureSequenceDataset(args.data_dir, gesture_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    feature_extractor = ViTFrameFeatureExtractor(checkpoint_path=args.checkpoint, num_classes=len(gesture_labels), device=args.device)
    lstm_params = dict(input_dim=768, hidden_dim=256, num_layers=2, num_classes=len(gesture_labels))
    lstm = BiLSTMAttention(**lstm_params).to(args.device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total_loss = 0
        for frames_batch, labels in loader:
            feats_batch = []
            for frames in frames_batch:
                feats = feature_extractor.extract_features([transforms.ToPILImage()(f) for f in frames])
                feats_batch.append(feats)
            feats_batch = torch.stack(feats_batch).to(args.device)  # (batch, seq_len, feat_dim)
            labels = labels.to(args.device)
            logits, _ = lstm(feats_batch)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(loader):.4f}")
    torch.save(lstm.state_dict(), 'lstm_jester_best.pth')

if __name__ == '__main__':
    main()
