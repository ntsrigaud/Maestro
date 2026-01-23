import argparse
import torch
from recognition_engine.temporal.dynamic_gesture_recognizer import DynamicGestureRecognizer
from torchvision import transforms
from PIL import Image
import os

def load_frames_from_folder(folder, max_frames=16):
    files = sorted([f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))])
    files = files[:max_frames]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    frames = [transform(Image.open(os.path.join(folder, f)).convert('RGB')) for f in files]
    return [transforms.ToPILImage()(f) for f in frames]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=str, required=True, help='Directory with gesture frames')
    parser.add_argument('--vit_model', type=str, default='vit_base_patch16_224')
    parser.add_argument('--lstm_weights', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    gesture_labels = ['swipe_left', 'swipe_right', 'zoom', 'pinch', 'wave']  # Example
    lstm_params = dict(input_dim=768, hidden_dim=256, num_layers=2, num_classes=len(gesture_labels))
    recognizer = DynamicGestureRecognizer(args.vit_model, lstm_params, gesture_labels, device=args.device)
    recognizer.lstm.load_state_dict(torch.load(args.lstm_weights, map_location=args.device))
    recognizer.lstm.eval()

    frames = load_frames_from_folder(args.frames_dir)
    result = recognizer.predict(frames)
    print(result)

if __name__ == '__main__':
    main()
