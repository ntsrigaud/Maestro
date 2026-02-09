#!/usr/bin/env python3
"""
Gesture Recognition Inference Script
Uses fine-tuned ViT-B/16 model from HaGRID training
"""

import argparse
from pathlib import Path
import torch
import timm
from torchvision import transforms
from PIL import Image
import json

# Gesture classes (alphabetically sorted as in training)
CLASS_NAMES = [
    'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one',
    'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted',
    'three', 'three2', 'two_up', 'two_up_inverted'
]

class GestureClassifier:
    def __init__(self, checkpoint_path, device=None):
        """
        Initialize gesture classifier
        
        Args:
            checkpoint_path: Path to best_model.pth file
            device: 'cuda', 'mps', 'cpu', or None (auto-detect)
        """
        self.num_classes = len(CLASS_NAMES)
        self.class_names = CLASS_NAMES
        
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = timm.create_model('vit_base_patch16_224', 
                                       pretrained=False, 
                                       num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transform (matches training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Model loaded from {checkpoint_path}")
    
    def predict(self, image_path, top_k=3):
        """
        Predict gesture from image
        
        Args:
            image_path: Path to image file
            top_k: Return top-k predictions
            
        Returns:
            List of (gesture_name, confidence) tuples
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append((self.class_names[idx.item()], prob.item()))
        
        return results
    
    def predict_batch(self, image_paths, batch_size=32):
        """
        Predict gestures for multiple images
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for inference
            
        Returns:
            List of prediction results
        """
        from torch.utils.data import DataLoader, Dataset
        
        class ImageDataset(Dataset):
            def __init__(self, paths, transform):
                self.paths = paths
                self.transform = transform
            
            def __len__(self):
                return len(self.paths)
            
            def __getitem__(self, idx):
                img = Image.open(self.paths[idx]).convert('RGB')
                return self.transform(img), str(self.paths[idx])
        
        dataset = ImageDataset(image_paths, self.transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_results = []
        with torch.no_grad():
            for images, paths in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                for path, pred, prob_dist in zip(paths, preds, probs):
                    all_results.append({
                        'image': Path(path).name,
                        'gesture': self.class_names[pred],
                        'confidence': prob_dist[pred].item()
                    })
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description='Gesture Recognition Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to best_model.pth checkpoint')
    parser.add_argument('--image', type=str,
                       help='Path to single image file')
    parser.add_argument('--images', type=str,
                       help='Path to directory containing images')
    parser.add_argument('--output', type=str,
                       help='Save results to JSON file')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Return top-k predictions (default: 3)')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = GestureClassifier(args.checkpoint, device=args.device)
    
    # Single image inference
    if args.image:
        print(f"\nPredicting: {args.image}")
        results = classifier.predict(args.image, top_k=args.top_k)
        
        print("\nTop predictions:")
        for i, (gesture, conf) in enumerate(results, 1):
            print(f"  {i}. {gesture:20s} {conf:6.2%}")
    
    # Batch inference
    elif args.images:
        image_dir = Path(args.images)
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        if not image_paths:
            print(f"No images found in {image_dir}")
            return
        
        print(f"\nProcessing {len(image_paths)} images...")
        results = classifier.predict_batch(image_paths)
        
        print("\nResults:")
        for r in results:
            print(f"  {r['image']:30s} -> {r['gesture']:20s} ({r['confidence']:6.2%})")
        
        # Save to JSON if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Saved results to {args.output}")
    
    else:
        print("Please provide --image or --images argument")
        parser.print_help()


if __name__ == '__main__':
    main()
