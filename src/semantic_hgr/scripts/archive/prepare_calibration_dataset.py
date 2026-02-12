#!/usr/bin/env python3
"""
Calibration Dataset Preparation

Purpose: Generate labeled validation dataset for calibration study by:
         1. Capturing gesture samples from camera
         2. Encoding gestures to CLIP embeddings
         3. Manually labeling with ground truth intents
         4. Saving as JSON for calibration analysis

This creates the dataset required for calibration_study.py

Usage:
    python scripts/prepare_calibration_dataset.py \
        --output data/validation_gestures.json \
        --samples 100 \
        --interactive

Controls (interactive mode):
    - Follow on-screen prompts to perform gestures
    - Type ground truth intent for each captured gesture
    - Press 's' to skip current sample
    - Press 'q' to quit and save partial dataset
"""

import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from semantic_hgr.input_manager.input_manager import InputManager
from semantic_hgr.recognition_engine.hand_detection.hand_detector import HandDetector
from semantic_hgr.representation import GestureEncoder


@dataclass
class LabeledSample:
    """Single labeled validation sample."""
    gesture_embedding: np.ndarray
    intent: str
    sample_id: int


class DatasetCollector:
    """Collect and label gesture samples for calibration validation."""
    
    def __init__(self, camera_index: int = 0):
        """Initialize data collection components."""
        print("Initializing components...")
        
        self.input_mgr = InputManager(camera_index=camera_index, buffer_size=10)
        self.hand_detector = HandDetector(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.encoder = GestureEncoder(
            model_name="ViT-B-32",
            pretrained="openai",
            device="cpu"
        )
        
        self.samples: List[LabeledSample] = []
        
        print("✓ Components initialized")
    
    def capture_and_encode_gesture(self) -> Optional[np.ndarray]:
        """
        Capture current frame, detect hand, encode to embedding.
        
        Returns:
            CLIP embedding (512-dim) or None if no hand detected
        """
        frame_data = self.input_mgr.get_latest_frame()
        if frame_data is None:
            return None
        
        frame = frame_data.frame
        
        # Detect hand
        detections = self.hand_detector.detect(frame)
        if not detections:
            return None
        
        # Extract hand region
        detection = detections[0]  # Use first hand
        x, y, w, h = detection.bbox
        hand_region = frame[y:y+h, x:x+w]
        
        if hand_region.size == 0:
            return None
        
        # Encode to CLIP embedding
        embedding = self.encoder.encode_image(hand_region)
        return embedding
    
    def collect_interactive(self, target_samples: int, intent_vocabulary: List[str]):
        """
        Collect samples interactively with user labeling.
        
        Args:
            target_samples: Number of samples to collect
            intent_vocabulary: List of valid intent labels
        """
        print("\n" + "="*70)
        print("INTERACTIVE DATA COLLECTION")
        print("="*70)
        print(f"\nTarget: {target_samples} samples")
        print(f"Intent vocabulary: {', '.join(intent_vocabulary)}")
        print("\nInstructions:")
        print("  1. Position hand in frame to perform gesture")
        print("  2. Press SPACE when gesture is stable")
        print("  3. Type ground truth intent label")
        print("  4. Press ENTER to confirm and continue")
        print("\nControls:")
        print("  SPACE = Capture gesture")
        print("  S = Skip current sample")
        print("  Q = Quit and save partial dataset")
        print("="*70 + "\n")
        
        # Start camera
        self.input_mgr.start_continuous_capture()
        
        sample_id = 0
        
        while len(self.samples) < target_samples:
            # Get frame for display
            frame_data = self.input_mgr.get_latest_frame()
            if frame_data is None:
                continue
            
            frame = frame_data.frame.copy()
            
            # Detect and visualize hand
            detections = self.hand_detector.detect(frame)
            
            # Draw detection overlay
            if detections:
                for detection in detections:
                    x, y, w, h = detection.bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{detection.handedness} {detection.confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw status
            status_text = f"Collected: {len(self.samples)}/{target_samples}"
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            instruction = "Press SPACE to capture gesture" if detections else "Show hand in frame"
            cv2.putText(frame, instruction, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Data Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE - capture
                if not detections:
                    print("✗ No hand detected - show hand in frame")
                    continue
                
                print(f"\nCapturing sample {sample_id + 1}...")
                embedding = self.capture_and_encode_gesture()
                
                if embedding is None:
                    print("✗ Failed to encode gesture - try again")
                    continue
                
                print("✓ Gesture captured and encoded")
                print(f"  Embedding shape: {embedding.shape}")
                
                # Get ground truth label from user
                print(f"\nAvailable intents: {', '.join(intent_vocabulary)}")
                intent = input("Enter ground truth intent: ").strip()
                
                if not intent:
                    print("✗ Empty label - skipping sample")
                    continue
                
                if intent not in intent_vocabulary:
                    print(f"⚠ Warning: '{intent}' not in predefined vocabulary")
                    confirm = input("Use this label anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        print("✗ Sample skipped")
                        continue
                
                # Add sample
                sample = LabeledSample(
                    gesture_embedding=embedding,
                    intent=intent,
                    sample_id=sample_id
                )
                self.samples.append(sample)
                sample_id += 1
                
                print(f"✓ Sample {sample_id} added: intent='{intent}'")
                print(f"  Progress: {len(self.samples)}/{target_samples}")
            
            elif key == ord('s'):  # Skip
                print("⊘ Sample skipped by user")
            
            elif key == ord('q'):  # Quit
                print("\n⚠ Quit requested - saving partial dataset")
                break
        
        cv2.destroyAllWindows()
        self.input_mgr.stop_capture()
        
        print(f"\n✓ Collection complete: {len(self.samples)} samples")
    
    def save_dataset(self, output_path: Path):
        """
        Save collected samples as JSON dataset.
        
        Args:
            output_path: Path to save dataset JSON
        """
        if not self.samples:
            print("✗ No samples to save")
            return
        
        dataset = {
            "metadata": {
                "num_samples": len(self.samples),
                "encoder_model": "ViT-B-32",
                "encoder_pretrained": "openai",
                "embedding_dim": 512,
                "collection_date": "2026-02-11"
            },
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "intent": sample.intent,
                    "gesture_embedding": sample.gesture_embedding.tolist()
                }
                for sample in self.samples
            ]
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n✓ Dataset saved: {output_path}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    def get_intent_distribution(self) -> Dict[str, int]:
        """Get distribution of intents in collected samples."""
        distribution = {}
        for sample in self.samples:
            distribution[sample.intent] = distribution.get(sample.intent, 0) + 1
        return distribution
    
    def cleanup(self):
        """Cleanup resources."""
        self.input_mgr.stop_capture()
        self.hand_detector.close()
        cv2.destroyAllWindows()


def create_synthetic_dataset(output_path: Path, 
                            num_samples: int,
                            intent_vocabulary: List[str]):
    """
    Create synthetic dataset for testing (uses random embeddings).
    
    WARNING: This is for testing the calibration pipeline only.
             Real calibration requires actual gesture samples!
    
    Args:
        output_path: Path to save synthetic dataset
        num_samples: Number of samples to generate
        intent_vocabulary: List of intent labels to randomly assign
    """
    print("\n⚠ WARNING: Creating SYNTHETIC dataset for testing")
    print("  This should NOT be used for real calibration analysis!")
    print("  Use --interactive mode to collect real gesture samples.\n")
    
    np.random.seed(42)
    
    samples = []
    for i in range(num_samples):
        # Generate random normalized embedding
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Random intent
        intent = np.random.choice(intent_vocabulary)
        
        samples.append({
            "sample_id": i,
            "intent": intent,
            "gesture_embedding": embedding.tolist()
        })
    
    dataset = {
        "metadata": {
            "num_samples": num_samples,
            "encoder_model": "ViT-B-32 (SYNTHETIC)",
            "encoder_pretrained": "openai",
            "embedding_dim": 512,
            "collection_date": "2026-02-11",
            "WARNING": "SYNTHETIC DATA - NOT FOR REAL CALIBRATION"
        },
        "samples": samples
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✓ Synthetic dataset saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare calibration validation dataset"
    )
    parser.add_argument('--output', type=Path, required=True,
                        help='Path to save validation dataset JSON')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples to collect')
    parser.add_argument('--interactive', action='store_true',
                        help='Collect samples interactively from camera')
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic dataset (testing only)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index for capture')
    
    args = parser.parse_args()
    
    # Default intent vocabulary (can be customized)
    intent_vocabulary = [
        "scroll_up", "scroll_down", "scroll_left", "scroll_right",
        "zoom_in", "zoom_out",
        "swipe_left", "swipe_right", "swipe_up", "swipe_down",
        "click", "double_click", "right_click",
        "drag", "drop", "select",
        "rotate_cw", "rotate_ccw",
        "volume_up", "volume_down"
    ]
    
    print("="*70)
    print("CALIBRATION DATASET PREPARATION")
    print("="*70)
    
    if args.synthetic:
        create_synthetic_dataset(args.output, args.samples, intent_vocabulary)
    
    elif args.interactive:
        collector = DatasetCollector(camera_index=args.camera)
        
        try:
            collector.collect_interactive(args.samples, intent_vocabulary)
            
            # Display intent distribution
            distribution = collector.get_intent_distribution()
            print("\n" + "="*70)
            print("INTENT DISTRIBUTION")
            print("="*70)
            for intent, count in sorted(distribution.items(), key=lambda x: -x[1]):
                print(f"  {intent:<20} {count:>3} samples")
            print("="*70)
            
            collector.save_dataset(args.output)
        
        finally:
            collector.cleanup()
    
    else:
        print("\n✗ Error: Must specify either --interactive or --synthetic")
        print("  Use --interactive for real data collection")
        print("  Use --synthetic for testing (not real calibration)")
        return 1
    
    print("\n✓ Dataset preparation complete")
    print(f"  Use this dataset with: python scripts/calibration_study.py --dataset {args.output}")
    

if __name__ == "__main__":
    main()
