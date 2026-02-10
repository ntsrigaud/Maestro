#!/usr/bin/env python3
"""
Dataset Class Balance Audit Script
Analyzes class distribution across splits and computes imbalance metrics.
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import entropy

def count_images_per_class(split_path):
    """Count images per class in a split directory."""
    if not split_path.exists():
        return {}
    
    class_counts = {}
    for class_dir in sorted(split_path.iterdir()):
        if class_dir.is_dir():
            # Count image files (jpg, jpeg, png)
            img_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
            count = sum(1 for f in class_dir.iterdir() 
                       if f.is_file() and f.suffix in img_extensions)
            class_counts[class_dir.name] = count
    
    return class_counts


def compute_imbalance_metrics(class_counts):
    """Compute imbalance indicators."""
    if not class_counts:
        return {}
    
    counts = np.array(list(class_counts.values()))
    total = counts.sum()
    
    if total == 0:
        return {}
    
    # Basic stats
    max_count = counts.max()
    min_count = counts.min()
    mean_count = counts.mean()
    std_count = counts.std()
    
    # Imbalance ratio (max/min)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    # Gini coefficient
    sorted_counts = np.sort(counts)
    n = len(counts)
    gini = (2 * np.sum((n - np.arange(n)) * sorted_counts)) / (n * total) - 1
    
    # Entropy (normalized)
    probs = counts / total
    entropy_val = entropy(probs)
    max_entropy = np.log(n)
    normalized_entropy = entropy_val / max_entropy if max_entropy > 0 else 0
    
    return {
        'total_samples': int(total),
        'num_classes': int(n),
        'max_count': int(max_count),
        'min_count': int(min_count),
        'mean_count': float(mean_count),
        'std_count': float(std_count),
        'imbalance_ratio': float(imbalance_ratio),
        'gini_coefficient': float(gini),
        'normalized_entropy': float(normalized_entropy)
    }


def audit_dataset(dataset_root):
    """Audit class balance across all splits."""
    dataset_root = Path(dataset_root)
    
    results = {
        'dataset_root': str(dataset_root),
        'splits': {}
    }
    
    for split in ['train', 'val', 'test']:
        split_path = dataset_root / split
        
        if not split_path.exists():
            print(f"Warning: {split} split not found at {split_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Analyzing {split.upper()} split...")
        print(f"{'='*60}")
        
        class_counts = count_images_per_class(split_path)
        metrics = compute_imbalance_metrics(class_counts)
        
        results['splits'][split] = {
            'class_counts': class_counts,
            'metrics': metrics
        }
        
        # Print results
        print(f"\nClass Counts ({len(class_counts)} classes):")
        print(f"{'Class':<20} {'Count':>10} {'Percentage':>12}")
        print(f"{'-'*44}")
        
        total = sum(class_counts.values())
        for cls_name in sorted(class_counts.keys()):
            count = class_counts[cls_name]
            pct = 100 * count / total if total > 0 else 0
            print(f"{cls_name:<20} {count:>10} {pct:>11.2f}%")
        
        print(f"\n{'-'*44}")
        print(f"{'Total':<20} {total:>10} {'100.00%':>12}")
        
        if metrics:
            print(f"\nImbalance Metrics:")
            print(f"  Imbalance Ratio (max/min): {metrics['imbalance_ratio']:.2f}")
            print(f"  Gini Coefficient: {metrics['gini_coefficient']:.3f} (0=perfect balance, 1=max imbalance)")
            print(f"  Normalized Entropy: {metrics['normalized_entropy']:.3f} (1=perfect balance, 0=max imbalance)")
            print(f"  Mean ± Std: {metrics['mean_count']:.1f} ± {metrics['std_count']:.1f}")
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    all_classes = set()
    for split_data in results['splits'].values():
        all_classes.update(split_data['class_counts'].keys())
    
    print(f"\nTotal unique classes: {len(all_classes)}")
    print(f"Classes: {sorted(all_classes)}")
    
    # Check class consistency
    if len(results['splits']) > 1:
        split_keys = list(results['splits'].keys())
        reference_classes = set(results['splits'][split_keys[0]]['class_counts'].keys())
        
        print(f"\nClass Consistency Check:")
        for split in split_keys[1:]:
            current_classes = set(results['splits'][split]['class_counts'].keys())
            if current_classes != reference_classes:
                missing = reference_classes - current_classes
                extra = current_classes - reference_classes
                print(f"  {split}: INCONSISTENT")
                if missing:
                    print(f"    Missing: {missing}")
                if extra:
                    print(f"    Extra: {extra}")
            else:
                print(f"  {split}: CONSISTENT")
    
    # Decision gate recommendation
    print(f"\n{'='*60}")
    print("DECISION GATE RECOMMENDATION")
    print(f"{'='*60}")
    
    train_metrics = results['splits'].get('train', {}).get('metrics', {})
    if train_metrics:
        ratio = train_metrics.get('imbalance_ratio', 0)
        
        print(f"\nTrain Imbalance Ratio: {ratio:.2f}")
        
        if ratio > 2.0:
            print("  ⚠️  HIGH IMBALANCE (>2×)")
            print("  Recommendation: Use WeightedRandomSampler + class-weighted loss")
        elif ratio > 1.5:
            print("  ⚠️  MODERATE IMBALANCE (>1.5×)")
            print("  Recommendation: Try class-weighted loss first; add sampler if needed")
        else:
            print("  ✓ LOW IMBALANCE (<1.5×)")
            print("  Recommendation: Class-weighted loss may be sufficient")
    
    return results


def save_results(results, output_path):
    """Save audit results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to {output_path}")


def main():
    """Main execution."""
    if len(sys.argv) < 2:
        print("Usage: python audit_dataset_balance.py <dataset_root> [output_json]")
        print("\nExample:")
        print("  python audit_dataset_balance.py data/hagrid_subset_cropped")
        print("  python audit_dataset_balance.py data/hagrid_subset notebooks/training_results/class_counts.json")
        sys.exit(1)
    
    dataset_root = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else 'notebooks/training_results/class_counts.json'
    
    results = audit_dataset(dataset_root)
    save_results(results, output_json)
    
    print(f"\n{'='*60}")
    print("AUDIT COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
