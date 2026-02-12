#!/usr/bin/env python3
"""
Intent Overlap Analysis and Resolution

Purpose: Analyze semantic overlap between intent embeddings and design
         resolution strategy to reduce confusion and improve accuracy.

Analysis Components:
- Pairwise cosine similarity matrix
- Confusion matrix from validation data  
- High-overlap cluster identification
- Root cause investigation

Resolution Strategies:
- Intent merging (semantically redundant)
- Intent rewording (improved separation)
- Hierarchical classification (coarse → fine)
- Rejection thresholds for ambiguous cases

Usage:
    python scripts/intent_overlap_analysis.py \
        --dataset data/validation_gestures.json \
        --output results/overlap_analysis.json \
        --similarity-threshold 0.85 \
        --apply-resolution
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from sklearn.metrics import confusion_matrix
import warnings

from semantic_hgr.representation.clip_encoder import CLIPEncoder
from semantic_hgr.semantics.clip_intent import SemanticIntentExtractor


@dataclass
class OverlapPair:
    """Pair of intents with high semantic overlap."""
    intent_1: str
    intent_2: str
    similarity: float
    confusion_count: int = 0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class IntentCluster:
    """Cluster of semantically overlapping intents."""
    intents: List[str]
    avg_similarity: float
    resolution_strategy: str
    
    def to_dict(self):
        return asdict(self)


class IntentOverlapAnalyzer:
    """Analyze and resolve semantic overlap between intents."""
    
    def __init__(self, intent_extractor: SemanticIntentExtractor):
        """
        Initialize overlap analyzer.
        
        Args:
            intent_extractor: Semantic intent extraction model
        """
        self.extractor = intent_extractor
        self.intent_names = extractor.intent_names
        self.text_embeddings = extractor.text_embeddings
        
        self.similarity_matrix: Optional[np.ndarray] = None
        self.confusion_mat: Optional[np.ndarray] = None
        self.high_overlap_pairs: List[OverlapPair] = []
        self.clusters: List[IntentCluster] = []
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Compute pairwise cosine similarity between all intent embeddings.
        
        Returns:
            N×N similarity matrix where N = number of intents
        """
        print(f"\nComputing similarity matrix for {len(self.intent_names)} intents...")
        
        # text_embeddings is already L2-normalized, so dot product = cosine similarity
        similarity_matrix = np.dot(self.text_embeddings, self.text_embeddings.T)
        
        self.similarity_matrix = similarity_matrix
        
        # Statistics
        n = len(self.intent_names)
        triu_indices = np.triu_indices(n, k=1)  # Upper triangle, excluding diagonal
        similarities = similarity_matrix[triu_indices]
        
        print(f"✓ Similarity matrix computed")
        print(f"  Mean similarity: {similarities.mean():.4f}")
        print(f"  Std similarity: {similarities.std():.4f}")
        print(f"  Min similarity: {similarities.min():.4f}")
        print(f"  Max similarity: {similarities.max():.4f}")
        
        return similarity_matrix
    
    def identify_high_overlap_pairs(self, threshold: float = 0.85) -> List[OverlapPair]:
        """
        Identify pairs of intents with similarity above threshold.
        
        Args:
            threshold: Minimum cosine similarity to flag as high overlap
            
        Returns:
            List of high-overlap pairs sorted by similarity
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        print(f"\nIdentifying high-overlap pairs (threshold: {threshold})...")
        
        pairs = []
        n = len(self.intent_names)
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self.similarity_matrix[i, j]
                if similarity >= threshold:
                    pairs.append(OverlapPair(
                        intent_1=self.intent_names[i],
                        intent_2=self.intent_names[j],
                        similarity=float(similarity)
                    ))
        
        # Sort by similarity descending
        pairs.sort(key=lambda p: p.similarity, reverse=True)
        
        self.high_overlap_pairs = pairs
        
        print(f"✓ Found {len(pairs)} high-overlap pairs")
        if pairs:
            print(f"  Top overlap: {pairs[0].intent_1} ↔ {pairs[0].intent_2} "
                  f"(sim={pairs[0].similarity:.4f})")
        
        return pairs
    
    def compute_confusion_matrix(self, 
                                 validation_samples: List[Tuple[np.ndarray, str]]) -> np.ndarray:
        """
        Compute confusion matrix from validation samples.
        
        Args:
            validation_samples: List of (embedding, true_intent) tuples
            
        Returns:
            N×N confusion matrix
        """
        print(f"\nComputing confusion matrix from {len(validation_samples)} samples...")
        
        true_labels = []
        pred_labels = []
        
        for i, (embedding, true_intent) in enumerate(validation_samples):
            results = self.extractor.extract_intent(embedding, top_k=1, min_confidence=0.0)
            
            if results:
                pred_intent = results[0]['intent']
            else:
                pred_intent = "UNKNOWN"
            
            true_labels.append(true_intent)
            pred_labels.append(pred_intent)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(validation_samples)} samples")
        
        # Build confusion matrix
        all_labels = sorted(set(true_labels + pred_labels))
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        
        true_indices = [label_to_idx[label] for label in true_labels]
        pred_indices = [label_to_idx[label] for label in pred_labels]
        
        confusion_mat = confusion_matrix(true_indices, pred_indices, labels=range(len(all_labels)))
        
        self.confusion_mat = confusion_mat
        self.confusion_labels = all_labels
        
        # Compute accuracy
        accuracy = np.trace(confusion_mat) / confusion_mat.sum()
        
        print(f"✓ Confusion matrix computed")
        print(f"  Overall accuracy: {accuracy:.4f}")
        print(f"  Unique labels: {len(all_labels)}")
        
        return confusion_mat
    
    def find_confusion_pairs(self, min_confusion_count: int = 3) -> List[Tuple[str, str, int]]:
        """
        Find frequently confused intent pairs from confusion matrix.
        
        Args:
            min_confusion_count: Minimum number of confusions to report
            
        Returns:
            List of (intent_1, intent_2, confusion_count) tuples
        """
        if self.confusion_mat is None:
            raise ValueError("Compute confusion matrix first")
        
        print(f"\nFinding confusion pairs (min count: {min_confusion_count})...")
        
        confusion_pairs = []
        n = len(self.confusion_labels)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # Skip diagonal
                
                count = int(self.confusion_mat[i, j])
                if count >= min_confusion_count:
                    confusion_pairs.append((
                        self.confusion_labels[i],
                        self.confusion_labels[j],
                        count
                    ))
        
        # Sort by count descending
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"✓ Found {len(confusion_pairs)} confusion pairs")
        if confusion_pairs:
            print(f"  Top confusion: {confusion_pairs[0][0]} → {confusion_pairs[0][1]} "
                  f"({confusion_pairs[0][2]} times)")
        
        return confusion_pairs
    
    def correlate_overlap_and_confusion(self):
        """
        Correlate high embedding similarity with actual confusion.
        Update high_overlap_pairs with confusion counts.
        """
        if not self.high_overlap_pairs or self.confusion_mat is None:
            raise ValueError("Compute similarity and confusion first")
        
        print("\nCorrelating embedding similarity with confusion...")
        
        confusion_pairs = self.find_confusion_pairs(min_confusion_count=1)
        confusion_dict = {(p[0], p[1]): p[2] for p in confusion_pairs}
        
        for pair in self.high_overlap_pairs:
            # Check both directions
            count_1 = confusion_dict.get((pair.intent_1, pair.intent_2), 0)
            count_2 = confusion_dict.get((pair.intent_2, pair.intent_1), 0)
            pair.confusion_count = count_1 + count_2
        
        # Sort by confusion count
        self.high_overlap_pairs.sort(key=lambda p: p.confusion_count, reverse=True)
        
        print("✓ Correlation complete")
        
        # Report top correlated pairs
        correlated = [p for p in self.high_overlap_pairs if p.confusion_count > 0]
        print(f"  {len(correlated)}/{len(self.high_overlap_pairs)} overlap pairs show confusion")
        
        if correlated:
            top = correlated[0]
            print(f"  Top: {top.intent_1} ↔ {top.intent_2}")
            print(f"       Similarity: {top.similarity:.4f}, Confusions: {top.confusion_count}")
    
    def form_overlap_clusters(self, similarity_threshold: float = 0.85) -> List[IntentCluster]:
        """
        Form clusters of intents with high mutual similarity.
        
        Args:
            similarity_threshold: Minimum similarity for cluster membership
            
        Returns:
            List of intent clusters
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        print(f"\nForming overlap clusters (threshold: {similarity_threshold})...")
        
        n = len(self.intent_names)
        visited = set()
        clusters = []
        
        for i in range(n):
            if i in visited:
                continue
            
            # Find all intents similar to i
            cluster_indices = [i]
            for j in range(i + 1, n):
                if self.similarity_matrix[i, j] >= similarity_threshold:
                    cluster_indices.append(j)
            
            # Check if cluster members are mutually similar
            if len(cluster_indices) > 1:
                cluster_intents = [self.intent_names[idx] for idx in cluster_indices]
                
                # Compute average similarity within cluster
                similarities = []
                for idx1 in cluster_indices:
                    for idx2 in cluster_indices:
                        if idx1 < idx2:
                            similarities.append(self.similarity_matrix[idx1, idx2])
                
                avg_sim = np.mean(similarities) if similarities else 1.0
                
                cluster = IntentCluster(
                    intents=cluster_intents,
                    avg_similarity=float(avg_sim),
                    resolution_strategy="TBD"
                )
                clusters.append(cluster)
                
                visited.update(cluster_indices)
        
        self.clusters = clusters
        
        print(f"✓ Formed {len(clusters)} clusters")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i+1}: {len(cluster.intents)} intents, "
                  f"avg_sim={cluster.avg_similarity:.4f}")
            print(f"    Intents: {', '.join(cluster.intents[:3])}{'...' if len(cluster.intents) > 3 else ''}")
        
        return clusters
    
    def propose_resolution_strategy(self) -> Dict[str, List[Dict]]:
        """
        Propose resolution strategies for overlap and confusion.
        
        Returns:
            Dictionary of resolution recommendations
        """
        print("\n" + "="*70)
        print("RESOLUTION STRATEGY PROPOSAL")
        print("="*70)
        
        strategies = {
            "merge_candidates": [],
            "reword_candidates": [],
            "hierarchical_candidates": [],
            "rejection_candidates": []
        }
        
        # Analyze each high-overlap pair
        for pair in self.high_overlap_pairs:
            intent_1, intent_2 = pair.intent_1, pair.intent_2
            sim = pair.similarity
            conf = pair.confusion_count
            
            # Decision logic
            if sim >= 0.95 and conf > 5:
                # Very high similarity + high confusion → merge
                strategies["merge_candidates"].append({
                    "intent_1": intent_1,
                    "intent_2": intent_2,
                    "similarity": sim,
                    "confusion_count": conf,
                    "rationale": "Semantically indistinguishable - merge into single intent"
                })
            
            elif sim >= 0.90 and conf > 3:
                # High similarity + moderate confusion → reword
                strategies["reword_candidates"].append({
                    "intent_1": intent_1,
                    "intent_2": intent_2,
                    "similarity": sim,
                    "confusion_count": conf,
                    "rationale": "Semantically close but distinct - reword for better separation"
                })
            
            elif sim >= 0.85 and conf > 0:
                # Moderate similarity + some confusion → rejection threshold
                strategies["rejection_candidates"].append({
                    "intent_1": intent_1,
                    "intent_2": intent_2,
                    "similarity": sim,
                    "confusion_count": conf,
                    "rationale": "Borderline distinction - apply rejection threshold"
                })
        
        # Analyze clusters for hierarchical grouping
        for cluster in self.clusters:
            if len(cluster.intents) >= 3 and cluster.avg_similarity >= 0.88:
                strategies["hierarchical_candidates"].append({
                    "intents": cluster.intents,
                    "avg_similarity": cluster.avg_similarity,
                    "rationale": f"Cluster of {len(cluster.intents)} related intents - "
                                f"consider coarse-to-fine classification"
                })
        
        # Print summary
        for strategy_type, candidates in strategies.items():
            print(f"\n{strategy_type.upper().replace('_', ' ')}:")
            print(f"  {len(candidates)} candidates")
            
            if candidates and len(candidates) <= 5:
                for i, candidate in enumerate(candidates, 1):
                    print(f"  {i}. {candidate}")
        
        print("\n" + "="*70)
        
        return strategies
    
    def plot_similarity_matrix(self, save_path: Optional[Path] = None):
        """
        Visualize similarity matrix as heatmap.
        
        Args:
            save_path: Optional path to save plot
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        print("\nGenerating similarity matrix heatmap...")
        
        plt.figure(figsize=(12, 10))
        
        # Use seaborn for better heatmap
        sns.heatmap(
            self.similarity_matrix,
            xticklabels=self.intent_names,
            yticklabels=self.intent_names,
            cmap='RdYlGn',
            center=0.7,
            vmin=0,
            vmax=1,
            annot=False,
            fmt='.2f',
            cbar_kws={'label': 'Cosine Similarity'}
        )
        
        plt.title('Intent Embedding Similarity Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Intent', fontsize=12)
        plt.ylabel('Intent', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved similarity matrix: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(self, save_path: Optional[Path] = None, normalize: bool = True):
        """
        Visualize confusion matrix as heatmap.
        
        Args:
            save_path: Optional path to save plot
            normalize: Normalize by true label counts
        """
        if self.confusion_mat is None:
            raise ValueError("Compute confusion matrix first")
        
        print("\nGenerating confusion matrix heatmap...")
        
        confusion_display = self.confusion_mat.copy()
        
        if normalize:
            # Normalize by row (true label)
            row_sums = confusion_display.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            confusion_display = confusion_display / row_sums
        
        plt.figure(figsize=(14, 12))
        
        sns.heatmap(
            confusion_display,
            xticklabels=self.confusion_labels,
            yticklabels=self.confusion_labels,
            cmap='Blues',
            annot=False,
            fmt='.2f' if normalize else 'd',
            cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
        )
        
        title = 'Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Intent', fontsize=12)
        plt.ylabel('True Intent', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved confusion matrix: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, output_path: Path, resolution_strategies: Dict):
        """
        Generate comprehensive overlap analysis report.
        
        Args:
            output_path: Path to save JSON report
            resolution_strategies: Resolution strategy recommendations
        """
        if self.similarity_matrix is None:
            raise ValueError("Compute similarity matrix first")
        
        # Overall statistics
        n = len(self.intent_names)
        triu_indices = np.triu_indices(n, k=1)
        similarities = self.similarity_matrix[triu_indices]
        
        report = {
            "analysis_overview": {
                "date": "2026-02-11",
                "num_intents": n,
                "mean_similarity": float(similarities.mean()),
                "std_similarity": float(similarities.std()),
                "max_similarity": float(similarities.max()),
                "num_high_overlap_pairs": len(self.high_overlap_pairs),
                "num_clusters": len(self.clusters)
            },
            "high_overlap_pairs": [pair.to_dict() for pair in self.high_overlap_pairs],
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "resolution_strategies": resolution_strategies,
            "recommendations": {
                "immediate": [],
                "short_term": [],
                "long_term": []
            }
        }
        
        # Generate recommendations
        if len(resolution_strategies["merge_candidates"]) > 0:
            report["recommendations"]["immediate"].append(
                f"Merge {len(resolution_strategies['merge_candidates'])} redundant intent pairs"
            )
        
        if len(resolution_strategies["reword_candidates"]) > 0:
            report["recommendations"]["short_term"].append(
                f"Reword {len(resolution_strategies['reword_candidates'])} intent descriptions "
                "for better embedding separation"
            )
        
        if len(resolution_strategies["hierarchical_candidates"]) > 0:
            report["recommendations"]["long_term"].append(
                f"Implement hierarchical classification for "
                f"{len(resolution_strategies['hierarchical_candidates'])} intent clusters"
            )
        
        if similarities.mean() > 0.85:
            report["recommendations"]["immediate"].append(
                "High average similarity (>0.85) indicates vocabulary needs refinement"
            )
        
        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Overlap analysis report saved: {output_path}")
        return report


def load_validation_samples(dataset_path: Path) -> List[Tuple[np.ndarray, str]]:
    """Load validation dataset."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    samples = []
    for sample in data['samples']:
        embedding = np.array(sample['gesture_embedding'], dtype=np.float32)
        intent = sample['intent']
        samples.append((embedding, intent))
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Intent Overlap Analysis and Resolution"
    )
    parser.add_argument('--dataset', type=Path, required=True,
                        help='Path to validation dataset JSON')
    parser.add_argument('--output', type=Path, required=True,
                        help='Path to save analysis report JSON')
    parser.add_argument('--similarity-threshold', type=float, default=0.85,
                        help='Threshold for high-overlap detection')
    parser.add_argument('--plot-similarity', type=Path, default=None,
                        help='Path to save similarity matrix plot')
    parser.add_argument('--plot-confusion', type=Path, default=None,
                        help='Path to save confusion matrix plot')
    parser.add_argument('--apply-resolution', action='store_true',
                        help='Apply recommended resolution strategies')
    
    args = parser.parse_args()
    
    print("="*70)
    print("INTENT OVERLAP ANALYSIS")
    print("="*70)
    
    # Load validation dataset
    print(f"\nLoading validation dataset: {args.dataset}")
    validation_samples = load_validation_samples(args.dataset)
    print(f"✓ Loaded {len(validation_samples)} samples")
    
    # Initialize extractor
    print("\nInitializing semantic intent extractor...")
    encoder = CLIPEncoder(
        model_name="ViT-B-32",
        pretrained="openai",
        device="cpu"
    )
    extractor = SemanticIntentExtractor(
        clip_model=encoder.model,
        device=encoder.device
    )
    print(f"✓ Extractor initialized")
    print(f"  Intents: {len(extractor.intent_names)}")
    
    # Create analyzer
    analyzer = IntentOverlapAnalyzer(extractor)
    
    # Compute similarity matrix
    analyzer.compute_similarity_matrix()
    
    # Identify high-overlap pairs
    analyzer.identify_high_overlap_pairs(threshold=args.similarity_threshold)
    
    # Compute confusion matrix
    analyzer.compute_confusion_matrix(validation_samples)
    
    # Correlate overlap with confusion
    analyzer.correlate_overlap_and_confusion()
    
    # Form clusters
    analyzer.form_overlap_clusters(similarity_threshold=args.similarity_threshold)
    
    # Propose resolution strategies
    resolution_strategies = analyzer.propose_resolution_strategy()
    
    # Generate plots
    if args.plot_similarity:
        analyzer.plot_similarity_matrix(save_path=args.plot_similarity)
    
    if args.plot_confusion:
        analyzer.plot_confusion_matrix(save_path=args.plot_confusion, normalize=True)
    
    # Generate report
    report = analyzer.generate_report(args.output, resolution_strategies)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"High-overlap pairs: {len(analyzer.high_overlap_pairs)}")
    print(f"Intent clusters: {len(analyzer.clusters)}")
    print(f"Mean similarity: {report['analysis_overview']['mean_similarity']:.4f}")
    print("\nRecommendations:")
    for category, recs in report["recommendations"].items():
        if recs:
            print(f"  {category.upper()}:")
            for rec in recs:
                print(f"    - {rec}")
    print("="*70)


if __name__ == "__main__":
    main()
