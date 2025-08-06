import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Import the pipeline
from similarity_pipeline import FingerprintSimilarityPipeline


def create_synthetic_fingerprint_data(num_pairs=1000, embedding_size=512, noise_level=0.1):
    """
    Create synthetic fingerprint data for demonstration purposes.
    
    Args:
        num_pairs: Number of fingerprint pairs to generate
        embedding_size: Size of feature vectors
        noise_level: Amount of noise to add to features
        
    Returns:
        image_pairs: List of synthetic image pairs
        labels: Ground truth labels (1 for same person, 0 for different)
        true_features: Actual feature vectors for analysis
    """
    print(f"Creating synthetic data with {num_pairs} pairs...")
    
    # Generate synthetic feature vectors
    np.random.seed(42)  # For reproducibility
    
    # Create base features for different "persons"
    num_persons = 50
    person_features = np.random.randn(num_persons, embedding_size)
    person_features = person_features / np.linalg.norm(person_features, axis=1, keepdims=True)
    
    image_pairs = []
    labels = []
    true_features = []
    
    for i in range(num_pairs):
        if i < num_pairs // 2:
            # Genuine pair (same person)
            person_id = np.random.randint(0, num_persons)
            base_feature = person_features[person_id]
            
            # Add noise to create two slightly different features
            feature1 = base_feature + noise_level * np.random.randn(embedding_size)
            feature2 = base_feature + noise_level * np.random.randn(embedding_size)
            
            # Normalize
            feature1 = feature1 / np.linalg.norm(feature1)
            feature2 = feature2 / np.linalg.norm(feature2)
            
            labels.append(1)  # Same person
        else:
            # Impostor pair (different persons)
            person1_id = np.random.randint(0, num_persons)
            person2_id = np.random.randint(0, num_persons)
            while person2_id == person1_id:
                person2_id = np.random.randint(0, num_persons)
            
            feature1 = person_features[person1_id]
            feature2 = person_features[person2_id]
            
            labels.append(0)  # Different persons
        
        # Create synthetic "images" (just for demonstration)
        # In real usage, these would be actual fingerprint images
        img1 = np.random.rand(224, 224)  # Synthetic image 1
        img2 = np.random.rand(224, 224)  # Synthetic image 2
        
        image_pairs.append((img1, img2))
        true_features.append((feature1, feature2))
    
    return image_pairs, labels, true_features


class MockFingerprintSimilarityPipeline:
    """
    Mock pipeline for demonstration purposes that uses synthetic data.
    """
    
    def __init__(self, embedding_size=512):
        self.embedding_size = embedding_size
        print("Mock pipeline initialized for demonstration")
    
    def compare_fingerprints(self, img1, img2, similarity_method='euclidean'):
        """
        Mock comparison that returns synthetic similarity scores.
        """
        # In real usage, this would extract features from actual images
        # For demonstration, we'll generate synthetic features
        np.random.seed(hash(str(img1.shape) + str(img2.shape)) % 2**32)
        
        feature1 = torch.randn(self.embedding_size)
        feature2 = torch.randn(self.embedding_size)
        
        # Normalize features
        feature1 = feature1 / torch.norm(feature1)
        feature2 = feature2 / torch.norm(feature2)
        
        # Calculate similarity
        if similarity_method == 'euclidean':
            diff = feature1 - feature2
            distance = torch.sqrt(torch.sum(diff * diff))
            similarity = 1.0 / (1.0 + distance.item())
        elif similarity_method == 'cosine':
            dot_product = torch.sum(feature1 * feature2)
            norm1 = torch.norm(feature1)
            norm2 = torch.norm(feature2)
            similarity = (dot_product / (norm1 * norm2)).item()
            similarity = (similarity + 1) / 2
        else:
            similarity = np.random.random()
        
        return {
            'similarity_score': similarity,
            'features1': feature1,
            'features2': feature2,
            'method': similarity_method
        }
    
    def batch_similarity_analysis(self, image_pairs, labels, similarity_method='euclidean'):
        """
        Mock batch analysis using synthetic data.
        """
        print(f"Performing mock batch analysis with {len(image_pairs)} pairs...")
        
        similarities = []
        all_features1 = []
        all_features2 = []
        
        # Use synthetic features for demonstration
        for i, (img1, img2) in enumerate(image_pairs):
            if i % 200 == 0:
                print(f"Processing pair {i}/{len(image_pairs)}")
            
            # Generate synthetic similarity score
            if labels[i] == 1:  # Genuine pair
                # Higher similarity for same person
                similarity = np.random.beta(5, 2)  # Skewed towards higher values
            else:  # Impostor pair
                # Lower similarity for different persons
                similarity = np.random.beta(2, 5)  # Skewed towards lower values
            
            similarities.append(similarity)
            
            # Generate synthetic features
            feature1 = torch.randn(self.embedding_size)
            feature2 = torch.randn(self.embedding_size)
            feature1 = feature1 / torch.norm(feature1)
            feature2 = feature2 / torch.norm(feature2)
            
            all_features1.append(feature1)
            all_features2.append(feature2)
        
        # Convert to tensors
        similarities = torch.tensor(similarities)
        labels = torch.tensor(labels)
        
        # Calculate metrics
        from model.metrics import get_all_metrics
        metrics = get_all_metrics(similarities, labels)
        
        return {
            'similarities': similarities,
            'labels': labels,
            'metrics': metrics,
            'features1': torch.stack(all_features1),
            'features2': torch.stack(all_features2)
        }
    
    def plot_similarity_distributions(self, similarities, labels, save_path=None):
        """
        Plot similarity score distributions.
        """
        genuine_scores = similarities[labels == 1]
        impostor_scores = similarities[labels == 0]
        
        plt.figure(figsize=(12, 8))
        
        # Plot histograms
        plt.subplot(2, 2, 1)
        plt.hist(genuine_scores.numpy(), bins=50, alpha=0.7, label='Genuine', color='green')
        plt.hist(impostor_scores.numpy(), bins=50, alpha=0.7, label='Impostor', color='red')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Similarity Score Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot ROC curve
        plt.subplot(2, 2, 2)
        from model.metrics import calculate_roc_metrics
        roc_metrics = calculate_roc_metrics(similarities, labels)
        plt.plot(roc_metrics['fpr'], roc_metrics['tpr'], 
                color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_metrics["roc_auc"]:.3f})')
        plt.plot(roc_metrics['eer'], 1-roc_metrics['eer'], 'ro', 
                markersize=10, label=f'EER = {roc_metrics["eer"]:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot DET curve
        plt.subplot(2, 2, 3)
        from model.metrics import calculate_fmr_fnmr_at_threshold
        fmr_values = []
        fnmr_values = []
        thresholds = np.arange(0.1, 1.0, 0.01)
        
        for threshold in thresholds:
            fmr_val, fnmr_val = calculate_fmr_fnmr_at_threshold(similarities, labels, threshold)
            fmr_values.append(fmr_val)
            fnmr_values.append(fnmr_val)
        
        plt.semilogx(fmr_values, fnmr_values, color='blue', lw=2)
        plt.xlabel('False Match Rate (FMR)')
        plt.ylabel('False Non-Match Rate (FNMR)')
        plt.title('DET Curve')
        plt.grid(True, alpha=0.3)
        
        # Plot threshold analysis
        plt.subplot(2, 2, 4)
        threshold_results = self.evaluate_threshold_performance(similarities, labels)
        thresholds = list(threshold_results.keys())
        fmr_values = [threshold_results[t]['fmr'] for t in thresholds]
        fnmr_values = [threshold_results[t]['fnmr'] for t in thresholds]
        accuracy_values = [threshold_results[t]['accuracy'] for t in thresholds]
        
        plt.plot(thresholds, fmr_values, label='FMR', color='red')
        plt.plot(thresholds, fnmr_values, label='FNMR', color='blue')
        plt.plot(thresholds, accuracy_values, label='Accuracy', color='green')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title('Performance vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Similarity analysis plots saved to {save_path}")
        
        plt.show()
    
    def evaluate_threshold_performance(self, similarities, labels, thresholds=None):
        """
        Evaluate performance at different thresholds.
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)
        
        results = {}
        
        for threshold in thresholds:
            predictions = (similarities >= threshold).float()
            from model.metrics import calculate_fmr_fnmr_at_threshold
            fmr_val, fnmr_val = calculate_fmr_fnmr_at_threshold(similarities, labels, threshold)
            
            results[threshold] = {
                'fmr': fmr_val,
                'fnmr': fnmr_val,
                'accuracy': ((predictions == labels).float().mean()).item()
            }
        
        return results
    
    def generate_similarity_report(self, similarities, labels, save_path=None):
        """
        Generate a comprehensive similarity analysis report.
        """
        from model.metrics import get_all_metrics, calculate_roc_metrics
        
        metrics = get_all_metrics(similarities, labels)
        roc_metrics = calculate_roc_metrics(similarities, labels)
        
        report = f"""
FINGERPRINT SIMILARITY ANALYSIS REPORT (SYNTHETIC DATA)
======================================================

Dataset Statistics:
- Total pairs: {len(similarities)}
- Genuine pairs: {(labels == 1).sum().item()}
- Impostor pairs: {(labels == 0).sum().item()}

Similarity Score Statistics:
- Mean similarity: {similarities.mean().item():.4f}
- Std similarity: {similarities.std().item():.4f}
- Min similarity: {similarities.min().item():.4f}
- Max similarity: {similarities.max().item():.4f}

Performance Metrics:
- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1 Score: {metrics['f1']:.4f}

Verification Metrics:
- False Acceptance Rate (FAR): {metrics['far']:.4f}
- False Rejection Rate (FRR): {metrics['frr']:.4f}
- False Match Rate (FMR): {metrics['fmr']:.4f}
- False Non-Match Rate (FNMR): {metrics['fnmr']:.4f}
- Genuine Acceptance Rate (GAR): {metrics['gar']:.4f}

ROC Analysis:
- ROC AUC: {roc_metrics['roc_auc']:.4f}
- Equal Error Rate (EER): {roc_metrics['eer']:.4f}
- EER Threshold: {roc_metrics['eer_threshold']:.4f}

NOTE: This is synthetic data for demonstration purposes.
Real fingerprint data would show different performance characteristics.
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Similarity report saved to {save_path}")
        
        return report


def main():
    """
    Demonstrate the complete similarity pipeline with synthetic data.
    """
    print("=== FINGERPRINT SIMILARITY PIPELINE DEMONSTRATION ===\n")
    
    # Create synthetic data
    print("1. Creating synthetic fingerprint data...")
    image_pairs, labels, true_features = create_synthetic_fingerprint_data(
        num_pairs=1000, embedding_size=512, noise_level=0.1
    )
    
    # Initialize mock pipeline
    print("\n2. Initializing similarity pipeline...")
    pipeline = MockFingerprintSimilarityPipeline(embedding_size=512)
    
    # Example 1: Direct comparison
    print("\n3. Example: Direct fingerprint comparison...")
    img1, img2 = image_pairs[0]
    result = pipeline.compare_fingerprints(img1, img2, 'euclidean')
    print(f"   Similarity score: {result['similarity_score']:.4f}")
    print(f"   Method used: {result['method']}")
    
    # Example 2: Batch analysis
    print("\n4. Example: Batch similarity analysis...")
    batch_results = pipeline.batch_similarity_analysis(image_pairs, labels, 'euclidean')
    
    print(f"   ROC AUC: {batch_results['metrics']['roc_auc']:.4f}")
    print(f"   EER: {batch_results['metrics']['eer']:.4f}")
    print(f"   Accuracy: {batch_results['metrics']['accuracy']:.4f}")
    print(f"   FMR: {batch_results['metrics']['fmr']:.4f}")
    print(f"   FNMR: {batch_results['metrics']['fnmr']:.4f}")
    
    # Example 3: Generate comprehensive report
    print("\n5. Example: Generating similarity report...")
    report = pipeline.generate_similarity_report(
        batch_results['similarities'], 
        batch_results['labels']
    )
    print(report)
    
    # Example 4: Plot analysis
    print("\n6. Example: Plotting similarity analysis...")
    pipeline.plot_similarity_distributions(
        batch_results['similarities'], 
        batch_results['labels'],
        save_path='output/similarity_analysis.png'
    )
    
    # Example 5: Threshold analysis
    print("\n7. Example: Threshold performance analysis...")
    threshold_results = pipeline.evaluate_threshold_performance(
        batch_results['similarities'], 
        batch_results['labels']
    )
    
    print("   Threshold Performance:")
    for threshold, metrics in list(threshold_results.items())[:5]:  # Show first 5
        print(f"   - Threshold {threshold:.2f}: FMR={metrics['fmr']:.4f}, FNMR={metrics['fnmr']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    print("\n=== DEMONSTRATION COMPLETE ===")
    print("\nKey takeaways:")
    print("1. The pipeline processes fingerprint images through preprocessing, enhancement, and feature extraction")
    print("2. Multiple similarity methods are available (Euclidean, Cosine, Manhattan, Siamese)")
    print("3. Comprehensive evaluation metrics are calculated (ROC AUC, EER, FMR, FNMR)")
    print("4. Threshold optimization helps balance security and usability")
    print("5. Visualization tools help understand system performance")
    print("\nTo use with real data, replace the mock pipeline with the actual FingerprintSimilarityPipeline class.")


if __name__ == "__main__":
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Run demonstration
    main() 