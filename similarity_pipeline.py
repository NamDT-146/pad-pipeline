import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path

# Import the core functionality
from model import get_architecture
from model.metrics import (
    calculate_eer, calculate_roc_metrics, get_all_metrics,
    calculate_fmr_fnmr_at_threshold, plot_roc_curve
)
from dataset.preprocess import get_default_args
from dataset.preprocess.preprocessing import create_fingerprint_transforms
from dataset.preprocess.enhancing import create_fingerprint_enhancement


class FingerprintSimilarityPipeline:
    """
    Complete pipeline for fingerprint similarity measurement after feature extraction.
    This class demonstrates all the steps involved in measuring similarity between fingerprints.
    """
    
    def __init__(self, model_path, model_type='mobilenetv2', device='cuda'):
        """
        Initialize the similarity pipeline.
        
        Args:
            model_path: Path to the trained model weights
            model_type: Type of model ('mobilenetv2', 'siamese', 'minutiaenet', etc.)
            device: Device to run the model on
        """
        self.device = device
        self.model_type = model_type
        
        # Load the model
        print(f"Loading {model_type} model from {model_path}")
        self.model = get_architecture(model_type, device=device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        try:
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        except RuntimeError as e:
            print(f"Warning: Could not load checkpoint. Using default weights.\n{e}")
        
        self.model.eval()
        
        # Initialize preprocessing
        self.args = get_default_args(mode='test')
        self.preprocessor = create_fingerprint_transforms(self.args)
        self.enhancer = create_fingerprint_enhancement(self.args)
        
        print(f"Pipeline initialized with {model_type} model")
    
    def preprocess_image(self, image):
        """
        Step 1: Preprocess the fingerprint image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed image
        """
        if isinstance(image, str):
            image = Image.open(image).convert('L')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('L')
        
        # Apply preprocessing transforms
        preprocessed = self.preprocessor(image)
        return preprocessed
    
    def enhance_image(self, image):
        """
        Step 2: Enhance the fingerprint image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Enhanced image
        """
        enhanced_results = self.enhancer(image)
        return enhanced_results['enhanced']
    
    def extract_features(self, image):
        """
        Step 3: Extract features from the fingerprint image.
        
        Args:
            image: Enhanced image
            
        Returns:
            Feature vector (embedding)
        """
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
            elif len(image.shape) == 3 and image.shape[0] == 1:
                image = torch.from_numpy(image).float().unsqueeze(0)
            else:
                image = torch.from_numpy(image).float()
        
        # Move to device and extract features
        with torch.no_grad():
            image = image.to(self.device)
            features = self.model.extract_features(image)
        
        return features
    
    def calculate_similarity_scores(self, features1, features2, method='euclidean'):
        """
        Step 4: Calculate similarity between two feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            method: Similarity method ('euclidean', 'cosine', 'manhattan', 'siamese')
            
        Returns:
            Similarity score(s)
        """
        if method == 'siamese':
            # Use the model's built-in similarity network
            with torch.no_grad():
                score = self.model(features1.unsqueeze(0), features2.unsqueeze(0))
                return score.item()
        
        elif method == 'euclidean':
            # Euclidean distance-based similarity
            diff = features1 - features2
            distance = torch.sqrt(torch.sum(diff * diff))
            # Convert distance to similarity (0-1 scale)
            similarity = 1.0 / (1.0 + distance.item())
            return similarity
        
        elif method == 'cosine':
            # Cosine similarity
            dot_product = torch.sum(features1 * features2)
            norm1 = torch.norm(features1)
            norm2 = torch.norm(features2)
            similarity = (dot_product / (norm1 * norm2)).item()
            # Convert from [-1, 1] to [0, 1]
            similarity = (similarity + 1) / 2
            return similarity
        
        elif method == 'manhattan':
            # Manhattan distance-based similarity
            diff = torch.abs(features1 - features2)
            distance = torch.sum(diff)
            similarity = 1.0 / (1.0 + distance.item())
            return similarity
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def compare_fingerprints(self, img1, img2, similarity_method='euclidean'):
        """
        Complete pipeline: Compare two fingerprint images.
        
        Args:
            img1: First fingerprint image (PIL Image, numpy array, or file path)
            img2: Second fingerprint image (PIL Image, numpy array, or file path)
            similarity_method: Method to calculate similarity
            
        Returns:
            Dictionary with comparison results
        """
        print(f"Comparing fingerprints using {similarity_method} similarity...")
        
        # Step 1: Preprocess both images
        pre1 = self.preprocess_image(img1)
        pre2 = self.preprocess_image(img2)
        
        # Step 2: Enhance both images
        enh1 = self.enhance_image(pre1)
        enh2 = self.enhance_image(pre2)
        
        # Step 3: Extract features
        feat1 = self.extract_features(enh1)
        feat2 = self.extract_features(enh2)
        
        # Step 4: Calculate similarity
        similarity_score = self.calculate_similarity_scores(feat1, feat2, similarity_method)
        
        return {
            'similarity_score': similarity_score,
            'features1': feat1,
            'features2': feat2,
            'preprocessed1': pre1,
            'preprocessed2': pre2,
            'enhanced1': enh1,
            'enhanced2': enh2,
            'method': similarity_method
        }
    
    def batch_similarity_analysis(self, image_pairs, labels, similarity_method='euclidean'):
        """
        Analyze similarity for a batch of image pairs.
        
        Args:
            image_pairs: List of (img1, img2) tuples
            labels: List of ground truth labels (1 for same person, 0 for different)
            similarity_method: Method to calculate similarity
            
        Returns:
            Dictionary with analysis results
        """
        print(f"Performing batch similarity analysis with {len(image_pairs)} pairs...")
        
        similarities = []
        all_features1 = []
        all_features2 = []
        
        for i, (img1, img2) in enumerate(image_pairs):
            if i % 100 == 0:
                print(f"Processing pair {i}/{len(image_pairs)}")
            
            # Get comparison results
            result = self.compare_fingerprints(img1, img2, similarity_method)
            similarities.append(result['similarity_score'])
            all_features1.append(result['features1'])
            all_features2.append(result['features2'])
        
        # Convert to tensors
        similarities = torch.tensor(similarities)
        labels = torch.tensor(labels)
        
        # Calculate comprehensive metrics
        metrics = get_all_metrics(similarities, labels)
        
        return {
            'similarities': similarities,
            'labels': labels,
            'metrics': metrics,
            'features1': torch.stack(all_features1),
            'features2': torch.stack(all_features2)
        }
    
    def evaluate_threshold_performance(self, similarities, labels, thresholds=None):
        """
        Evaluate performance at different thresholds.
        
        Args:
            similarities: Similarity scores
            labels: Ground truth labels
            thresholds: List of thresholds to evaluate (default: 0.1 to 0.9)
            
        Returns:
            Dictionary with threshold analysis
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)
        
        results = {}
        
        for threshold in thresholds:
            # Apply threshold
            predictions = (similarities >= threshold).float()
            
            # Calculate metrics at this threshold
            fmr_val, fnmr_val = calculate_fmr_fnmr_at_threshold(similarities, labels, threshold)
            
            results[threshold] = {
                'fmr': fmr_val,
                'fnmr': fnmr_val,
                'accuracy': ((predictions == labels).float().mean()).item()
            }
        
        return results
    
    def plot_similarity_distributions(self, similarities, labels, save_path=None):
        """
        Plot similarity score distributions for genuine and impostor pairs.
        
        Args:
            similarities: Similarity scores
            labels: Ground truth labels
            save_path: Path to save the plot
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
    
    def generate_similarity_report(self, similarities, labels, save_path=None):
        """
        Generate a comprehensive similarity analysis report.
        
        Args:
            similarities: Similarity scores
            labels: Ground truth labels
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        # Calculate all metrics
        metrics = get_all_metrics(similarities, labels)
        roc_metrics = calculate_roc_metrics(similarities, labels)
        
        # Generate report
        report = f"""
FINGERPRINT SIMILARITY ANALYSIS REPORT
=====================================

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

Threshold Analysis:
"""
        
        # Add threshold analysis
        threshold_results = self.evaluate_threshold_performance(similarities, labels)
        for threshold, results in threshold_results.items():
            report += f"- Threshold {threshold:.2f}: FMR={results['fmr']:.4f}, FNMR={results['fnmr']:.4f}, Acc={results['accuracy']:.4f}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Similarity report saved to {save_path}")
        
        return report


def main():
    """
    Example usage of the complete similarity pipeline.
    """
    # Initialize the pipeline
    model_path = "weights/best_model.pth"  # Update with your model path
    pipeline = FingerprintSimilarityPipeline(model_path, model_type='mobilenetv2')
    
    # Example 1: Compare two fingerprint images
    print("\n=== Example 1: Direct Fingerprint Comparison ===")
    
    # You would replace these with actual image paths
    img1_path = "path/to/fingerprint1.png"
    img2_path = "path/to/fingerprint2.png"
    
    # Uncomment when you have actual images
    # result = pipeline.compare_fingerprints(img1_path, img2_path, 'euclidean')
    # print(f"Similarity score: {result['similarity_score']:.4f}")
    
    # Example 2: Batch analysis (for evaluation)
    print("\n=== Example 2: Batch Similarity Analysis ===")
    
    # This would be used with your test dataset
    # image_pairs = [...]  # List of (img1, img2) tuples
    # labels = [...]       # List of ground truth labels
    
    # Uncomment when you have actual data
    # batch_results = pipeline.batch_similarity_analysis(image_pairs, labels)
    # print(f"ROC AUC: {batch_results['metrics']['roc_auc']:.4f}")
    # print(f"EER: {batch_results['metrics']['eer']:.4f}")
    
    # Example 3: Generate comprehensive report
    print("\n=== Example 3: Generate Similarity Report ===")
    
    # Uncomment when you have actual data
    # pipeline.plot_similarity_distributions(batch_results['similarities'], batch_results['labels'])
    # report = pipeline.generate_similarity_report(batch_results['similarities'], batch_results['labels'])
    # print(report)
    
    print("\nPipeline demonstration complete!")
    print("To use with real data, uncomment the relevant sections and provide actual image paths.")


if __name__ == "__main__":
    main() 