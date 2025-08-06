import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

# Import from refactored modules
from dataset.siamesepair import create_siamese_dataloaders
from model import get_architecture
from model.metrics import (
    accuracy, precision, recall, f1_score, 
    fmr, fnmr, gar, far, frr,
    calculate_eer, calculate_roc_metrics, plot_roc_curve,
    get_all_metrics
)

# Constants
BATCH_SIZE = 16
EPOCHS = 150
OUTPUT_DIR = 'output/minutiaenet_training'  # Define output directory

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving outputs to: {OUTPUT_DIR}/")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=150, output_dir='output'):
    """Enhanced training function with comprehensive metrics"""
    best_val_loss = float('inf')
    best_eer = float('inf')
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'val_eer': [], 'val_roc_auc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_acc, train_f1 = 0.0, 0.0, 0.0
        
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_acc += accuracy(outputs, labels).item()
            train_f1 += f1_score(outputs, labels).item()
        
        # Normalize metrics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_f1 /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
        all_val_outputs = []
        all_val_labels = []
        
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                outputs = model(img1, img2)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_acc += accuracy(outputs, labels).item()
                val_f1 += f1_score(outputs, labels).item()
                
                # Collect outputs for ROC analysis
                all_val_outputs.extend(outputs.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        # Normalize metrics
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_f1 /= len(val_loader)
        
        # Calculate ROC metrics
        all_val_outputs = np.array(all_val_outputs)
        all_val_labels = np.array(all_val_labels)
        roc_metrics = calculate_roc_metrics(all_val_outputs, all_val_labels)
        val_eer = roc_metrics['eer']
        val_roc_auc = roc_metrics['roc_auc']
        
        # Save best model based on EER (lower is better)
        if val_eer < best_eer:
            best_eer = val_eer
            model_path = os.path.join(output_dir, 'best_model_eer.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eer': best_eer,
                'roc_auc': val_roc_auc,
            }, model_path)
            print(f"Saved new best model with EER: {best_eer:.4f}, ROC AUC: {val_roc_auc:.4f}")
        
        # Also save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(output_dir, 'best_model_loss.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, model_path)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_eer'].append(val_eer)
        history['val_roc_auc'].append(val_roc_auc)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"Val EER: {val_eer:.4f}, Val ROC AUC: {val_roc_auc:.4f}")
        print("-" * 60)
    
    return history

def evaluate_model(model, test_loader, output_dir):
    """Comprehensive model evaluation with all metrics"""
    model.eval()
    all_outputs = []
    all_labels = []
    test_loss = 0.0
    
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader)
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    
    # Calculate all metrics
    metrics = get_all_metrics(all_outputs, all_labels)
    
    # Print comprehensive results
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST RESULTS")
    print("="*60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"EER: {metrics['eer']:.4f}")
    print(f"EER Threshold: {metrics['eer_threshold']:.4f}")
    print(f"FMR: {metrics['fmr']:.4f}")
    print(f"FNMR: {metrics['fnmr']:.4f}")
    print(f"GAR: {metrics['gar']:.4f}")
    print(f"FAR: {metrics['far']:.4f}")
    print(f"FRR: {metrics['frr']:.4f}")
    print(f"APCER: {metrics['apcer']:.4f}")
    print(f"BPCER: {metrics['bpcer']:.4f}")
    print("="*60)
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Test metrics saved to: {metrics_file}")
    
    # Plot ROC curve
    roc_plot_path = os.path.join(output_dir, 'roc_curve.png')
    plot_roc_curve(all_outputs, all_labels, save_path=roc_plot_path, 
                   title="MinutiaeNet ROC Curve")
    
    return metrics

def plot_training_history(history, output_dir):
    """Plot comprehensive training history"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[0, 2].plot(history['train_f1'], label='Train F1')
    axes[0, 2].plot(history['val_f1'], label='Val F1')
    axes[0, 2].set_title('F1 Score')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # EER
    axes[1, 0].plot(history['val_eer'], label='Val EER', color='red')
    axes[1, 0].set_title('Equal Error Rate (EER)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('EER')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ROC AUC
    axes[1, 1].plot(history['val_roc_auc'], label='Val ROC AUC', color='green')
    axes[1, 1].set_title('ROC AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ROC AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Combined metrics
    axes[1, 2].plot(history['val_acc'], label='Accuracy', alpha=0.7)
    axes[1, 2].plot(history['val_f1'], label='F1 Score', alpha=0.7)
    axes[1, 2].plot(history['val_roc_auc'], label='ROC AUC', alpha=0.7)
    axes[1, 2].set_title('Combined Validation Metrics')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {plot_path}")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print("Starting MinutiaeNet training with enhanced metrics...")
    print("="*60)
    
    # Set data path
    dataset = 'LIVDET'  # Change to 'SOKOTO' if needed
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_siamese_dataloaders(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        args=None,
        genuine_rate=0.125
    )
    
    # Initialize the simplified MinutiaeNet model with pretrained weights
    pretrained_path = "weights/minutiaenet_livdet.h5"
    print(f"Loading pretrained MinutiaeNet weights from: {pretrained_path}")
    model = get_architecture('minutiaenet_simple', device=device, pretrained_path=pretrained_path)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    history = train_model(model, train_loader, val_loader, criterion, optimizer, 
                          num_epochs=EPOCHS, output_dir=OUTPUT_DIR)
    
    # Plot training history
    plot_training_history(history, OUTPUT_DIR)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, OUTPUT_DIR)
    
    # Save the final model
    final_model_path = os.path.join(OUTPUT_DIR, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Save the feature extraction model separately
    feature_model = model.get_feature_extractor()
    feature_model_path = os.path.join(OUTPUT_DIR, 'feature_model.pth')
    torch.save(feature_model.state_dict(), feature_model_path)
    print(f"Feature extraction model saved to: {feature_model_path}")
    
    print(f"\nTraining completed! All results saved in {OUTPUT_DIR}")
    print("="*60) 