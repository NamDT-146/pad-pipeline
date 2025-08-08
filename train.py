import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import from refactored modules
from dataset.siamesepair import create_siamese_dataloaders
from model.siamesenetwork import create_siamese_model
from model import get_architecture
from model.metrics import accuracy, precision, recall, f1_score, get_all_metrics, plot_roc_curve

def parse_args():
    parser = argparse.ArgumentParser(description='Train a fingerprint verification model')
    
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, default='FVC', 
                        help='Dataset to use (FVC, SOKOTO, etc.)')
    parser.add_argument('--architecture', type=str, default='siamese', 
                        help='Model architecture (siamese, mobilenetv2)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=150, 
                        help='Number of epochs to train')
    parser.add_argument('--num_epoch_per_val', type=int, default=1, 
                        help='Number of times to loop through validation dataset per validation')
    
    # Environment parameters
    parser.add_argument('--output_dir', type=str, default='output', 
                        help='Base output directory')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use (cuda, cpu, or None for auto)')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to a saved checkpoint to resume training from')
    
    return parser.parse_args()

def get_output_dir(args):
    """Create a unique output directory based on model, dataset and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.resume:
        # If resuming, include "resume" in the directory name
        resume_base = os.path.basename(os.path.normpath(args.resume))
        return os.path.join(args.output_dir, f"{args.architecture}_{args.dataset}_{resume_base}_resume_{timestamp}")
    else:
        return os.path.join(args.output_dir, f"{args.architecture}_{args.dataset}_{timestamp}")

def train_model(model, train_loader, val_loader, criterion, optimizer, args, output_dir, start_epoch=0):
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    # If resuming, try to load best validation loss
    if args.resume:
        try:
            checkpoint_path = os.path.join(args.resume, 'best_model.pth')
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            best_val_loss = checkpoint.get('loss', float('inf'))
            print(f"Resuming with best validation loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Could not load best validation loss: {e}")
    
    # Initialize log file
    log_file = os.path.join(output_dir, 'log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset}, Architecture: {args.architecture}\n")
        f.write(f"Batch size: {args.batch_size}, Epochs: {args.epochs}\n")
        f.write(f"Device: {args.device}, Validation loops: {args.num_epoch_per_val}\n")
        if args.resume:
            f.write(f"Resuming from: {args.resume}, starting at epoch {start_epoch}\n")
        f.write("-" * 50 + "\n\n")
    
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()
        train_loss, train_acc, train_f1 = 0.0, 0.0, 0.0
        
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(args.device), img2.to(args.device), labels.to(args.device)
            
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
        
        with torch.no_grad():
            # Loop through validation set multiple times if requested
            for _ in range(args.num_epoch_per_val):
                for img1, img2, labels in val_loader:
                    img1, img2, labels = img1.to(args.device), img2.to(args.device), labels.to(args.device)
                    outputs = model(img1, img2)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    val_acc += accuracy(outputs, labels).item()
                    val_f1 += f1_score(outputs, labels).item()
        
        # Normalize metrics (accounting for multiple validation loops)
        val_loss /= (len(val_loader) * args.num_epoch_per_val)
        val_acc /= (len(val_loader) * args.num_epoch_per_val)
        val_f1 /= (len(val_loader) * args.num_epoch_per_val)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, model_path)
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
            print(f"Model saved to: {model_path}")
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'best_loss': best_val_loss,
            'history': history,
        }, checkpoint_path)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        
        # Log results
        log_msg = f"Epoch {epoch+1}/{args.epochs}\n"
        log_msg += f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}\n"
        log_msg += f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}\n"
        log_msg += "-" * 50 + "\n"
        
        print(log_msg)
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(log_msg)
    
    writer.close()
    return history

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    print(f"Using device: {args.device}")
    
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create output directory with model, dataset and timestamp
    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_siamese_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=0,
        args=None,
        genuine_rate=0.125
    )
    
    # Initialize the model
    model = get_architecture(args.architecture, device=args.device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_epoch = 0
    
    # Resume training if requested
    if args.resume:
        if os.path.isdir(args.resume):
            checkpoint_path = os.path.join(args.resume, 'best_model.pth')
        else:
            checkpoint_path = args.resume
            
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
    
    # Train the model
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        args,
        output_dir=output_dir,
        start_epoch=start_epoch
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.tight_layout()
    
    # Save plot to output directory
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path)
    print(f"Training history plot saved to: {plot_path}")
    plt.close()  # Close instead of show for headless environments
    
    # Save the feature extraction model separately
    feature_model = model.get_feature_extractor()
    feature_model_path = os.path.join(output_dir, 'feature_model.pth')
    torch.save(feature_model.state_dict(), feature_model_path)
    print(f"Feature extraction model saved to: {feature_model_path}")
    
    # Save the full model
    full_model_path = os.path.join(output_dir, 'full_model.pth')
    torch.save(model.state_dict(), full_model_path)
    print(f"Full model saved to: {full_model_path}")
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(args.device), img2.to(args.device), labels.to(args.device)
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            
            # Accumulate loss
            test_loss += loss.item()
            
            # Collect predictions and labels for metric calculation
            all_preds.append(outputs)
            all_labels.append(labels)

    # Normalize loss
    test_loss /= len(test_loader)

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Calculate all metrics using get_all_metrics
    metrics = get_all_metrics(all_preds, all_labels)

    # Log test results
    test_results = f"Test results:\n"
    test_results += f"Loss: {test_loss:.4f}\n"
    test_results += f"Accuracy: {metrics['accuracy']:.4f}\n"
    test_results += f"F1 Score: {metrics['f1']:.4f}\n"
    test_results += f"FAR: {metrics['far']:.4f}\n"
    test_results += f"FRR: {metrics['frr']:.4f}\n"
    test_results += f"EER: {metrics['eer']:.4f} (threshold: {metrics['eer_threshold']:.4f})\n"
    test_results += f"ROC AUC: {metrics['roc_auc']:.4f}\n"
    print(test_results)

    # Save test results to file
    test_results_path = os.path.join(output_dir, 'test_results.txt')
    with open(test_results_path, 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"FAR: {metrics['far']:.4f}\n")
        f.write(f"FRR: {metrics['frr']:.4f}\n")
        f.write(f"EER: {metrics['eer']:.4f} (threshold: {metrics['eer_threshold']:.4f})\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
    print(f"Test results saved to: {test_results_path}")

    # Generate and save ROC curve
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plot_roc_curve(all_preds, all_labels, save_path=roc_path, 
                title=f"ROC Curve for {args.architecture} on {args.dataset}")

    # Log final results to TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
    writer.add_scalar('Test/loss', test_loss)
    writer.add_scalar('Test/accuracy', metrics['accuracy'])
    writer.add_scalar('Test/f1', metrics['f1'])
    writer.add_scalar('Test/far', metrics['far'])
    writer.add_scalar('Test/frr', metrics['frr'])
    writer.add_scalar('Test/eer', metrics['eer'])
    writer.add_scalar('Test/roc_auc', metrics['roc_auc'])
    writer.close()

    # Log final results to file
    log_file = os.path.join(output_dir, 'log.txt')
    with open(log_file, 'a') as f:
        f.write(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(test_results)