import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Import from refactored modules
from dataset.siamesepair import create_siamese_dataloaders
from model.siamesenetwork import create_siamese_model
from model.metrics import accuracy, precision, recall, f1_score

# Constants
BATCH_SIZE = 16
EPOCHS = 150
OUTPUT_DIR = 'output'  # Define output directory

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving outputs to: {OUTPUT_DIR}/")

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=150, output_dir='output'):
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
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
        
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                outputs = model(img1, img2)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_acc += accuracy(outputs, labels).item()
                val_f1 += f1_score(outputs, labels).item()
        
        # Normalize metrics
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_f1 /= len(val_loader)
        
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
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print("-" * 50)
    
    return history

# Main execution
if __name__ == "__main__":
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set data path
    dataset = 'LIVDET'  # Change to 'SOKOTO' if needed
    
    # Create data loaders - all dataset handling now happens in the siamesepair module
    train_loader, val_loader, test_loader = create_siamese_dataloaders(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=4
    )
    
    # Initialize the model
    model = create_siamese_model(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    history = train_model(model, train_loader, val_loader, criterion, optimizer, 
                          num_epochs=EPOCHS, output_dir=OUTPUT_DIR)
    
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
    plot_path = os.path.join(OUTPUT_DIR, 'training_history.png')
    plt.savefig(plot_path)
    print(f"Training history plot saved to: {plot_path}")
    plt.show()
    
    # Save the feature extraction model separately
    feature_model = model.get_feature_extractor()
    feature_model_path = os.path.join(OUTPUT_DIR, 'feature_model.pth')
    torch.save(feature_model.state_dict(), feature_model_path)
    print(f"Feature extraction model saved to: {feature_model_path}")
    
    # Save the full model
    full_model_path = os.path.join(OUTPUT_DIR, 'full_model.pth')
    torch.save(model.state_dict(), full_model_path)
    print(f"Full model saved to: {full_model_path}")
    
    # Evaluate on test set
    model.eval()
    test_loss, test_acc, test_f1 = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            test_acc += accuracy(outputs, labels).item()
            test_f1 += f1_score(outputs, labels).item()
    
    # Normalize metrics
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    test_f1 /= len(test_loader)
    
    print(f"Test results:")
    print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")
    
    # Save test results to file
    test_results_path = os.path.join(OUTPUT_DIR, 'test_results.txt')
    with open(test_results_path, 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
    print(f"Test results saved to: {test_results_path}")

