import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from dataset import create_siamese_dataloaders

# Create output directory if it doesn't exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

dataset = 'LIVDET'

train_loader, val_loader, test_loader = create_siamese_dataloaders(
    dataset, batch_size=1, num_workers=0)

print(f"Train loader: {len(train_loader)} batches")
print(f"Validation loader: {len(val_loader)} batches")
print(f"Test loader: {len(test_loader)} batches")

# Save images from training batch
for i, batch in enumerate(train_loader):
    if i >= 5:  # Only save first 5 batches
        break
        
    x1, x2, labels = batch
    print(f"Train batch {i}: Size: {x1.size(0)}, Shape: {x1.shape}, Labels: {labels.item()}")
    
    # Save images using torchvision's save_image
    save_image(x1, f"{output_dir}/train_{i}_img1.png")
    save_image(x2, f"{output_dir}/train_{i}_img2.png")
    
    # Alternative: Save using matplotlib
    plt.figure(figsize=(10, 5))
    
    # First image
    plt.subplot(1, 2, 1)
    img1_np = x1.squeeze().cpu().numpy()
    plt.imshow(img1_np, cmap='gray')
    plt.title(f"Image 1 - {'Same' if labels.item() > 0.5 else 'Different'} subject")
    
    # Second image
    plt.subplot(1, 2, 2)
    img2_np = x2.squeeze().cpu().numpy()
    plt.imshow(img2_np, cmap='gray')
    plt.title(f"Image 2 - {'Same' if labels.item() > 0.5 else 'Different'} subject")
    
    plt.suptitle(f"Fingerprint Pair (Label: {labels.item():.1f})")
    plt.savefig(f"{output_dir}/train_{i}_pair.png")
    plt.close()

# Save images from validation batch
for i, batch in enumerate(val_loader):
    if i >= 3:  # Only save first 3 batches
        break
        
    x1, x2, labels = batch
    print(f"Validation batch {i}: Size: {x1.size(0)}, Shape: {x1.shape}, Labels: {labels.item()}")
    
    # Save images
    save_image(x1, f"{output_dir}/val_{i}_img1.png")
    save_image(x2, f"{output_dir}/val_{i}_img2.png")

# Save images from test batch
for i, batch in enumerate(test_loader):
    if i >= 3:  # Only save first 3 batches
        break
        
    x1, x2, labels = batch
    print(f"Test batch {i}: Size: {x1.size(0)}, Shape: {x1.shape}, Labels: {labels.item()}")
    
    # Save images
    save_image(x1, f"{output_dir}/test_{i}_img1.png")
    save_image(x2, f"{output_dir}/test_{i}_img2.png")

print(f"Images saved to {output_dir}/ directory")