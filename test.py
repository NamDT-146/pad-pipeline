from dataset import create_dataloaders

data_path = 'data/socofing'  # Adjust to your dataset path

train_loader, val_loader, test_loader = create_dataloaders(
    data_path, batch_size=1, num_workers=4)

print(f"Train loader: {len(train_loader)} batches")
print(f"Validation loader: {len(val_loader)} batches")
print(f"Test loader: {len(test_loader)} batches")
for batch in train_loader:
    x1, x2, labels = batch
    print(f"Batch size: {x1.size(0)}, Image shape: {x1.shape}, Labels shape: {labels.shape}")
    break
# This will print the first batch of the training loader
# and its shapes for verification
# You can further test the dataset and dataloader functionality here
for batch in val_loader:
    x1, x2, labels = batch
    print(f"Validation Batch size: {x1.size(0)}, Image shape: {x1.shape}, Labels shape: {labels.shape}")
    break
for batch in test_loader:
    x1, x2, labels = batch
    print(f"Test Batch size: {x1.size(0)}, Image shape: {x1.shape}, Labels shape: {labels.shape}")
    break