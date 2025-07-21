import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch


from .preprocessing import create_fingerprint_transforms
from .enhancing import create_fingerprint_enhancement

class SOKOTODataset(Dataset):
    """
    Dataset class for SOKOTO fingerprint images with preprocessing and enhancement.
    Loads both real and altered fingerprints.
    """
    def __init__(self, data_path: str, split: str = 'train', args: Optional[Any] = None, 
                 altered_levels: list = ['Easy', 'Medium', 'Hard'], subjects: list = None):
        """
        Args:
            data_path: Path to SOKOTO dataset
            split: 'train', 'val', or 'test'
            args: Configuration arguments for preprocessing and enhancement
            altered_levels: Which difficulty levels to include from the Altered folder
            subjects: List of subject IDs to include (if None, all subjects are used)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.altered_levels = altered_levels
        
        # Create preprocessing and enhancement pipelines
        self.args = args or create_default_args(mode=split)
        self.preprocessor = create_fingerprint_transforms(self.args, mode=split)
        self.enhancer = create_fingerprint_enhancement(self.args)
        
        # Load real fingerprint images
        self.real_path = self.data_path / 'Real'
        self.real_images = sorted(list(self.real_path.glob('*.BMP')))
        print(f"Found {len(self.real_images)} real fingerprint images in {self.real_path}")
        
        # Load altered fingerprint images
        self.altered_images = []
        for level in altered_levels:
            altered_path = self.data_path / 'Altered' / f'Altered-{level}'
            if altered_path.exists():
                level_images = sorted(list(altered_path.glob('*.BMP')))
                self.altered_images.extend(level_images)
                print(f"Found {len(level_images)} altered fingerprint images in {altered_path}")
        
        # Combine real and altered images
        self.all_images = self.real_images + self.altered_images
        
        # Extract subject IDs from filenames using first 5 elements (e.g., "1__M_Left_index")
        self.subject_ids = []
        for path in self.all_images:
            parts = path.stem.split('_')
            if len(parts) >= 5:
                # Join the first 5 elements with underscores
                subject_id = '_'.join(parts[:5])
                self.subject_ids.append(subject_id)
            else:
                # Fallback if filename doesn't have enough parts
                self.subject_ids.append(path.stem)
        
        # Get unique subjects (all or filtered)
        if subjects is None:
            self.unique_subjects = sorted(set(self.subject_ids))
            self.subjects = self.unique_subjects
        else:
            self.unique_subjects = sorted(set(self.subject_ids))
            self.subjects = subjects
            
        print(f"Identified {len(self.unique_subjects)} unique subjects total")
        print(f"Using {len(self.subjects)} subjects for {split} split")
        
        # Filter images for the current split based on subject list
        self.images = [img for img, subj_id in zip(self.all_images, self.subject_ids) 
                       if subj_id in self.subjects]
        
        # Create subject ID to index mapping
        self.subject_to_idx = {subj: idx for idx, subj in enumerate(self.unique_subjects)}
        
        print(f"{split} set: {len(self.images)} images from {len(self.subjects)} subjects")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Get subject ID for this image using first 5 elements
        parts = img_path.stem.split('_')
        if len(parts) >= 5:
            subject_id = '_'.join(parts[:5])
        else:
            subject_id = img_path.stem
        
        # Apply preprocessing
        preprocessed = self.preprocessor(img)
        
        # Apply enhancement (if PIL image or numpy array, otherwise use as is)
        if False:
        # if isinstance(preprocessed, Image.Image) or isinstance(preprocessed, np.ndarray):
            # Get enhanced image (extract only the enhanced result, not all stages)
            enhanced_results = self.enhancer(preprocessed)
            final_img = enhanced_results['thinned']  # Use the original enhanced image
        else:
            # If already a tensor, use as is
            final_img = preprocessed
        
        # Ensure correct shape: [1, H, W] for a single-channel image
        if isinstance(final_img, torch.Tensor) and final_img.dim() == 2:
            final_img = final_img.unsqueeze(0)
        elif isinstance(final_img, np.ndarray) and final_img.ndim == 2:
            final_img = torch.from_numpy(final_img).unsqueeze(0).float()
        
        return final_img, subject_id


class SiamesePairDataset(Dataset):
    """
    Dataset class for generating Siamese network pairs.
    """
    def __init__(self, dataset, genuine_rate=0.5):

        self.dataset = dataset
        self.genuine_rate = genuine_rate
        
        self.subject_to_indices = {}
        for idx, (_, subject_id) in enumerate(self.dataset):
            if subject_id not in self.subject_to_indices:
                self.subject_to_indices[subject_id] = []
            self.subject_to_indices[subject_id].append(idx)        
        self.subjects = list(self.subject_to_indices.keys())
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img1, subject_id = self.dataset[idx]
        is_genuine = random.random() < self.genuine_rate  

        if is_genuine:
            if len(self.subject_to_indices[subject_id]) > 1:
                ref_indices = [i for i in self.subject_to_indices[subject_id] if i != idx]
                ref_idx = random.choice(ref_indices)
            else:
                ref_idx = self.subject_to_indices[subject_id][0]
            img2, _ = self.dataset[ref_idx]
            label = 1.0
        else:
            other_subjects = [s for s in self.subjects if s != subject_id]
            other_subject = random.choice(other_subjects)
            
            ref_idx = random.choice(self.subject_to_indices[other_subject])
            img2, _ = self.dataset[ref_idx]
            label = 0.0

        # print(img1, img2, label)
        return img1, img2, torch.tensor([label], dtype=torch.float32)


def create_default_args(mode='train'):
    """
    Create arguments object with default parameters, matching test_enhancement.py.
    
    Args:
        mode: 'train', 'val', or 'test' - affects augmentation settings
    
    Returns:
        Args object with all required parameters
    """
    class Args:
        def __init__(self):
            # Common parameters
            self.img_size = 224  # Standard image size for training
            
            # Preprocessing parameters
            self.fingerprint_normalization = True
            self.histogram_equalization = False
            
            # Augmentation parameters (only applied in training mode)
            self.rotation_degrees = 30 if mode == 'train' else 0
            self.horizontal_flip_p = 0.5 if mode == 'train' else 0
            self.vertical_flip_p = 0.3 if mode == 'train' else 0
            self.blur_p = 0.7 if mode == 'train' else 0
            self.elastic_p = 0.3 if mode == 'train' else 0
            self.crop_scale = (0.9, 1.1) if mode == 'train' else (1.0, 1.0)
            self.crop_ratio = (0.9, 1.1) if mode == 'train' else (1.0, 1.0)
            
            # Enhancement parameters (from test_enhancement.py)
            self.apply_orientation = False
            self.apply_ogorman = False
            self.apply_binarization = True
            self.apply_thinning = True  # Thinning might remove too much information for verification
            self.orientation_block_size = 16
            self.orientation_smooth_sigma = 1.0
            self.ogorman_filter_size = 7
            self.ogorman_sigma_u = 2.0
            self.ogorman_sigma_v = 0.5
            self.binarization_window_size = 9
            self.thinning_iterations = 1
    
    return Args()


def get_fingerprint_datasets(data_path: str, args=None):
    """
    Creates train, validation and test datasets.
    
    Args:
        data_path: Path to SOKOTO dataset
        args: Arguments object for preprocessing and enhancement
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # First, create a dataset to get all subjects
    temp_dataset = SOKOTODataset(data_path, split='all', args=args, subjects=None)
    all_subjects = temp_dataset.unique_subjects
    
    # Split subjects into train, val, and test
    train_subjects, test_subjects = train_test_split(
        all_subjects, test_size=0.2, random_state=42)
    
    train_subjects, val_subjects = train_test_split(
        train_subjects, test_size=0.25, random_state=42)
    
    # Create datasets with the appropriate subject lists
    train_dataset = SOKOTODataset(data_path, split='train', args=args, subjects=train_subjects)
    val_dataset = SOKOTODataset(data_path, split='val', args=args, subjects=val_subjects)
    test_dataset = SOKOTODataset(data_path, split='test', args=args, subjects=test_subjects)
    
    return train_dataset, val_dataset, test_dataset

def get_siamese_datasets(data_path: str, args=None):
    """
    Creates Siamese pair datasets for training, validation and testing.
    
    Args:
        data_path: Path to SOKOTO dataset
        args: Arguments object for preprocessing and enhancement
    
    Returns:
        Tuple of (train_pair_dataset, val_pair_dataset, test_pair_dataset)
    """
    train_dataset, val_dataset, test_dataset = get_fingerprint_datasets(data_path, args)
    
    train_pair_dataset = SiamesePairDataset(train_dataset)
    val_pair_dataset = SiamesePairDataset(val_dataset)
    test_pair_dataset = SiamesePairDataset(test_dataset)
    
    return train_pair_dataset, val_pair_dataset, test_pair_dataset


def create_dataloaders(data_path: str, batch_size=32, num_workers=4, args=None):
    """
    Creates DataLoaders for training, validation and testing.
    
    Args:
        data_path: Path to SOKOTO dataset
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for DataLoaders
        args: Arguments object for preprocessing and enhancement
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_pair_dataset, val_pair_dataset, test_pair_dataset = get_siamese_datasets(
        data_path, args)
    
    train_loader = DataLoader(
        train_pair_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_pair_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_pair_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    print(f"Created DataLoaders: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    data_path = '../data/socofing'  # Adjust to your dataset path
    args = create_default_args(mode='train')
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path, batch_size=1, num_workers=4, args=args)
    
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
    # This will print the first batch of the validation and test loaders
    # and their shapes for verification
    # You can further test the dataset and dataloader functionality here
    # This is useful to ensure the dataset and dataloader are working as expected
    # and that the images are being loaded and preprocessed correctly.  