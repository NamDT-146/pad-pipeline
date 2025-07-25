import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

from .preprocess import create_fingerprint_transforms, get_default_args, create_fingerprint_enhancement
from .dataset_define import SubjectsFingerprint, get_SOKOTO_fingerprint_datasets, get_LivDet_fingerprint_datasets

DATASET = {
    'SOKOTO': {
        'get_datasets_func': get_SOKOTO_fingerprint_datasets,
        'altered_levels': ['Easy', 'Medium', 'Hard'],
        'data_path': 'data/socofing',
        },
    'LIVDET': {
        'get_datasets_func': get_LivDet_fingerprint_datasets,
        'data_path': 'data/livedet/Live',
    }
}

class SiamesePairDataset(Dataset):
    """
    Dataset class for generating Siamese network pairs.
    """
    def __init__(self, dataset, genuine_rate=0.5):

        self.dataset = dataset
        self.genuine_rate = genuine_rate
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        subject : SubjectsFingerprint = self.dataset.subjects[idx]
        is_genuine = random.random() < self.genuine_rate  

        if is_genuine:
            img_path_1, img_path_2 = subject.get_filepath_pair()
            label = 1.0
        else:
            while True:
                imposter_idx = random.randint(0, len(self.dataset) - 1)
                if imposter_idx != idx:
                    break
            imposter_subject : SubjectsFingerprint= self.dataset.subjects[imposter_idx]
            label = 0.0
            img_path_1 = subject.get_filepath()
            img_path_2 = imposter_subject.get_filepath()

        return self.dataset[img_path_1], self.dataset[img_path_2], torch.tensor([label], dtype=torch.float32)

def get_siamese_datasets(dataset: str, genuine_rate=0.5, args=None):

    get_dataset_func = DATASET[dataset]['get_datasets_func']
    data_path = DATASET[dataset]['data_path']
    train_dataset, val_dataset, test_dataset = get_dataset_func(data_path, args=args)

    train_pair_dataset = SiamesePairDataset(train_dataset, genuine_rate=genuine_rate)
    val_pair_dataset = SiamesePairDataset(val_dataset)
    test_pair_dataset = SiamesePairDataset(test_dataset)
    
    return train_pair_dataset, val_pair_dataset, test_pair_dataset


def create_siamese_dataloaders(dataset: str, batch_size=32, num_workers=4, args=None):

    train_pair_dataset, val_pair_dataset, test_pair_dataset = get_siamese_datasets(dataset=dataset)
    
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
    args = get_default_args(mode='train')
    
    train_loader, val_loader, test_loader = create_siamese_dataloaders(
        data_path, batch_size=1, num_workers=4, args=args)
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Validation loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    for batch in train_loader:
        x1, x2, labels = batch
        print(f"Batch size: {x1.size(0)}, Image shape: {x1.shape}, Labels shape: {labels.shape}")
        break
    for batch in val_loader:
        x1, x2, labels = batch
        print(f"Validation Batch size: {x1.size(0)}, Image shape: {x1.shape}, Labels shape: {labels.shape}")
        break
    for batch in test_loader:
        x1, x2, labels = batch
        print(f"Test Batch size: {x1.size(0)}, Image shape: {x1.shape}, Labels shape: {labels.shape}")
        break