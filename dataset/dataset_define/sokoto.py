import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, Dict, Any
from torch.utils.data import Dataset
import torch

from .basedataset import SubjectsFingerprint, BaseDataset
from dataset.preprocess.preprocessing import create_fingerprint_transforms, get_default_args
from dataset.preprocess.enhancing import create_fingerprint_enhancement
from sklearn.model_selection import train_test_split


class SOKOTODataset(BaseDataset):
    def __init__(self, data_path: str, split: str = 'train', args: Optional[Any] = None, 
                 altered_levels: list = ['Easy', 'Medium', 'Hard'], subjects: list = None, subject_to_id: dict = {}):
        super().__init__(data_path, args, split, subjects, subject_to_id)
    
def get_SOKOTO_fingerprint_datasets(data_path: str, args=None, altered_levels=['Easy', 'Medium', 'Hard']):

    data_path = Path(data_path)
    real_path = data_path / 'Real'
    real_images = sorted(list(real_path.glob('*.BMP')))
    print(f"Found {len(real_images)} real fingerprint images in {real_path}")
    
    altered_images = []
    for level in altered_levels:
        altered_path = data_path / 'Altered' / f'Altered-{level}'
        if altered_path.exists():
            level_images = sorted(list(altered_path.glob('*.BMP')))
            altered_images.extend(level_images)
            print(f"Found {len(level_images)} altered fingerprint images in {altered_path}")
    
    # Combine real and altered images
    all_images = real_images + altered_images
    
    # Extract subject IDs from filenames using first 5 elements (e.g., "1__M_Left_index")
    subjects_to_idx = {}
    subjects: list[SubjectsFingerprint] = []
    for path in all_images:
        parts = path.stem.split('_')
        subject_id = ""
        if len(parts) >= 5:
            subject_id = '_'.join(parts[:5])
        else:
            subject_id = path.stem
            
        if subject_id in subjects_to_idx:
            idx = subjects_to_idx[subject_id]
            subjects[idx].add_filepath(path)
        else:
            # Create a new subject entry if it doesn't exist
            new_subject = SubjectsFingerprint(subject_id)
            new_subject.add_filepath(path)
            subjects_to_idx[subject_id] = len(subjects)
            subjects.append(new_subject)

    print(f"Identified {len(subjects)} unique subjects total")


    # Split subjects into train, val, and test
    train_subjects, test_subjects = train_test_split(
        subjects, test_size=0.2, random_state=42)
    train_subjects, val_subjects = train_test_split(
        train_subjects, test_size=0.2, random_state=42)
    
    train_subject_to_id, val_subjects_to_id, test_subject_to_id = {}, {}, {}
    for idx, subject in enumerate(train_subjects):
        train_subject_to_id[subject.get_id()] = idx
    for idx, subject in enumerate(val_subjects):
        val_subjects_to_id[subject.get_id()] = idx
    for idx, subject in enumerate(test_subjects):
        test_subject_to_id[subject.get_id()] = idx

    
    
    # Create datasets with the appropriate subject lists
    train_dataset = SOKOTODataset(data_path, split='train', args=args, subjects=train_subjects, altered_levels=altered_levels, subject_to_id=train_subject_to_id)
    val_dataset = SOKOTODataset(data_path, split='val', args=args, subjects=val_subjects, altered_levels=altered_levels, subject_to_id=val_subjects_to_id)
    test_dataset = SOKOTODataset(data_path, split='test', args=args, subjects=test_subjects, altered_levels=altered_levels, subject_to_id=test_subject_to_id)
    
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    # Example usage
    data_path = "path/to/sokoto/dataset"
    train_dataset, val_dataset, test_dataset = get_SOKOTO_fingerprint_datasets(data_path)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")