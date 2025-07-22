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


class LivDetDataset(BaseDataset):
    def __init__(self, data_path: str, split: str = 'train', args: Optional[Any] = None, 
                 altered_levels: list = ['Easy', 'Medium', 'Hard'], subjects: list = None, subject_to_id: dict = {}):
        super().__init__(data_path, args, split, subjects, subject_to_id)

def get_LivDet_fingerprint_datasets(data_path: str, args=None, altered_levels=['Easy', 'Medium', 'Hard']):

    data_path = Path(data_path)
    live_images = sorted(list(data_path.glob('*.bmp')))
    print(f"Found {len(live_images)} live fingerprint images in {data_path}")

    

    num_identify_elements = 2
    # Extract subject IDs from filenames using first n elements (e.g., "1__M_Left_index")
    subjects_to_idx = {}
    subjects: list[SubjectsFingerprint] = []
    for path in live_images:
        parts = path.stem.split('_')
        subject_id = ""
        if len(parts) >= num_identify_elements:
            subject_id = '_'.join(parts[:num_identify_elements])
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
    train_dataset = LivDetDataset(data_path, split='train', args=args, subjects=train_subjects, altered_levels=altered_levels, subject_to_id=train_subject_to_id)
    val_dataset = LivDetDataset(data_path, split='val', args=args, subjects=val_subjects, altered_levels=altered_levels, subject_to_id=val_subjects_to_id)
    test_dataset = LivDetDataset(data_path, split='test', args=args, subjects=test_subjects, altered_levels=altered_levels, subject_to_id=test_subject_to_id)
    
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    data_path = 'data/livedet/Live'  # Adjust to your dataset path
    train_dataset, val_dataset, test_dataset = get_livdet_fingerprint_datasets(data_path)
    
    print(f"Train dataset: {len(train_dataset)} subjects")
    print(f"Validation dataset: {len(val_dataset)} subjects")
    print(f"Test dataset: {len(test_dataset)} subjects")
    
    # Example of accessing a subject's filepaths
    for subject in train_dataset.subjects[:5]:  # Print first 5 subjects
        print(subject)
        print("Filepaths:", subject.filepaths)