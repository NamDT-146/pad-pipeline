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


class FVCDataset(BaseDataset):
    def __init__(self, data_path: str, split: str = 'train', args: Optional[Any] = None, 
                 dbase: list = ['DB0','DB1', 'DB2', 'DB3', 'DB4'], subjects: list = None, subject_to_id: dict = {}):
        super().__init__(data_path, args, split, subjects, subject_to_id)

def get_FVC_fingerprint_datasets(data_path: str, args=None, dbase=['DB0']):

    data_path = Path(data_path)
    all_db_train_images = []
    all_db_test_images = []
    for database in dbase:
        train_path = Path(os.path.join(data_path, f'{database}_A'))
        if train_path.exists():
            train_images = sorted(list(train_path.glob('*.tif')))
            all_db_train_images.extend(train_images)
            print(f"Found {len(train_images)} training fingerprint images in {train_path}")

        test_path = Path(os.path.join(data_path, f'{database}_B'))
        if test_path.exists():
            test_images = sorted(list(test_path.glob('*.tif')))
            all_db_test_images.extend(test_images)
            print(f"Found {len(test_images)} testing fingerprint images in {test_path}")

    # Extract subject IDs from filenames using first 5 elements (e.g., "1__M_Left_index")
    subjects_to_idx = {}
    train_subjects: list[SubjectsFingerprint] = []
    test_subjects: list[SubjectsFingerprint] = []

    # If no images found at all, raise a clear error
    if len(all_db_train_images) == 0 and len(all_db_test_images) == 0:
        raise FileNotFoundError(
            f"No FVC images found under {data_path}. Expected folders like '{dbase[0]}_A' and '{dbase[0]}_B' containing .tif files."
        )

    # Build subject lists from discovered images
    for subjects, images in [(train_subjects, all_db_train_images), (test_subjects, all_db_test_images)]:
        for path in images:
            parts = path.stem.split('_')
            subject_id = ""
            if len(parts) >= 1:
                subject_id = '_'.join(parts[:1])
            else:
                subject_id = path.stem
                
            if subject_id in subjects_to_idx:
                idx = subjects_to_idx[subject_id]
                subjects[idx].add_filepath(path)
            else:
                new_subject = SubjectsFingerprint(subject_id)
                new_subject.add_filepath(path)
                subjects_to_idx[subject_id] = len(subjects)
                subjects.append(new_subject)

    print(f"Found {len(train_subjects)} training fingerprint instances across all databases")
    print(f"Found {len(test_subjects)} testing fingerprint instances across all databases")

    # Split train into train/val if we have any training subjects
    if len(train_subjects) > 0:
        train_subjects, val_subjects = train_test_split(
            train_subjects, test_size=0.2, random_state=42)
    else:
        val_subjects = []

    train_subject_to_id, val_subjects_to_id, test_subject_to_id = {}, {}, {}
    for idx, subject in enumerate(train_subjects):
        train_subject_to_id[subject.get_id()] = idx
    for idx, subject in enumerate(val_subjects):
        val_subjects_to_id[subject.get_id()] = idx
    for idx, subject in enumerate(test_subjects):
        test_subject_to_id[subject.get_id()] = idx

    train_dataset = FVCDataset(data_path, split='train', args=args, subjects=train_subjects, dbase=dbase, subject_to_id=train_subject_to_id)
    val_dataset = FVCDataset(data_path, split='val', args=args, subjects=val_subjects, dbase=dbase, subject_to_id=val_subjects_to_id)
    test_dataset = FVCDataset(data_path, split='test', args=args, subjects=test_subjects, dbase=dbase, subject_to_id=test_subject_to_id)

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    # Example usage
    data_path = "data/fvc"
    train_dataset, val_dataset, test_dataset = get_FVC_fingerprint_datasets(data_path)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")