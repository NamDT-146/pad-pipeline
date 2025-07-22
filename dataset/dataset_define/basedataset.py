import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, Dict, Any
from torch.utils.data import Dataset
import torch
from torchvision.transforms import functional as F


from dataset.preprocess.preprocessing import create_fingerprint_transforms, get_default_args
from dataset.preprocess.enhancing import create_fingerprint_enhancement


class SubjectsFingerprint:
    def __init__(self, subject_identity):
        self.subject_identity = subject_identity
        self.filepaths = []  # List of all filepath variations
    
    def add_filepath(self, filepath):
        if filepath not in self.filepaths:
            self.filepaths.append(filepath)
            return True
        return False
    
    def get_id(self):
        return self.subject_identity
    
    def get_filepath(self):
        return random.choice(self.filepaths) if self.filepaths else None
    
    def get_filepath_pair(self):
        filepath1 = random.choice(self.filepaths)
        filepath2 = random.choice(self.filepaths)
        if len(self.filepaths) >= 2:
            while filepath1 == filepath2:
                filepath2 = random.choice(self.filepaths)
        return filepath1, filepath2

    def __len__(self):
        """Return the number of filepaths."""
        return len(self.filepaths)
    
    def __str__(self):
        """String representation of the object."""
        return f"ObjectsFingerprint(id={self.subject_identity}, images={len(self.filepaths)})"


class BaseDataset(Dataset):
    def __init__(self, data_path: str, args: Optional[Any] = None, split: str = 'train', subjects: list[SubjectsFingerprint] = None, subject_to_id : dict = {}):
        self.data_path = Path(data_path)
        self.subjects = subjects 
        self.args = args or get_default_args(mode=split)
        self.preprocessor = create_fingerprint_transforms(self.args, mode=split)
        self.enhancer = create_fingerprint_enhancement(self.args)
        self.subject_to_id = subject_to_id

    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, img_path):
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        preprocessed = self.preprocessor(img)
        if isinstance(preprocessed, Image.Image) or isinstance(preprocessed, np.ndarray):
            enhanced_results = self.enhancer(preprocessed)
            final_img = enhanced_results['enhanced']  # Use the enhanced image
        else:
            final_img = preprocessed
        
        if isinstance(final_img, torch.Tensor) and final_img.dim() == 2:
            final_img = final_img.unsqueeze(0)
        elif isinstance(final_img, np.ndarray) and final_img.ndim == 2:
            final_img = torch.from_numpy(final_img).unsqueeze(0).float()
        
        # target_size = (224, 224)  # Adjust this to your needed size
        # if final_img.shape[-2:] != target_size:
        #     final_img = F.resize(final_img, target_size)

        return final_img
    
def get_image_files(directory):
    """Get all image files with supported extensions from a directory."""
    extensions = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}  # Use set for faster lookups
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            yield file_path