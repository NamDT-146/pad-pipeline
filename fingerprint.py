import os
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from io import BytesIO

# Import the core functionality directly
from model import get_architecture
from dataset.preprocess import get_default_args 
from dataset.preprocess.preprocessing import create_fingerprint_transforms
from dataset.preprocess.enhancing import create_fingerprint_enhancement


class FingerprintMatcher:

    def __init__(self, database_path, model_path, model='siamese', device="cuda"):
        
        self.database_path = database_path
        self.device = device
        
        os.makedirs(os.path.dirname(os.path.abspath(database_path)), exist_ok=True)
        
        print(f"Loading model from {model_path}")
        self.model = get_architecture(model, device=device)
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        self.args = get_default_args(mode='test')
        
        # Initialize/load database
        self.database = self._load_or_create_database()
        print(f"Database initialized with {len(self.database)} fingerprints")
    
    def _load_or_create_database(self):
        try:
            database = torch.load(self.database_path, map_location=self.device)
            print(f"Loaded existing database from {self.database_path}")
            return database
        except FileNotFoundError:
            print(f"Database not found at {self.database_path}, creating new database")
            # Create empty database
            database = {}
            torch.save(database, self.database_path)
            return database
        except Exception as e:
            print(f"Error loading database: {e}. Creating new database.")
            database = {}
            torch.save(database, self.database_path)
            return database
    
    def register_fingerprint(self, fingerprint_data, subject_id):
        
        print(f"Registering fingerprint for subject {subject_id}")
        
        # Check if subject already exists
        if subject_id in self.database:
            return False, f"Subject {subject_id} already exists in database"
        
        try:
            # Convert to PIL Image if binary data
            if isinstance(fingerprint_data, bytes):
                image = Image.open(BytesIO(fingerprint_data)).convert('L')
            else:
                image = fingerprint_data.convert('L')
            
            # Process image
            preprocessed = self._preprocess_image(image)
            enhanced = self._enhance_image(preprocessed)
            features = self._extract_features(enhanced)
            
            # Add to database
            self.database[subject_id] = features
            
            # Save database
            torch.save(self.database, self.database_path)
            
            return True, f"Successfully registered fingerprint for subject {subject_id}"
            
        except Exception as e:
            return False, f"Error registering fingerprint: {e}"
    
    def verify_fingerprint(self, fingerprint_data, threshold=0.75):
        print(f"Verifying fingerprint with threshold {threshold}")
        
        if len(self.database) == 0:
            return "no_database", 0.0
        
        try:
            if isinstance(fingerprint_data, bytes):
                image = Image.open(BytesIO(fingerprint_data)).convert('L')
            else:
                image = fingerprint_data.convert('L')
            
            # Process image
            preprocessed = self._preprocess_image(image)
            enhanced = self._enhance_image(preprocessed)
            features = self._extract_features(enhanced)
            
            # Match against database
            return self._match_features(features, threshold)
            
        except Exception as e:
            print(f"Error verifying fingerprint: {e}")
            return "error", 0.0
    
    # ===================== SIMPLE FINGERPRINT COMPARISON =====================
    def compare_fingerprints(self, fingerprint_data1, fingerprint_data2, threshold=0.75):
        """
        Compare two fingerprint images and return (same_person: bool, score: float).
        Returns True if the similarity score >= threshold, else False.
        """
        try:
            # Convert to PIL Image if binary data
            if isinstance(fingerprint_data1, bytes):
                image1 = Image.open(BytesIO(fingerprint_data1)).convert('L')
            else:
                image1 = fingerprint_data1.convert('L')
            if isinstance(fingerprint_data2, bytes):
                image2 = Image.open(BytesIO(fingerprint_data2)).convert('L')
            else:
                image2 = fingerprint_data2.convert('L')
            # Process images
            pre1 = self._preprocess_image(image1)
            enh1 = self._enhance_image(pre1)
            feat1 = self._extract_features(enh1)
            pre2 = self._preprocess_image(image2)
            enh2 = self._enhance_image(pre2)
            feat2 = self._extract_features(enh2)
            # Compute similarity (same as in _match_features)
            diff = feat1 - feat2
            diff_squared = diff * diff
            sim_score = 1.0 - torch.sqrt(torch.sum(diff_squared)).item() / 2.0
            return sim_score >= threshold, sim_score
        except Exception as e:
            print(f"Error comparing fingerprints: {e}")
            return False, 0.0
    # ===================== END SIMPLE FINGERPRINT COMPARISON =====================
    
    def _preprocess_image(self, image):
        preprocessor = create_fingerprint_transforms(self.args)
        return preprocessor(image)
    
    def _enhance_image(self, image):
        enhancer = create_fingerprint_enhancement(self.args)
        
        # Apply enhancement
        if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
            enhanced_results = enhancer(image)
            return enhanced_results['enhanced']
        
        # Fallback to original image
        if isinstance(image, Image.Image):
            return np.array(image)
        return image
    
    def _extract_features(self, image):
        if isinstance(image, np.ndarray):
            # Convert to tensor if it's a numpy array
            if len(image.shape) == 2:
                # Add batch and channel dimensions
                image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
            elif len(image.shape) == 3 and image.shape[0] == 1:
                # Add batch dimension
                image = torch.from_numpy(image).float().unsqueeze(0)
            else:
                image = torch.from_numpy(image).float()
        
        with torch.no_grad():
            # Ensure image is on the correct device
            if isinstance(image, torch.Tensor):
                image = image.to(self.device)
                feature_vector = self.model.extract_features(image)
            else:
                # Fallback if image isn't a tensor
                raise ValueError(f"Expected tensor or numpy array, got {type(image)}")
        
        return feature_vector
    
    def _match_features(self, feature_vector, threshold=0.75):
        best_match = None
        best_score = 0.0
        
        for subject_id, reference_features in self.database.items():
            diff = reference_features - feature_vector
            diff_squared = diff * diff
            sim_score = 1.0 - torch.sqrt(torch.sum(diff_squared)).item() / 2.0
            
            if sim_score > best_score:
                best_score = sim_score
                best_match = subject_id
        
        if best_score >= threshold:
            return best_match, best_score
        else:
            return "unrecognized", best_score

    def get_all_subjects(self):
        return list(self.database.keys())
    
    def remove_subject(self, subject_id):
        if subject_id in self.database:
            del self.database[subject_id]
            torch.save(self.database, self.database_path)
            return True, f"Subject {subject_id} removed from database"
        return False, f"Subject {subject_id} not found in database"