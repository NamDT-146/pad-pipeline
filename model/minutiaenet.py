import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class MinutiaeNetFeatureExtractor(nn.Module):
    """
    MinutiaeNet feature extractor for fingerprint matching.
    Converts the TensorFlow/Keras MinutiaeNet to PyTorch for feature extraction.
    """
    
    def __init__(self, input_shape=(400, 400, 1), embedding_size=512, pretrained_path=None):
        super(MinutiaeNetFeatureExtractor, self).__init__()
        
        self.input_shape = input_shape
        self.embedding_size = embedding_size
        
        # Encoder layers (simplified version of MinutiaeNet)
        # Block 1
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Feature extraction layers
        self.feature_conv = nn.Conv2d(256, embedding_size, kernel_size=3, padding=1)
        self.feature_bn = nn.BatchNorm2d(embedding_size)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
    
    def forward(self, x):
        """
        Forward pass through MinutiaeNet feature extractor.
        
        Args:
            x: Input tensor of shape (B, 1, H, W) or (B, H, W, 1)
            
        Returns:
            Feature embeddings of shape (B, embedding_size)
        """
        # Handle different input formats
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        elif x.dim() == 4 and x.shape[1] != 1:
            if x.shape[3] == 1:
                x = x.permute(0, 3, 1, 2)  # (B, H, W, 1) -> (B, 1, H, W)
        
        # Normalize input
        x = x / 255.0
        
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        
        # Feature extraction
        x = F.relu(self.feature_bn(self.feature_conv(x)))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def load_pretrained_weights(self, weights_path):
        """
        Load pretrained weights from TensorFlow/Keras model.
        
        Args:
            weights_path: Path to pretrained weights file
        """
        try:
            from utils.weight_converter import load_minutiaenet_weights
            updated_model = load_minutiaenet_weights(self, weights_path)
            # Copy the loaded weights back to self
            self.load_state_dict(updated_model.state_dict())
            print(f"Successfully loaded pretrained weights from {weights_path}")
        except ImportError:
            print(f"Warning: Could not import weight converter. Loading weights from {weights_path}")
            print("Continuing with random initialization...")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Continuing with random initialization...")

class MinutiaeNetSiamese(nn.Module):
    """
    Siamese network using MinutiaeNet as feature extractor.
    """
    
    def __init__(self, input_shape=(400, 400, 1), embedding_size=512, pretrained_path=None):
        super(MinutiaeNetSiamese, self).__init__()
        
        # MinutiaeNet feature extractor
        self.feature_extractor = MinutiaeNetFeatureExtractor(
            input_shape=input_shape,
            embedding_size=embedding_size,
            pretrained_path=pretrained_path
        )
        
        # Similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward_one(self, x):
        """Extract features from a single fingerprint"""
        return self.feature_extractor(x)
    
    def forward(self, x1, x2):
        """
        Process a pair of fingerprints and compute similarity score.
        
        Args:
            x1: First fingerprint image
            x2: Second fingerprint image
            
        Returns:
            Similarity score between 0 and 1
        """
        # Extract features for both images
        features1 = self.forward_one(x1)
        features2 = self.forward_one(x2)
        
        # Compute absolute difference
        diff = torch.abs(features1 - features2)
        
        # Compute similarity score
        similarity = self.similarity_net(diff)
        
        return similarity
    
    def extract_features(self, x):
        """Extract fingerprint features for a batch of images"""
        return self.forward_one(x)
    
    def get_feature_extractor(self):
        """Return the feature extraction part of the network"""
        return self.feature_extractor

def create_minutiaenet_model(device, pretrained_path=None, **kwargs):
    """
    Factory function to create and initialize a MinutiaeNetSiamese model.
    
    Args:
        device: torch.device for model placement
        pretrained_path: Path to pretrained MinutiaeNet weights
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized MinutiaeNetSiamese on the specified device
    """
    model = MinutiaeNetSiamese(pretrained_path=pretrained_path, **kwargs).to(device)
    return model 