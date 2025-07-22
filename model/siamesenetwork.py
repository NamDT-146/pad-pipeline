import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """
    Siamese network for fingerprint verification.
    Uses a shared CNN for feature extraction and a similarity network for matching.
    """
    def __init__(self, input_channels=1, base_filters=32, embedding_size=128):
        super(SiameseNetwork, self).__init__()
        
        # Feature extraction network
        self.feature_net = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(0.15),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(base_filters, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.LeakyReLU(0.15),
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(base_filters*2, embedding_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_size),
            nn.LeakyReLU(0.15),
            nn.Dropout(0.4),
            nn.AdaptiveAvgPool2d((1, 1)),  # Add this instead
            
            # # Block 4
            # nn.Conv2d(base_filters*4, embedding_size, kernel_size=3, padding=1),
            # nn.BatchNorm2d(embedding_size),
            # nn.LeakyReLU(0.15),
            # nn.Dropout(0.4),
            # nn.AdaptiveAvgPool2d((1, 1)),  # Add this instead
            
            # # Block 5
            # nn.Conv2d(embedding_size, embedding_size, kernel_size=3, padding=1),
            # nn.BatchNorm2d(embedding_size),
            # nn.LeakyReLU(0.15),
            # nn.Dropout(0.4),
            # nn.AdaptiveAvgPool2d((1, 1)),  # Add this instead
            # nn.MaxPool2d(2),

            # # Block 6
            # nn.Conv2d(embedding_size, embedding_size, kernel_size=3, padding=1),
            # nn.BatchNorm2d(embedding_size),
            # nn.SiLU(),
            # nn.Dropout(0.4),
            # nn.MaxPool2d(2),

            # # Block 7
            # nn.Conv2d(embedding_size, embedding_size, kernel_size=3, padding=1),
            # nn.BatchNorm2d(embedding_size),
            # nn.SiLU(),
            # nn.Dropout(0.4),
            # nn.AdaptiveAvgPool2d((1, 1)),  # Add this instead
        )

        # Similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(embedding_size, 16),
            nn.LeakyReLU(0.15),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def unit_normalize(self, x):
        """Normalize feature vectors to unit length"""
        return F.normalize(x, p=2, dim=1)
    
    def forward_one(self, x):
        """Process a single input through the feature extraction network"""
        # Reshape to [B, C, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[3] == 1:
            x = x.permute(0, 3, 1, 2)
            
        # Extract features
        x = self.feature_net(x)
        x = torch.flatten(x, 1)
        x = self.unit_normalize(x)
        return x
        
    def forward(self, x1, x2):
        """
        Process a pair of inputs and compute similarity score.
        
        Args:
            x1: First fingerprint image
            x2: Second fingerprint image
            
        Returns:
            Similarity score between 0 and 1 (1 = same finger, 0 = different)
        """
        # Get embeddings for both images
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        
        # Compute squared difference
        diff = out1 - out2
        diff = diff * diff  # Element-wise square
        
        # Compute similarity score
        score = self.similarity_net(diff)
        return score

    def get_feature_extractor(self):
        """
        Returns the feature extraction part of the network.
        Useful for transfer learning or feature extraction.
        """
        return self.feature_net
    
    def extract_features(self, x):
        """
        Extract fingerprint features for a batch of images.
        Useful for embedding generation or retrieval.
        """
        return self.forward_one(x)

# Add a file with metrics utility functions
def create_siamese_model(device):
    """
    Factory function to create and initialize a SiameseNetwork.
    
    Args:
        device: torch.device for model placement
        
    Returns:
        Initialized SiameseNetwork on the specified device
    """
    model = SiameseNetwork().to(device)
    return model